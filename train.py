#!/usr/bin/env python
"""Training script for the models module
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: June 16 2021
"""
import logging
import os
import time

import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import EchoNetDataset
from models import EchoNet


@click.command()
@click.option(
    "--net",
    default="echonet",
    help="Model backbone",
)
@click.option(
    "--lr1",
    default=1e-4,
    help="Learning rate for the branch involved UNet and ResNet-18",
)
@click.option("--lr2", default=1e-4, help="Learning rate for LSTM")
@click.option(
    "--batch_size", default=16, help="Number of instances per training/validation batch"
)
@click.option("--epochs", default=50, help="Number of epochs")
@click.option(
    "--use_gt_ef",
    is_flag=True,
    help="Whether to use ground truth EF to compute loss",
)
@click.option(
    "--log_every", default=200, help="Number of iterations to report training progress"
)
@click.option(
    "--device",
    default="cuda",
    type=click.Choice(["cuda", "cpu"]),
    help="Device to train the model",
)
@click.option("--load", default=None, help="Load model checkpoint")
def train_net(net, lr1, lr2, batch_size, epochs, use_gt_ef, log_every, device, load):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Using device {device}")

    # ====================================
    # Model
    # ====================================
    if net == "echonet":
        model = EchoNet(device).to(device)
        if load != None:
            logging.info(f"Loading checkpoint file:{load}")
            checkpoint_path = os.path.join(checkpoint_dir, load)
            model.load_state_dict(torch.load(checkpoint_path))

    # ====================================
    # Data
    # ====================================
    trainset = EchoNetDataset(
        root_dir=root_dir,
        split="train",
    )
    valset = EchoNetDataset(
        root_dir=root_dir,
        split="val",
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size)
    dataloaders = {"train": trainloader, "val": valloader}

    # ====================================
    # Losses
    # ====================================
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_smoothl1 = nn.SmoothL1Loss()

    # ====================================
    # Optimizers
    # ====================================
    optimizer1 = optim.SGD(
        list(model.unet.parameters())
        + list(model.encoder.parameters())
        + list(model.regressor.parameters()),
        lr=lr1,
        momentum=0.9,
        weight_decay=1e-4,
    )
    optimizer2 = optim.SGD(
        model.decoder.parameters(), lr=lr2, momentum=0.9, weight_decay=1e-4
    )

    # ====================================
    # Training loop
    # ====================================
    flaot_max = torch.finfo(torch.float32).max
    best_val_loss = flaot_max
    best_loss_seg = flaot_max
    best_loss_volume = flaot_max
    best_loss_ef = flaot_max
    N_val = len(valloader.dataset)

    iterations = 0
    since = time.time()
    for epoch in range(epochs):
        logging.info("Epoch {}/{}".format(epoch, epochs - 1))
        logging.info("-" * 20)

        for phase in ("train", "val"):
            running_loss_seg = 0.0
            running_loss_volume = 0.0
            running_loss_ef = 0.0
            within_epoch_interations = 0

            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, labels, masks, contains_edv_esv_idxes in dataloaders[phase]:
                video_tensor = inputs[0].to(device)
                input_frames = torch.cat(inputs[1:]).to(device)
                volumes = torch.cat(labels[1:]).unsqueeze(1).float().to(device)
                masks = torch.cat(masks).float().to(device)
                efs = labels[0].unsqueeze(1).float().to(device)  # Omit gt ef

                if phase == "train":
                    iterations += 1
                    within_epoch_interations += 1

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                with torch.set_grad_enabled(phase == "train"):

                    masks_pred, volumes_pred = model(input_frames, goal="mask&volume")
                    efs_pred = model(video_tensor, goal="ef")
                    loss_seg = criterion_bce(masks_pred, masks)
                    loss_volume = criterion_smoothl1(volumes_pred, volumes)

                    if phase == "train":
                        loss_seg.backward(retain_graph=True)
                        loss_volume.backward()
                        optimizer1.step()
                        optimizer1.zero_grad()

                    pseudo_efs = model._get_pseudo_ef(
                        video_tensor
                    )  # Weekly supervision of EF
                    if use_gt_ef:
                        idx = torch.where(contains_edv_esv_idxes == 1)
                        pseudo_efs[idx] = efs[idx]
                    loss_ef = criterion_smoothl1(efs_pred, pseudo_efs)

                    if phase == "train":
                        loss_ef.backward()
                        optimizer2.step()

                running_loss_seg += loss_seg.item() * 2 * batch_size
                running_loss_volume += loss_volume.item() * 2 * batch_size
                running_loss_ef += loss_ef.item() * batch_size

                if not iterations % log_every and phase == "train":
                    iter_elapsed = time.time() - since
                    seg_loss = running_loss_seg / (
                        within_epoch_interations * 2 * batch_size
                    )
                    v_loss = running_loss_volume / (
                        within_epoch_interations * 2 * batch_size
                    )
                    ef_loss = running_loss_ef / (within_epoch_interations * batch_size)
                    train_loss = seg_loss + v_loss + ef_loss
                    logging.info(
                        (
                            f"[{iter_elapsed//60:>3.0f}m {iter_elapsed%60:2.0f}s] "
                            f"Iteration: {iterations:>4.0f} | "
                            f"{phase}\n"
                            f"\tTotal loss: {train_loss:.5f}\n"
                            f"\tSegmentation BCE Loss: {seg_loss:.5f}\n"
                            f"\tLoss(v): {v_loss:.5f}\n"
                            f"\tLoss(ef): {ef_loss:.5f}"
                        )
                    )

            if phase == "val":
                seg_loss = running_loss_seg / (N_val * 2)
                v_loss = running_loss_volume / (N_val * 2)
                ef_loss = running_loss_ef / N_val
                val_loss = seg_loss + v_loss + ef_loss
                logging.info(
                    (
                        "\n"
                        f"Total {phase} loss: {val_loss:.5f} | "
                        f"Segmentation BCE Loss: {seg_loss:.5f} | "
                        f"Loss(v): {v_loss:.5f} | "
                        f"Loss(ef): {v_loss:.5f}\n"
                    )
                )
                if val_loss < best_val_loss:
                    if epoch != 0:
                        logging.info(
                            f"Loss improved from {best_val_loss:.5f} to {val_loss:.5f}. Saving model checkpoint to {save_path}"
                        )
                    else:
                        logging.info(
                            f"Initial loss: {val_loss:.5f}. Saving model checkpoint to {save_path}"
                        )
                    torch.save(model.state_dict(), save_path)
                    best_val_loss = val_loss
                    best_loss_seg = running_loss_seg / (N_val * 2)
                    best_loss_volume = running_loss_volume / (N_val * 2)
                    best_loss_ef = running_loss_ef / (N_val)

    time_elapsed = time.time() - since
    logging.info(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    logging.info(
        (
            f"Best validation loss: {best_val_loss:.5f} => "
            f"Segmentation BCE Loss.: {best_loss_seg:.5f} | Loss(v): {best_loss_volume:.5f} | Loss(ef): {best_loss_ef:.5f}"
        )
    )


if __name__ == "__main__":

    root_dir = "/home/tienyu/data/EchoNet-Dynamic"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_checkpoint_name = "echonetv2.pt"
    save_path = os.path.join(checkpoint_dir, model_checkpoint_name)

    train_net()

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from datasets import EchoNetDataset
from models import EchoNet

video_dir = "/home/tienyu/data/EchoNet-Dynamic/Videos"
target_csv = "/home/tienyu/data/EchoNet-Dynamic/FileList.csv"

batch_size = 16
num_epochs = 50
log_every = 200
lr = 1e-4


# Transformation for frames -> grayscale images
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
    ]
)

# Input management
trainset = EchoNetDataset(
    video_dir=video_dir,
    target_csv=target_csv,
    split="train",
    transform=transform,
)

valset = EchoNetDataset(
    video_dir=video_dir,
    target_csv=target_csv,
    split="val",
    transform=transform,
)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size)
dataloaders = {"train": trainloader, "val": valloader}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = EchoNet(device)
model = model.to("cuda")

# Metric
criterion_bce = nn.BCEWithLogitsLoss()
criterion_smoothl1 = nn.SmoothL1Loss()

# Optimization Setting
optimizer1 = optim.SGD(
    list(model.unet.parameters())
    + list(model.encoder.parameters())
    + list(model.regressor.parameters()),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4,
)
optimizer2 = optim.SGD(
    model.decoder.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
)


best_val_loss = torch.finfo(torch.float32).max
best_loss_seg = torch.finfo(torch.float32).max
best_loss_volume = torch.finfo(torch.float32).max
best_loss_ef = torch.finfo(torch.float32).max
N_val = len(valloader.dataset)

iterations = 0
since = time.time()
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 20)

    for phase in ("train", "val"):
        running_loss_seg = 0.0
        running_loss_volume = 0.0
        running_loss_ef = 0.0
        within_epoch_interations = 0

        if phase == "train":
            model.train()
        else:
            model.eval()
            print()

        for inputs, labels, masks, contains_edv_esv_idxes in dataloaders[phase]:
            video_tensor = inputs[0].to(device)
            input_frames = torch.cat(inputs[1:]).to(device)
            volumes = torch.cat(labels[1:]).unsqueeze(1).float().to(device)
            masks = torch.cat(masks).float().to(device)
            # efs = labels[0].unsqueeze(1).float().to(device) # Omit gt ef

            if phase == "train":
                iterations += 1
                within_epoch_interations += 1

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            with torch.set_grad_enabled(phase == "train"):

                masks_pred, volumes_pred = model(input_frames, goal="mask&volume")
                efs_pred = model(video_tensor, goal="ef")
                # Compute losses
                loss_seg = criterion_bce(masks_pred, masks)
                loss_volume = criterion_smoothl1(volumes_pred, volumes)

                if phase == "train":
                    loss_seg.backward(retain_graph=True)
                    loss_volume.backward()
                    optimizer1.step()
                    optimizer1.zero_grad()

                efs = model._get_pseudo_ef(video_tensor)  # Weekly supervision of ef
                loss_ef = criterion_smoothl1(efs_pred, efs)

                if phase == "train":
                    loss_ef.backward()
                    optimizer2.step()

            running_loss_seg += loss_seg.item() * 2 * batch_size
            running_loss_volume += loss_volume.item() * 2 * batch_size
            running_loss_ef += loss_ef.item() * batch_size

            if not iterations % log_every and phase == "train":
                iter_elapsed = time.time() - since
                print(
                    (
                        f"[{iter_elapsed//60:>3.0f}m {iter_elapsed%60:2.0f}s] "
                        f"Iteration: {iterations:>4.0f} | "
                        f"{phase.title()} | "
                        f"Segmentation BCE Loss: {running_loss_seg/(within_epoch_interations * 2 * batch_size):.5f} "
                        f"Loss(v): {running_loss_volume/(within_epoch_interations * 2 * batch_size):.5f} "
                        f"Loss(ef): {running_loss_ef/(within_epoch_interations * batch_size):.5f}"
                    )
                )

        if phase == "val":
            val_loss = (
                (running_loss_seg / (N_val * 2))
                + (running_loss_volume / (N_val * 2))
                + (running_loss_ef / N_val)
            )
            print(
                (
                    f"Total {phase.title()} loss: {val_loss:.5f} | "
                    f"Segmentation BCE Loss: {running_loss_seg/(N_val * 2):.5f} "
                    f"Loss(v): {running_loss_volume/(N_val * 2):.5f} "
                    f"Loss(ef): {running_loss_ef/(N_val):.5f}"
                )
            )
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), "checkpoints/best.pt")
                best_val_loss = val_loss
                best_loss_seg = running_loss_seg / (N_val * 2)
                best_loss_volume = running_loss_volume / (N_val * 2)
                best_loss_ef = running_loss_ef / (N_val)

    print()
time_elapsed = time.time() - since
print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print(
    (
        f"Best validation loss: {best_val_loss:.5f}\n"
        f"(Seg.: {best_loss_seg:.5f} | v: {best_loss_volume:.5f} | ef: {best_loss_ef:.5f})"
    )
)

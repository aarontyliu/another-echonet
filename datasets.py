#!/usr/bin/env python
"""Data module to load EchoNet-Dynamic (https://echonet.github.io/dynamic/) database
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: June 16 2021
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.draw import polygon
from torch.utils.data import Dataset
from torchvision.io import read_video
from tqdm import tqdm

IMG_SIZE = 112

# Transformation for frames -> default: grayscale
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
    ]
)


class EchoNetDataset(Dataset):
    """EchoNet dataset."""

    def __init__(
        self,
        root_dir,
        split="train",
        sampling_frequency=4,
        clip_length=16,
        transform=transform,
    ):
        """
        Args:
            root_dir (string): Path to the directory of EchoNet Dynamic database
            split (string): Split of the database
            sampling_frequency (int): Sampling frequency for temporal sub-sampling
            clip_length (int): Number to determine the length of sub-sampled clip
            transform (callable, optional): Optional transform to be applied on sample video
        """
        self.root_dir = root_dir
        self.valid_splits = ("train", "val", "test")
        assert (
            split in self.valid_splits
        ), "Please verify the split specification! It should be in ('train', 'val', 'test')"
        self.split = split.upper()
        self.sampling_frequency = sampling_frequency
        self.clip_length = clip_length
        self.video_dir = os.path.join(self.root_dir, "Videos")
        self.filelist = pd.read_csv(os.path.join(self.root_dir, "FileList.csv"))
        self.tracing = pd.read_csv(os.path.join(self.root_dir, "VolumeTracings.csv"))

        # Remove clip that has no annotation in the volume tracing file
        valid_clips = self.tracing.FileName.str.replace(
            ".avi", "", regex=False
        ).unique()
        valid_idx = np.isin(self.filelist.FileName, valid_clips)
        self.filelist = self.filelist[valid_idx]
        # Remove clip that has too few frames for temporal sub-sampling
        self.min_num_frames = self.clip_length * self.sampling_frequency
        self.filelist = self.filelist.loc[
            (self.filelist["Split"] == self.split)
            & (self.filelist["NumberOfFrames"] > self.min_num_frames)
        ]
        self.transform = transform

        # Generate masks using polygons from 20 bounding boxes in the volume tracing file
        os.makedirs(os.path.join(self.root_dir, "masks"), exist_ok=True)
        self.masks_npy = os.path.join(
            self.root_dir, "masks", f"masks_{self.split.lower()}.npy"
        )
        self.mask_frames_npy = os.path.join(
            self.root_dir, "masks", f"mask_frames_{self.split.lower()}.npy"
        )

        # Save masks for fast loading
        if not (
            os.path.exists(self.masks_npy) and os.path.exists(self.mask_frames_npy)
        ):
            self.masks = defaultdict(list)
            self.mask_frames = defaultdict(list)
            grouped_tracing = self.tracing.groupby("FileName")
            for k in tqdm(
                self.filelist.FileName,
                total=len(self.filelist.FileName),
                desc=f"Saving masks ...",
            ):
                g = grouped_tracing.get_group(k + ".avi").groupby("Frame")
                fs = g.Frame.unique()
                if len(g) == 2:
                    for f in fs:
                        frame_num = f.item()
                        x1, y1, x2, y2 = (
                            g.get_group(frame_num)[["X1", "Y1", "X2", "Y2"]]
                            .values[1:]
                            .T
                        )
                        mask = self.get_mask(x1, y1, x2, y2)
                        self.masks[k].append(mask)
                        self.mask_frames[k].append(frame_num)
                    if self.masks[k][1].sum() > self.masks[k][0].sum():
                        self.masks[k][0], self.masks[k][1] = (
                            self.masks[k][1],
                            self.masks[k][0],
                        )
                        self.mask_frames[k][0], self.mask_frames[k][1] = (
                            self.mask_frames[k][1],
                            self.mask_frames[k][0],
                        )
            np.save(self.masks_npy, self.masks)
            np.save(self.mask_frames_npy, self.mask_frames)
        else:
            # Load saved masks and indexes of annotated frames
            self.masks = np.load(self.masks_npy, allow_pickle=True)[()]
            self.mask_frames = np.load(self.mask_frames_npy, allow_pickle=True)[()]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_name = self.filelist.FileName.iloc[idx]
        video_path = os.path.join(self.video_dir, video_name) + ".avi"

        # Load EDV and ESV masks
        edv_mask, esv_mask = self.masks[video_name]
        frame_idx_edv, frame_idx_esv = self.mask_frames[video_name]

        # Temporal sub-sampling
        s = np.random.choice(
            self.filelist.NumberOfFrames.iloc[idx] - self.min_num_frames
        )
        sampled_index = list(range(s, s + self.min_num_frames, 4))

        # Indicate whether the sub-sampled clip contains ground truth EF
        contains_edv_esv_idxes = (frame_idx_esv in sampled_index) and (
            frame_idx_edv in sampled_index
        )

        # Read video and convert it from uint8[0,255] to float[0,1]
        video = read_video(video_path, pts_unit="sec")[0]
        video = video.permute(0, 3, 1, 2) / 255.0

        if self.transform:
            video = self.transform(video)

        # Check spatial size (IMG_SIZE x IMG_SIZE)
        if (video.shape[-1] != IMG_SIZE) or (video.shape[-2] != IMG_SIZE):
            video = F.interpolate(video, (IMG_SIZE, IMG_SIZE))

        video_tensor, edv_frame, esv_frame = (
            video[sampled_index],
            video[frame_idx_edv],
            video[frame_idx_esv],
        )

        # Extract EF, ESV, EDV measures from filelist
        ef, esv, edv = self.filelist.iloc[idx, 1:4]

        return (
            (video_tensor, edv_frame, esv_frame),
            (ef, edv, esv),
            (edv_mask, esv_mask),
            contains_edv_esv_idxes,
        )

    def get_mask(self, x1, y1, x2, y2):
        """Generate polygon from bounding boxes"""
        x = np.concatenate((x1, x2[::-1]))
        y = np.concatenate((y1, y2[::-1]))
        r, c = polygon(np.rint(y), np.rint(x), (IMG_SIZE, IMG_SIZE))
        mask = np.zeros((IMG_SIZE, IMG_SIZE))
        mask[r, c] = 1
        mask = mask[np.newaxis, :]  # Add channel dim

        return mask

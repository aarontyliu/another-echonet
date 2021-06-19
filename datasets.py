import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision
from skimage.draw import polygon
from torch.utils.data import Dataset
from torchvision.io import read_video
from tqdm import tqdm


class EchoNetDataset(Dataset):
    """EchoNet dataset."""

    def __init__(
        self,
        target_csv,
        video_dir,
        split="train",
        transform=None,
        sampling_frequency=4,
        clip_length=16,
    ):
        """
        Args:
            target_csv (string): Path to the csv file with annotations.
            video_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert split in (
            "train",
            "val",
            "test",
        ), "Please validate the split specification (train, val or test)"
        self.tracing = pd.read_csv(
            "/home/tienyu/data/EchoNet-Dynamic/VolumeTracings.csv"
        )
        valid_clips = self.tracing.FileName.str.replace(".avi", "").unique()

        self.split = split.upper()
        self.sampling_frequency = sampling_frequency
        self.clip_length = clip_length
        self.min_num_frames = self.clip_length * self.sampling_frequency
        self.df = pd.read_csv(target_csv)
        valid_idx = np.isin(self.df.FileName, valid_clips)
        self.df = self.df[valid_idx]
        self.df = self.df.loc[
            (self.df["Split"] == self.split)
            & (self.df["NumberOfFrames"] > self.min_num_frames)
        ]
        self.video_dir = video_dir
        self.transform = transform

        self.masks = defaultdict(list)
        self.mask_frames = defaultdict(list)
        grouped_tracing = self.tracing.groupby("FileName")
        for k in tqdm(
            self.df.FileName, total=len(self.df.FileName), desc="Fetching masks"
        ):
            g = grouped_tracing.get_group(k + ".avi").groupby("Frame")
            fs = g.Frame.unique()
            if len(g) == 2:
                for f in fs:
                    frame_num = f.item()
                    x1, y1, x2, y2 = (
                        g.get_group(frame_num)[["X1", "Y1", "X2", "Y2"]].values[1:].T
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

            assert (
                self.masks[k][0].sum() >= self.masks[k][1].sum()
            ), f"invalid video found: {k}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_name = self.df.iloc[idx, 0]
        video_path = os.path.join(self.video_dir, video_name) + ".avi"

        edv_mask, esv_mask = self.masks[video_name]
        frame_idx_edv, frame_idx_esv = self.mask_frames[video_name]

        s = np.random.choice(self.df.NumberOfFrames.iloc[idx] - self.min_num_frames)
        sampled_index = list(range(s, s + self.min_num_frames, 4))

        contains_edv_esv_idxes = (frame_idx_esv in sampled_index) and (
            frame_idx_edv in sampled_index
        )

        video = read_video(video_path, pts_unit="sec")[0]
        video = video.permute(0, 3, 1, 2) / 255.0

        if self.transform:
            video = self.transform(video)

        video_tensor, edv_frame, esv_frame = (
            video[sampled_index],
            video[frame_idx_edv],
            video[frame_idx_esv],
        )

        ef, esv, edv = self.df.iloc[idx, 1:4]

        return (
            (video_tensor, edv_frame, esv_frame),
            (ef, edv, esv),
            (edv_mask, esv_mask),
            contains_edv_esv_idxes,
        )

    def get_mask(self, x1, y1, x2, y2):
        x = np.concatenate((x1, x2[::-1]))
        y = np.concatenate((y1, y2[::-1]))
        r, c = polygon(np.rint(y), np.rint(x), (112, 112))
        mask = np.zeros((112, 112))
        mask[r, c] = 1
        mask = mask[np.newaxis, :]  # Add channel dim

        return mask

import os

import av
import pandas as pd
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset


class EchoNetDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        target_csv,
        root_dir,
        split="train",
        transform=None,
        frame_size=112,
        sampling_frequency=4,
        clip_length=16,
    ):
        """
        Args:
            target_csv (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert split in (
            "train",
            "val",
            "test",
        ), "Please validate the split specification (train, val or test)"
        self.split = split.upper()
        self.sampling_frequency = sampling_frequency
        self.clip_length = clip_length
        self.min_num_frames = self.clip_length * self.sampling_frequency
        self.df = pd.read_csv(target_csv)
        self.df = self.df.loc[
            (self.df["Split"] == self.split)
            & (self.df["NumberOfFrames"] > self.min_num_frames)
            # & (self.df['FrameHeight'] == 112)
            # & (self.df['FrameWidth'] == 112)
        ]
        # self.max_num_frames = self.df.NumberOfFrames.max()
        self.root_dir = root_dir
        self.transform = transform
        self.frame_size = frame_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_name = os.path.join(self.root_dir, self.df.iloc[idx, 0]) + ".avi"
        nof = self.df.NumberOfFrames.iloc[idx]
        candidates = nof - self.min_num_frames
        if self.split == "TRAIN":
            start_frame = np.random.choice(candidates)
        else:
            start_frame = 0
        end_frame = start_frame + self.min_num_frames
        sampled_index = np.arange(nof)[start_frame:end_frame:4]

        container = av.open(video_name)
        frames = []
        i = 0
        for j, frame in enumerate(container.decode(video=0)):
            if sampled_index[i] == j:
                f = frame.to_image()
                if self.transform:
                    f = self.transform(f)
                frames.append(f)
                i += 1
            if i == self.clip_length:
                break
        video_tensor = torch.stack(frames)

        ef, esv, edv = self.df.iloc[idx, 1:4]

        return video_tensor, (ef, esv, edv)

#!/usr/bin/env python
"""Simple UI to test the trained model on videos / or sampled test videos
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: June 16 2021
"""

import os
import time

import easygui as g
import numpy as np
import pandas as pd
import torch

from models import EchoNet
from utils import (
    get_masked_video_tensor,
    grayscale_tensor,
    video_2_masks_and_volumes,
    write_video,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "checkpoints/baseline.pt"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

video_path = "/home/tienyu/data/EchoNet-Dynamic/Videos/"
filelist_dir = "/home/tienyu/data/EchoNet-Dynamic/FileList.csv"
tracing_dir = "/home/tienyu/data/EchoNet-Dynamic/VolumeTracings.csv"

model = EchoNet(device).to(device)
model.load_state_dict(torch.load(checkpoint_dir))

model = EchoNet(device).to(device)
model.load_state_dict(torch.load(checkpoint_dir))


def generate_video(filename, clip_path, model, fps, ekg=True, device=device):
    """Generate output video"""
    since = time.time()
    grayscale_clip = grayscale_tensor(clip_path)
    grayscale_clip = grayscale_clip.to(device)
    masks, volumnes = video_2_masks_and_volumes(grayscale_clip, model)
    masks = masks.to(device)
    masked = get_masked_video_tensor(grayscale_clip, masks)
    write_video(f"{filename}", masked, fps, volumnes, ekg=ekg)
    time_elapsed = time.time() - since
    print(f"Time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")


def filename_handler(filename, from_dir=False):
    """Simple UI to play around the trained model"""
    bye = f"You didn't select a file :( Bye bye."
    if from_dir:
        if filename is not None:
            filename = filename.split("/")[-1]
            g.msgbox(f"You chose: {filename} ", "Selection result:")
            return filename
        else:
            g.msgbox(bye, "Selection result:")
    else:
        if filename is not None:
            g.msgbox(f"You chose: {filename} ", "Selection result:")
            return filename
        else:
            g.msgbox(bye, "Selection result:")


filelist = pd.read_csv(filelist_dir)
tracing = pd.read_csv(tracing_dir)
valid_clips = tracing.FileName.str.replace(".avi", "").unique()
test_filelist = filelist[filelist.Split == "TEST"]
valid_idx = np.isin(test_filelist.FileName, valid_clips)
test_filelist = test_filelist[valid_idx]
test_filelist.FileName = test_filelist.FileName.str[:] + ".avi"
sample_test_filenames = test_filelist.FileName.sample(6).tolist()


g.msgbox("Hello :) I am here to help you generate a video with the EchoNet!")

if g.ynbox("Do you want to select from video directory?"):
    filename = g.fileopenbox(default=video_path, filetypes="*.avi")
    filename = filename_handler(filename, from_dir=True)
    if filename is not None:
        g.msgbox(f"Generating output video...")
        clip_path = os.path.join(video_path, filename)
        fps = (
            filelist.FPS[filelist.FileName == filename.replace(".avi", "")]
            .astype(float)
            .item()
        )
        generate_video(filename, clip_path, model, fps)

else:
    msg = "Which file you want to proceed?"
    title = "Sample test files"
    filename = g.choicebox(msg, title, sample_test_filenames)
    filename = filename_handler(filename)
    if filename is not None:
        g.msgbox(f"Generating output video...")
        clip_path = os.path.join(video_path, filename)
        fps = (
            filelist.FPS[filelist.FileName == filename.replace(".avi", "")]
            .astype(float)
            .item()
        )
        generate_video(filename, clip_path, model, fps)

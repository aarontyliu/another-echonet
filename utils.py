import os

import easygui as g
import numpy as np
import torch
import torchvision.transforms as transforms
from imageio import mimwrite
from PIL import Image, ImageDraw
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader
from torchvision.io import read_video
from tqdm import tqdm


def grayscale_tensor(clip_path: str):
    video = read_video(clip_path, pts_unit="sec")[0].permute(0, 3, 1, 2) / 255.0
    grayscaled = transforms.Grayscale(num_output_channels=1)(video)

    return grayscaled


def video_2_masks_and_volumes(grayscaled, model, batch_size=32):
    masks, volumes = [], []
    model.eval()
    with torch.no_grad():
        loader = DataLoader(grayscaled, batch_size=batch_size)
        for vs in tqdm(
            loader, desc="Predicting batch masks and volumes in...", total=len(loader)
        ):
            m, x = model(vs)
            masks.append(m)
            volumes.append(x)
    masks = torch.cat(masks).repeat(1, 3, 1, 1)
    masks = (torch.sigmoid(masks)).float()
    volumes = torch.flatten(torch.cat(volumes))

    return masks, volumes


def get_masked_video_tensor(video_tensor, masks, color="blue", strict=True):
    valid_colors = ("red", "green", "blue")
    assert color in valid_colors, f"color arg needs to be in {valid_colors}!"
    palette = {"red": 0, "green": 1, "blue": 2}
    if strict:
        masks = (masks > 0.5).float()

    # color masks
    canvas = torch.zeros(masks.size()).to(masks.device)
    canvas[:, palette[color], :, :] = masks[:, palette[color], :, :]
    eps = torch.finfo(canvas.dtype).eps
    masked = torch.clip(canvas + video_tensor, min=eps, max=1 - eps)

    return masked


def write_video(
    file_name, masked_video_tensor, fps, volumes=None, ekg=True, output_dir="output"
):
    masked_video_tensor = masked_video_tensor.detach().cpu()
    assert isinstance(fps, float)
    if volumes is not None:
        assert len(masked_video_tensor) == len(volumes)
        volumes = volumes.detach().cpu().numpy()
    frames = minmax_scale(np.arange(len(masked_video_tensor)), feature_range=(5, 100))
    if ekg:
        scaled = minmax_scale(-volumes, feature_range=(10, 30))
    output = []
    for i, m in enumerate([transforms.ToPILImage()(m) for m in masked_video_tensor]):
        canvas = Image.new("RGB", (144, 144))
        canvas.paste(m, (0, 0))
        draw = ImageDraw.Draw(canvas)
        if ekg:
            annot = list(zip(frames[:i], scaled[:i] + 110))
            draw.line(annot, joint="curve", fill="green")
        output.append(canvas)
    output = np.stack(output)
    mimwrite(os.path.join(output_dir, file_name), output, fps=fps)
    g.msgbox(
        f"Completed! Output video is saved to {os.path.join(output_dir, file_name)}"
    )

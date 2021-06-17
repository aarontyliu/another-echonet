import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

from datasets import EchoNetDataset
from models import EchoNetRegressor

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = "/home/tienyu/data/EchoNet-Dynamic/Videos"
target_csv = "/home/tienyu/data/EchoNet-Dynamic/FileList.csv"

batch_size = 16
num_epochs = 30
log_every = 200
lr = 1e-4 

# Transformation for frames
transform = transforms.Compose(
    [
        # transforms.Resize((112, 112), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        
    ]
)

# Input management
trainset = EchoNetDataset(
    root_dir=root_dir,
    target_csv=target_csv,
    split="train",
    transform=transform,
)

valset = EchoNetDataset(
    root_dir=root_dir,
    target_csv=target_csv,
    split="val",
    transform=transform,
)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size)
dataloaders = {"train": trainloader, "val": valloader}

# Model
model = EchoNetRegressor()
model = model.to(device)

# Metric
criterion = nn.MSELoss()

# Optimization Setting
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

best_val_loss = torch.finfo(torch.float32).max

iterations = 0
since = time.time()
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 20)

    for phase in ("train", "val"):
        running_loss = 0.0
        within_epoch_interations = 0

        if phase == "train":
            model.train()
        else:
            model.eval()
            print()

        for video_tensor, labels in dataloaders[phase]:
            if phase == "train":
                iterations += 1
                within_epoch_interations += 1
            optimizer.zero_grad()

            ef, _, _ = [l.float().to(device) for l in labels]
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(video_tensor)

                ef_pred = outputs[:,0]
                
                # Compute MSE losses
                loss = criterion(ef_pred, ef)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * video_tensor.size(0)

            if not iterations % log_every and phase == "train":
                iter_elapsed = time.time() - since
                print(
                    (
                        f"[{iter_elapsed//60:>3.0f}m {iter_elapsed%60:2.0f}s] "
                        f"Iteration: {iterations:>4.0f} | {phase.title()} Loss: {running_loss/(within_epoch_interations*batch_size):.5f}"
                    )
                )

        if phase == "val":
            val_loss = running_loss / len(dataloaders[phase].dataset)
            print(
                (
                    f"{phase.title()} Loss: {val_loss:.5f}"
                )
            )
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), "checkpoints/best_checkpoint.pt")
                best_val_loss = val_loss

        else:
            scheduler.step()

    print()
time_elapsed = time.time() - since
print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
print(
    (
        f"Best validation loss: {best_val_loss:.5f}\n"
        f"Best ESV: {best_mse_esv:.5f}\n"
        f"Best EDV: {best_mse_edv:.5f}"
    )
)

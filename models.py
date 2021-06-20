import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from unet import UNet


class EchoNet(nn.Module):
    def __init__(self, device, encoder_hidden_dim=256):
        super(EchoNet, self).__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        # ====================================
        # Semantic segmentation network: UNet
        # ====================================
        self.device = device
        self.unet = UNet(n_channels=1, n_classes=1, bilinear=True)

        # ==================
        # Encoder: resnet18
        # ==================
        resnet18 = models.resnet18(pretrained=False)
        # Adapt to input channel (1)
        resnet18.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder = torch.nn.ModuleList(list(resnet18.children())[:-1])
        self.encoder.append(nn.Flatten())
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.regressor = torch.nn.Sequential(
            *[
                nn.Linear(resnet18.fc.in_features, self.encoder_hidden_dim),
                nn.BatchNorm1d(self.encoder_hidden_dim, momentum=0.01),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.encoder_hidden_dim, 1),
            ]
        )

        # ==================
        # Decoder: LSTM
        # ==================
        self.decoder = LSTMDecoder(
            input_dim=resnet18.fc.in_features,
            hidden_dim=256,
            num_layers=2,
            num_outputs=1,
            bidirectional=True,
            dropout_p=0.3,
        )

    def forward(self, x, goal="mask&volume"):
        assert goal in (
            "mask&volume",
            "ef",
        ), "Please verify the task ('mask&volume' / 'ef')!"

        if goal == "mask&volume":
            assert x.dim() == 4, "Input needs to be (N x C x H x W)!"
            masks = self.unet(x)
            x = self.encoder(masks)
            x = self.regressor(x)

            return masks, x

        elif goal == "ef":
            assert x.dim() == 5, "Input needs to be (L x N x C x H x W)!"

            x = self.embed(x)
            x = self.decoder(x)

            return x

    def embed(self, x):
        embeddings = torch.stack(
            [
                torch.cat(
                    [
                        self.encoder(self.unet(b.to(self.device))).detach().cpu()
                        for b in DataLoader(v, batch_size=32)
                    ]
                )
                for v in x
            ],
            dim=1,
        )
        return embeddings.to(self.device)

    def _get_pseudo_ef(self, x):
        with torch.no_grad():
            vs = torch.stack(
                [self.regressor(k) for k in self.embed(x).permute(1, 0, 2)]
            ).squeeze(2)
            return 100.0 * (
                (vs.max(dim=1)[0] - vs.min(dim=1)[0]) / vs.max(dim=1)[0]
            ).unsqueeze(1)


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_dim=256,
        hidden_dim=128,
        num_layers=2,
        num_outputs=1,
        bidirectional=True,
        dropout_p=0.3,
    ):
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            bidirectional=self.bidirectional,
        )
        self.direction_factor = 2 if self.bidirectional else 1
        self.fc1 = nn.Linear(self.direction_factor * self.hidden_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_outputs)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.permute(1, 0, 2)
        x = x[:, -1, :]
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

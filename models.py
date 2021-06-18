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
        # Encoder: resnext50_32x4d
        # ==================
        resnext50_32x4d = models.resnext50_32x4d(pretrained=False)
        # Adapt to input channel (1)
        resnext50_32x4d.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder = torch.nn.ModuleList(list(resnext50_32x4d.children())[:-1])
        self.encoder.append(nn.Flatten())
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.regressor = torch.nn.Sequential(
            *[
                nn.Linear(resnext50_32x4d.fc.in_features, self.encoder_hidden_dim),
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
            input_dim=resnext50_32x4d.fc.in_features + 1,
            hidden_dim=256,
            num_layers=2,
            num_outputs=1,
            bidirectional=True,
            dropout_p=0.3,
        )

    def forward(self, x, goal="mask&volume", tf_masks=None):
        assert goal in (
            "mask&volume",
            "ef",
        ), "Please verify the task ('mask&volume' / 'ef')!"

        if goal == "mask&volume":
            assert x.dim() == 4, "Input needs to be (N x C x H x W)!"
            masks = self.unet(x)
            if tf_masks is not None:
                x = self.encoder(tf_masks)
            else:
                x = self.encoder(masks)
            x = self.regressor(x)

            return masks, x

        elif goal == "ef":
            assert x.dim() == 5, "Input needs to be (L x N x C x H x W)!"

            embeddings = self.embed(x)
            x = self.decoder(
                torch.cat([embeddings, self.aug_features(embeddings)], dim=2)
            )

            return x

    def predict_mask(self, x):
        with torch.no_grad():
            return (self.unet(x) > 0.5).type(torch.float32)

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

    def aug_features(self, embeddings):
        aug_f = torch.stack(
            [
                torch.cat(
                    [
                        self.regressor(b.to(self.device)).detach().cpu()
                        for b in DataLoader(v, batch_size=32)
                    ]
                )
                for v in embeddings
            ],
            dim=0,
        )

        return aug_f.to(self.device)


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


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=64):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         x = x + self.pe[: x.size(0), :]

#         return self.dropout(x)


# class TransformerDecoder(nn.Module):
#     def __init__(self, d_model=256, num_outputs=2, nlayers=3, dropout=0.3, nhead=8):
#         super(TransformerDecoder, self).__init__()
#         self.nhead = nhead
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
#         decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=nhead)
#         self.transformer_decoder = nn.TransformerDecoder(
#             decoder_layer, num_layers=nlayers
#         )
#         self.linear = nn.Linear(d_model, num_outputs)

#     def forward(self, x):
#         x_ = self.pos_encoder(x)
#         output = self.transformer_decoder(x_, x)
#         output = output.permute(1, 0, 2)[:, -1, :]
#         output = self.linear(output)

#         return output

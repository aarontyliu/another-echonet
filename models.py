import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from torchvision import models


class EchoNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.encoder = EfficientNetEncoder(name="efficientnet-b0")
        self.encoder = ResNetEncoder()
        self.decoder = LSTMDecoder(256, 256, 2, 1)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x

    def encode(self, x):
        embeddings = torch.stack(
            [
                torch.cat(
                    [
                        self.encoder(b.to(self.device)).detach().cpu()
                        for b in DataLoader(v, batch_size=32)
                    ]
                )
                for v in x
            ],
            dim=1,
        )

        return embeddings.to(self.device)

    def decode(self, x):

        return self.decoder(x)


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.encoder = models.resnet50(pretrained=False)
        self.fc1 = nn.Linear(self.encoder.fc.in_features, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.01)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.01)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class EfficientNetEncoder(nn.Module):
    def __init__(self, name="efficientnet-b0"):
        super(EfficientNetEncoder, self).__init__()
        self.name = name
        self.encoder = EfficientNet.from_name(self.name)
        self.encoder._avg_pooling = nn.Identity()
        self.encoder._swish = nn.Identity()
        self.fc1 = nn.Linear(self.encoder._fc.in_features * 3 * 3, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.01)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.encoder._fc = nn.Identity()


    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, num_outputs=2):
        super(LSTMDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, self.num_layers, bidirectional=True)
        self.fc1 = nn.Linear(2 * self.hidden_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(1, 0, 2)
        x = lstm_out[:, -1, :]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]

        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=256, num_outputs=2, nlayers=3, dropout=0.3, nhead=8):
        super(TransformerDecoder, self).__init__()
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=nlayers
        )
        self.linear = nn.Linear(d_model, num_outputs)

    def forward(self, x):
        x_ = self.pos_encoder(x)
        output = self.transformer_decoder(x_, x)
        output = output.permute(1, 0, 2)[:, -1, :]
        output = self.linear(output)

        return output

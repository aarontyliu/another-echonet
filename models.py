from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
from torch.utils.data import DataLoader


class EchoNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: EfficientNet family
        self.encoder = EfficientNet.from_name("efficientnet-b0")
        self.encoder._avg_pooling = nn.Identity()
        self.encoder._fc = nn.Identity()
        self.encoder._swish = nn.Identity()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # LSTM Decoder
        self.decoder = LSTMDecoder(11520, 700, 2, 2)

    def forward(self, x):
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

        x = self.decoder(embeddings.to(self.device))

        return x


class LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2, num_outputs=2):
        super(LSTMDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            embedding_dim, self.hidden_dim, num_layers, bidirectional=True
        )
        self.linear = nn.Linear(2 * self.hidden_dim, num_outputs)

    def forward(self, x):
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        x = self.linear(lstm_out)
        return x

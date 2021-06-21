import torch

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, y_hat, y):
        x = y_hat - y
        return torch.mean(torch.log(torch.cosh(x + self.eps)))
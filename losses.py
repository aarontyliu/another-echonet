#!/usr/bin/env python
"""Loss module (import to models module)
   Author: Aaron Liu
   Email: tl254@duke.edu
   Created on: June 16 2021
"""

import torch

class LogCoshLoss(torch.nn.Module):
    '''Log Cosh Loss (regression loss)'''
    def __init__(self):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, y_hat, y):
        x = y_hat - y
        return torch.mean(torch.log(torch.cosh(x + self.eps)))
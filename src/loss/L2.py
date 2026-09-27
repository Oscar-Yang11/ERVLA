import torch.nn as nn


class L2Loss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, target):
        return self.loss(x, target)

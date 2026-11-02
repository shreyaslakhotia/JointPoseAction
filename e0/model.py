"""A small residual dilated TCN for segment-level classification."""
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(1, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(1, width), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class TCN(nn.Module):
    def __init__(self, input_dim, classes, width=64, dilations=(1, 2, 4), dropout=0.2):
        super().__init__()
        if classes < 2 or input_dim < 1 or width < 1 or any(d < 1 for d in dilations):
            raise ValueError("Invalid TCN dimensions")
        self.input_dim = input_dim
        self.encoder = nn.Sequential(nn.Conv1d(input_dim, width, 1),
                                     *[Block(width, d, dropout) for d in dilations])
        self.classifier = nn.Linear(width, classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError("Expected batch x time x features")
        return self.classifier(self.encoder(x.transpose(1, 2)).mean(-1))

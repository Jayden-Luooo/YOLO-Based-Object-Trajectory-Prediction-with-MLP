import torch
from torch import nn


class TrajectoryMLP(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

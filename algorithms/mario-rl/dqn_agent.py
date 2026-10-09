
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_channels=4, num_actions=7):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Determine the flattened size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 84, 84)
            flattened_size = self.conv(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # Pixel values are converted to the range [0, 1]
        x = x.float() / 255.0
        x = self.conv(x)
        return self.head(x)


if __name__ == "__main__":
    model = DQN()
    sample = torch.zeros(1, 4, 84, 84)
    q_values = model(sample)

    print("DQN model created successfully!")
    print("Q-values shape:", q_values.shape)
    print("Q-values:", q_values.detach().numpy())

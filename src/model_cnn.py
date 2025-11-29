# src/model_cnn.py
"""
CNN model for fruit classification.
A small, robust conv-net suitable for small datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        # Conv blocks
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # After 3 pools with input 300 -> 300/8=37.5 -> use adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(0.5)

        # Compute final linear size
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)           # /2
        x = F.relu(self.conv2(x))
        x = self.pool(x)           # /4
        x = F.relu(self.conv3(x))
        x = self.pool(x)           # /8
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

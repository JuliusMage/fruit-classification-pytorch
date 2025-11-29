# src/model_mlp.py
"""
Simple MLP model for baseline fruit classification.
This model flattens the input image and uses fully connected layers.
"""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_channels=3, img_size=300, hidden=512, num_classes=3, dropout=0.5):
        super().__init__()
        input_features = input_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

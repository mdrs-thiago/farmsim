"""Matched lightweight CNN and CNN-LSTM models for line estimation."""

import torch
import torch.nn as nn


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class LineEncoder(nn.Module):
    def __init__(self, feature_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            block(3, 16), block(16, 32), block(32, 64), block(64, feature_dim),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images):
        return self.net(images).flatten(1)


class CNNLineModel(nn.Module):
    """Processes every frame independently; accepts B,T,C,H,W for fairness."""
    kind = 'cnn'

    def __init__(self, feature_dim=96, hidden_dim=128):
        super().__init__()
        self.encoder = LineEncoder(feature_dim)
        self.head = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, 4))

    def forward(self, images):
        batch, steps = images.shape[:2]
        features = self.encoder(images.flatten(0, 1))
        return self.head(features).view(batch, steps, 4)


class CNNLSTMLineModel(nn.Module):
    """Uses the same visual encoder, then integrates each contiguous clip."""
    kind = 'cnn_lstm'

    def __init__(self, feature_dim=96, hidden_dim=128):
        super().__init__()
        self.encoder = LineEncoder(feature_dim)
        self.temporal = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 4)

    def forward(self, images):
        batch, steps = images.shape[:2]
        features = self.encoder(images.flatten(0, 1)).view(batch, steps, -1)
        temporal, _ = self.temporal(features)
        return self.head(temporal)


def make_model(kind):
    if kind == 'cnn':
        return CNNLineModel()
    if kind == 'cnn_lstm':
        return CNNLSTMLineModel()
    raise ValueError(f'Unknown model kind: {kind}')

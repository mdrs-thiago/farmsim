"""Spatial heatmap CNN and causal CNN-LSTM line estimators."""

import torch
import torch.nn as nn


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True),
    )


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(block(5, 24), block(24, 48), block(48, 96))

    def forward(self, image):
        n, _, h, w = image.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image.device, dtype=image.dtype),
            torch.linspace(-1, 1, w, device=image.device, dtype=image.dtype), indexing='ij')
        coords = torch.stack((xx, yy)).expand(n, -1, -1, -1)
        return self.net(torch.cat((image, coords), dim=1))


class SpatialFrameFeatures(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = SpatialEncoder()
        self.heatmap = nn.Sequential(nn.Conv2d(96, 64, 3, padding=1), nn.SiLU(),
                                     nn.Conv2d(64, 2, 1))
        self.embedding = nn.Sequential(nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
                                       nn.Linear(96 * 3 * 4, embedding_dim), nn.SiLU())

    @staticmethod
    def soft_argmax(logits):
        n, endpoints, h, w = logits.shape
        probability = logits.flatten(2).softmax(-1).view(n, endpoints, h, w)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=logits.device, dtype=logits.dtype),
            torch.linspace(-1, 1, w, device=logits.device, dtype=logits.dtype), indexing='ij')
        x = (probability * xx).sum((-2, -1)); y = (probability * yy).sum((-2, -1))
        return torch.stack((x, y), dim=-1).flatten(1)

    def forward(self, image):
        features = self.encoder(image)
        logits = self.heatmap(features)
        return self.soft_argmax(logits), self.embedding(features), logits


class SpatialCNNLineModel(nn.Module):
    kind = 'spatial_cnn'

    def __init__(self):
        super().__init__(); self.frames = SpatialFrameFeatures()
        self.residual = nn.Sequential(nn.Linear(132, 128), nn.SiLU(), nn.Linear(128, 4))

    def forward(self, images):
        batch, steps = images.shape[:2]
        base, embedding, logits = self.frames(images.flatten(0, 1))
        line = base + 0.5 * torch.tanh(self.residual(torch.cat((embedding, base), dim=-1)))
        return {'line': line.view(batch, steps, 4),
                'heatmap_logits': logits.view(batch, steps, 2, *logits.shape[-2:])}


class SpatialCNNLSTMLineModel(nn.Module):
    kind = 'spatial_cnn_lstm'

    def __init__(self):
        super().__init__(); self.frames = SpatialFrameFeatures()
        self.temporal = nn.LSTM(132, 128, batch_first=True)
        self.residual = nn.Linear(128, 4)

    def forward(self, images):
        batch, steps = images.shape[:2]
        base, embedding, logits = self.frames(images.flatten(0, 1))
        base = base.view(batch, steps, 4); embedding = embedding.view(batch, steps, -1)
        temporal, _ = self.temporal(torch.cat((embedding, base), dim=-1))
        line = base + 0.5 * torch.tanh(self.residual(temporal))
        return {'line': line,
                'heatmap_logits': logits.view(batch, steps, 2, *logits.shape[-2:])}


def make_model(kind):
    if kind == 'spatial_cnn': return SpatialCNNLineModel()
    if kind == 'spatial_cnn_lstm': return SpatialCNNLSTMLineModel()
    raise ValueError(kind)

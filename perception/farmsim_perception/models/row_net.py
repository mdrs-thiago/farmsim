"""
models.py

Two multi-task row-perception models sharing the same encoder, decoder and heads:

  RowCNN      per-frame: ResNet-18 encoder -> U-Net style decoder (semantic
              segmentation) + heads on the bottleneck (row-bed bbox + presence,
              row line near/far points + visibility, lateral offset / heading).

  RowCNNLSTM  identical, plus causal ConvLSTM cells on the /16 and /32 encoder
              features. The recurrent state carries the scene layout across
              frames. It is added back to the current frame's features as a
              zero-initialised residual, so the untrained model starts out
              equal to RowCNN and can only learn to use memory where it helps.
              Fine detail still comes from the current frame's /4 and /8 skips.

All models take float images in [0,1] shaped (B,3,H,W) or (B,T,3,H,W) and return
a dict of outputs with the same leading dims. RowCNNLSTM.step() runs one frame
with an explicit state for online (deployment-style) inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from farmsim_perception.data import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.register_buffer('mean', IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer('std', IMAGENET_STD.clone(), persistent=False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        f4 = self.layer1(self.stem(x))   # /4,  64
        f8 = self.layer2(f4)             # /8,  128
        f16 = self.layer3(f8)            # /16, 256
        f32 = self.layer4(f16)           # /32, 512
        return f4, f8, f16, f32


def conv_bn(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Decoder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.up32 = conv_bn(512 + 256, 256)
        self.up16 = conv_bn(256 + 128, 128)
        self.up8 = conv_bn(128 + 64, 64)
        self.refine = conv_bn(64, 64)
        self.out = nn.Conv2d(64, num_classes, 1)

    @staticmethod
    def _up_cat(x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1)

    def forward(self, f4, f8, f16, f32, out_hw):
        x = self.up32(self._up_cat(f32, f16))
        x = self.up16(self._up_cat(x, f8))
        x = self.up8(self._up_cat(x, f4))
        x = self.out(self.refine(x))
        return F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False)


class Heads(nn.Module):
    """Global heads on the /32 features. A coarse spatial pool (3x4) keeps where
    things are, which pure global average pooling would throw away."""

    def __init__(self, in_ch=512):
        super().__init__()
        self.reduce = conv_bn(in_ch, 128, 1)
        self.pool = nn.AdaptiveAvgPool2d((3, 4))
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(128 * 12, 256), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.bbox = nn.Linear(256, 5)   # 4 box coords (sigmoid) + presence logit
        self.line = nn.Linear(256, 6)   # u_near, v_near, u_far, v_far + 2 visibility logits
        self.pose = nn.Linear(256, 2)   # lateral offset (m), heading error (rad)

    def forward(self, f32):
        h = self.trunk(self.pool(self.reduce(f32)))
        b, l = self.bbox(h), self.line(h)
        # centre/size parametrisation guarantees x0 <= x1 and y0 <= y1 (GIoU is meaningless otherwise)
        cxcy, wh = torch.sigmoid(b[:, :2]), torch.sigmoid(b[:, 2:4])
        bbox = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)
        return {'bbox': bbox, 'bbox_logit': b[:, 4],
                'line': l[:, :4], 'line_vis_logit': l[:, 4:], 'pose': self.pose(h)}


class RowCNN(nn.Module):
    kind = 'cnn'

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder, self.decoder, self.heads = Encoder(pretrained), Decoder(), Heads()

    def frame_forward(self, x):
        f4, f8, f16, f32 = self.encoder(x)
        out = self.heads(f32)
        out['seg'] = self.decoder(f4, f8, f16, f32, x.shape[-2:])
        return out

    def forward(self, x):
        if x.dim() == 5:
            b, t = x.shape[:2]
            out = self.frame_forward(x.flatten(0, 1))
            return {k: v.view(b, t, *v.shape[1:]) for k, v in out.items()}
        return self.frame_forward(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, 3, padding=1)
        nn.init.zeros_(self.gates.bias)
        with torch.no_grad():
            self.gates.bias[hidden_ch:2 * hidden_ch].fill_(1.0)  # forget-gate bias: remember by default

    def forward(self, x, state):
        if state is None:
            z = x.new_zeros(x.shape[0], self.hidden_ch, *x.shape[-2:])
            state = (z, z)
        h, c = state
        i, f, o, g = self.gates(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, (h, c)


class TemporalFusion(nn.Module):
    """ConvLSTM over one feature level, fused back as a zero-init residual."""

    def __init__(self, ch, hidden):
        super().__init__()
        self.inp = conv_bn(ch, hidden, 1)
        self.cell = ConvLSTMCell(hidden, hidden)
        self.out = nn.Conv2d(hidden + ch, ch, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, f, state):
        h, state = self.cell(self.inp(f), state)
        return f + self.out(torch.cat([h, f], 1)), state


class RowCNNLSTM(RowCNN):
    kind = 'cnn_lstm'

    def __init__(self, pretrained=True):
        super().__init__(pretrained)
        self.t16 = TemporalFusion(256, 128)
        self.t32 = TemporalFusion(512, 256)

    def step(self, x, state=None):
        """One frame (B,3,H,W) with recurrent state -> (outputs, new state)."""
        s16, s32 = state if state is not None else (None, None)
        f4, f8, f16, f32 = self.encoder(x)
        f16, s16 = self.t16(f16, s16)
        f32, s32 = self.t32(f32, s32)
        out = self.heads(f32)
        out['seg'] = self.decoder(f4, f8, f16, f32, x.shape[-2:])
        return out, (s16, s32)

    def forward(self, x, state=None):
        if x.dim() == 4:
            return self.step(x, state)[0]
        b, t = x.shape[:2]
        # run the encoder once over all frames, then unroll the recurrence
        f4, f8, f16, f32 = self.encoder(x.flatten(0, 1))
        f16, f32 = f16.view(b, t, *f16.shape[1:]), f32.view(b, t, *f32.shape[1:])
        s16, s32 = state if state is not None else (None, None)
        o16, o32 = [], []
        for i in range(t):
            y16, s16 = self.t16(f16[:, i], s16)
            y32, s32 = self.t32(f32[:, i], s32)
            o16.append(y16); o32.append(y32)
        f16, f32 = torch.stack(o16, 1).flatten(0, 1), torch.stack(o32, 1).flatten(0, 1)
        out = self.heads(f32)
        out['seg'] = self.decoder(f4, f8, f16, f32, x.shape[-2:])
        return {k: v.view(b, t, *v.shape[1:]) for k, v in out.items()}


def make_model(kind, pretrained=True):
    return {'cnn': RowCNN, 'cnn_lstm': RowCNNLSTM}[kind](pretrained)

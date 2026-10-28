"""
engine.py

Loss, sequence-level inference and metrics shared by train.py and evaluate.py.

Evaluation always runs whole sequences in temporal order. The CNN sees each
frame independently. The CNN-LSTM carries its state from the first frame to the
last, as it would on the robot. That makes temporal metrics (flicker, jitter)
comparable between the two models.
"""

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou_loss

from farmsim_perception.data import CLASS_NAMES, NUM_CLASSES, augment_batch


def multitask_loss(out, batch, class_weights=None):
    """All tensors already flattened to (N, ...)."""
    seg = F.cross_entropy(out['seg'], batch['mask'], weight=class_weights)
    prob = out['seg'].float().softmax(1)
    onehot = F.one_hot(batch['mask'], NUM_CLASSES).permute(0, 3, 1, 2).float()
    inter = (prob * onehot).sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3)) + 1)).mean()

    present = batch['bbox_present']
    presence = F.binary_cross_entropy_with_logits(out['bbox_logit'].float(), present)
    if present.sum() > 0:
        m = present > 0
        pb, gb = out['bbox'][m].float(), batch['bbox'][m]
        box = F.l1_loss(pb, gb) + generalized_box_iou_loss(pb, gb, reduction='mean')
    else:
        box = out['bbox'].sum() * 0

    vis = batch['line_vis']
    point_mask = vis.repeat_interleave(2, dim=1)  # (N,4): near u,v then far u,v
    line_err = F.smooth_l1_loss(out['line'].float(), batch['line'], reduction='none', beta=0.1)
    line = (line_err * point_mask).sum() / point_mask.sum().clamp(min=1)
    line_vis = F.binary_cross_entropy_with_logits(out['line_vis_logit'].float(), vis)

    pose = F.smooth_l1_loss(out['pose'].float(), batch['pose'], beta=0.05)

    terms = {'seg': seg, 'dice': dice, 'presence': 0.2 * presence, 'box': box,
             'line': 2.0 * line, 'line_vis': 0.2 * line_vis, 'pose': 2.0 * pose}
    return sum(terms.values()), {k: v.item() for k, v in terms.items()}


def seq_generator(name, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    return g


@torch.no_grad()
def predict_sequence(model, seq, device, corrupt=False, chunk=24, window=None):
    """Run one sequence in order; returns numpy predictions for every frame.

    For the CNN-LSTM, `window=None` carries the state across the whole sequence
    (deployment mode); `window=k` predicts frame t from frames t-k+1..t only,
    starting from a zero state, matching the clip length it was trained on."""
    images = torch.from_numpy(seq['images']).to(device).permute(0, 3, 1, 2)
    if corrupt:
        x = augment_batch(images.unsqueeze(0), generator=seq_generator(seq['name'], device))[0]
    else:
        x = images.float() / 255.0
    preds = {k: [] for k in ('seg', 'bbox', 'bbox_prob', 'line', 'line_vis_prob', 'pose')}
    state = None
    for s in range(0, len(x), chunk):
        xc = x[s:s + chunk]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            if model.kind == 'cnn_lstm' and window:
                out = _windowed_chunk(model, x, s, len(xc), window)
            elif model.kind == 'cnn_lstm':
                outs = []
                for i in range(len(xc)):
                    o, state = model.step(xc[i:i + 1], state)
                    outs.append(o)
                out = {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
            else:
                out = model(xc)
        preds['seg'].append(out['seg'].argmax(1).byte().cpu())
        preds['bbox'].append(out['bbox'].float().cpu())
        preds['bbox_prob'].append(torch.sigmoid(out['bbox_logit'].float()).cpu())
        preds['line'].append(out['line'].float().cpu())
        preds['line_vis_prob'].append(torch.sigmoid(out['line_vis_logit'].float()).cpu())
        preds['pose'].append(out['pose'].float().cpu())
    return {k: torch.cat(v).numpy() for k, v in preds.items()}


def _windowed_chunk(model, x, start, n, window):
    outs = []
    for t in range(start, start + n):
        clip = x[max(0, t - window + 1):t + 1].unsqueeze(0)
        o = model(clip)
        outs.append({k: v[:, -1] for k, v in o.items()})
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


class Metrics:
    def __init__(self):
        self.conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        self.flicker_changed = 0
        self.flicker_stable = 0
        self.det = {'tp': 0, 'fp': 0, 'fn': 0, 'iou_sum': 0.0, 'iou_n': 0}
        self.line_err = {'near': [], 'far': []}
        self.pose_err = []
        self.pose_jitter = []
        self.line_jitter = []

    def add_sequence(self, seq, p):
        gt, pr = seq['masks'].astype(np.int64), p['seg'].astype(np.int64)
        self.conf += np.bincount(gt.ravel() * NUM_CLASSES + pr.ravel(),
                                 minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        if len(gt) > 1:
            stable = gt[1:] == gt[:-1]
            self.flicker_stable += int(stable.sum())
            self.flicker_changed += int((stable & (pr[1:] != pr[:-1])).sum())

        present = seq['bbox_present'] > 0
        pred_present = p['bbox_prob'] > 0.5
        iou = box_iou(torch.from_numpy(p['bbox']), torch.from_numpy(seq['bbox'])).diagonal().numpy()
        hit = present & pred_present & (iou >= 0.5)
        self.det['tp'] += int(hit.sum())
        self.det['fp'] += int((pred_present & ~hit).sum())
        self.det['fn'] += int((present & ~hit).sum())
        self.det['iou_sum'] += float(iou[present].sum())
        self.det['iou_n'] += int(present.sum())

        scale = np.array([320, 240, 320, 240], np.float32)  # normalised [-1,1] -> 640x480 pixels
        err = np.abs(p['line'] - seq['line']) * scale
        for j, name in enumerate(('near', 'far')):
            vis = seq['line_vis'][:, j] > 0
            self.line_err[name].extend(np.hypot(err[vis, 2 * j], err[vis, 2 * j + 1]).tolist())

        e = p['pose'] - seq['pose']
        self.pose_err.append(np.abs(e))
        if len(e) > 1:
            # frame-to-frame change of the error: 0 for a constant bias, large for flicker
            self.pose_jitter.append(np.abs(np.diff(e, axis=0)))
            both = (seq['line_vis'][1:, 1] > 0) & (seq['line_vis'][:-1, 1] > 0)
            d = np.diff((p['line'] - seq['line'])[:, 2:] * scale[2:], axis=0)
            self.line_jitter.extend(np.hypot(d[both, 0], d[both, 1]).tolist())

    def summary(self):
        c = self.conf
        iou = np.diag(c) / np.maximum(c.sum(0) + c.sum(1) - np.diag(c), 1)
        d = self.det
        prec = d['tp'] / max(d['tp'] + d['fp'], 1)
        rec = d['tp'] / max(d['tp'] + d['fn'], 1)
        pe = np.concatenate(self.pose_err) if self.pose_err else np.zeros((1, 2))
        pj = np.concatenate(self.pose_jitter) if self.pose_jitter else np.zeros((1, 2))
        return {
            'seg_mIoU': float(iou.mean()),
            **{f'seg_IoU_{n}': float(v) for n, v in zip(CLASS_NAMES, iou)},
            'seg_pixel_acc': float(np.diag(c).sum() / max(c.sum(), 1)),
            'seg_flicker_pct': 100 * self.flicker_changed / max(self.flicker_stable, 1),
            'det_precision@0.5': prec,
            'det_recall@0.5': rec,
            'det_F1@0.5': 2 * prec * rec / max(prec + rec, 1e-9),
            'det_mean_IoU': d['iou_sum'] / max(d['iou_n'], 1),
            'line_near_err_px': float(np.mean(self.line_err['near'])) if self.line_err['near'] else math.nan,
            'line_far_err_px': float(np.mean(self.line_err['far'])) if self.line_err['far'] else math.nan,
            'line_far_jitter_px': float(np.mean(self.line_jitter)) if self.line_jitter else math.nan,
            'lateral_MAE_cm': float(100 * pe[:, 0].mean()),
            'heading_MAE_deg': float(np.degrees(pe[:, 1].mean())),
            'lateral_jitter_cm': float(100 * pj[:, 0].mean()),
            'heading_jitter_deg': float(np.degrees(pj[:, 1].mean())),
        }


def evaluate(model, sequences, device, corrupt=False):
    model.eval()
    m = Metrics()
    for seq in sequences.seqs:
        m.add_sequence(seq, predict_sequence(model, seq, device, corrupt=corrupt))
    return m.summary()

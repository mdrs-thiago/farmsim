"""
inference.py

Runs a trained row-perception checkpoint on single camera frames, in order,
the way the robot sees them. The CNN-LSTM keeps its recurrent state between
calls; `reset()` clears it at the start of a run.

    perception = RowPerception('models/row_net/cnn/last.pt')
    est = perception(rgb_uint8)       # HxWx3 RGB, any size
    est['lateral_offset'], est['heading_error'], est['confidence']

`confidence` is the mean predicted visibility of the near and far row-line
points: low when the row is out of view or the image is corrupted. The
controller's speed policy slows down on it.
"""

import numpy as np
import torch
import torch.nn.functional as F

from farmsim_perception.models.row_net import make_model


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model = make_model(ckpt['model'], pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model.eval(), ckpt


def pick_device(device='auto'):
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


class RowPerception:

    def __init__(self, checkpoint, device='auto', input_size=(256, 192)):
        self.device = pick_device(device)
        self.model, self.ckpt = load_checkpoint(checkpoint, self.device)
        self.input_size = input_size
        self.state = None

    def reset(self):
        self.state = None

    def preprocess(self, rgb):
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).float() / 255.0
        w, h = self.input_size
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x.unsqueeze(0), size=(h, w), mode='area')[0]
        return x.unsqueeze(0)

    @torch.no_grad()
    def __call__(self, rgb):
        x = self.preprocess(rgb)
        if self.model.kind == 'cnn_lstm':
            out, self.state = self.model.step(x, self.state)
        else:
            out = self.model(x)
        pose = out['pose'][0].float().cpu().numpy()
        vis = torch.sigmoid(out['line_vis_logit'][0].float()).cpu().numpy()
        line = out['line'][0].float().cpu().numpy()
        return {'lateral_offset': float(pose[0]), 'heading_error': float(pose[1]),
                'confidence': float(vis.mean()), 'u_near': float(line[0]), 'v_near': float(line[1]),
                'u_far': float(line[2]), 'v_far': float(line[3]), 'seg': out['seg'][0].argmax(0).byte().cpu().numpy()}

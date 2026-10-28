#!/usr/bin/env python3
"""
optimize.py -- optimise CNN + control together.

method: control_aware (default)
  Fine-tunes the CNN and trains the controller (fuzzy by default) jointly on
  the recorded dataset. Gazebo is not differentiable, so instead of
  backpropagating through the simulator the loss looks at what the
  controller WOULD command:

    L = L_perception(CNN)                                    usual multitask loss
      + lambda_control   * | u(CNN estimate) - u(ground truth) |^2_W
      + lambda_reference * | u(ground truth) - u_ref(ground truth) |^2_W

  u = controller(state) = (v, omega), W weights v and omega. The second term
  makes perception errors cost what they cost the controller (an error that
  does not change the command is free; one that flips the steering is
  expensive). The third term anchors the trained controller to a reference
  law (PD by default) so it cannot make the second term vanish by ignoring
  its input. yaw rate is taken from the recorded command (cmd_vel), the
  closest signal the dataset has. With closed_loop_eval the result is then
  driven in the simulator with run_control.

method: rl (experimental)
  REINFORCE on the simulator (farmsim_control.rl_env): a Gaussian policy whose
  mean is the controller's (v, omega) on the CNN estimate; the controller's
  parameters and the log-std are learned. Runs, but untuned. Needs ROS and
  the workspace sourced in this shell.

  python optimize.py
  python optimize.py optimize.control_aware.epochs=10 control.strategy=pd
  python optimize.py optimize.method=rl

Output: models/optimize/<name>/ (perception.pt in the row_net checkpoint
format, controller.pt, control.yaml pointing at it, history.json).
"""

import json
import sys

import torch
import yaml

from farmsim_utils.config import parse
from farmsim_utils.paths import MODELS_DIR, RUNS_DIR
from farmsim_utils.pipeline import repo_path, timestamp


def command_loss(u_a, u_b, w):
    return w['v'] * ((u_a[0] - u_b[0]) ** 2).mean() + w['omega'] * ((u_a[1] - u_b[1]) ** 2).mean()


def batch_obs(pose, yaw_rate, confidence=None):
    obs = {'lateral_offset': pose[:, 0], 'heading_error': pose[:, 1], 'yaw_rate': yaw_rate,
           'speed': torch.zeros_like(yaw_rate)}
    if confidence is not None:
        obs['confidence'] = confidence
    return obs


def yaw_rates(root, sequences):
    """Recorded angular command per frame, aligned with the cache."""
    out = {}
    for s in sequences.seqs:
        rates = []
        for line in open(root / 'sequences' / s['name'] / 'annotations.jsonl'):
            cmd = json.loads(line).get('cmd_vel') or {}
            rates.append(float(cmd.get('angular_z', 0.0)))
        out[s['name']] = torch.tensor(rates)
    return out


def control_aware(cfg, out_dir):
    from torch.utils.data import DataLoader

    from farmsim_control.laws import make_controller
    from farmsim_perception.data import FrameDataset, Sequences, augment_batch
    from farmsim_perception.engine import multitask_loss
    from farmsim_perception.inference import load_checkpoint, pick_device
    from farmsim_perception.models.row_net import make_model

    oc = cfg['optimize']['control_aware']
    device = pick_device('auto')
    root = repo_path(oc['dataset'])
    if oc.get('checkpoint') and repo_path(oc['checkpoint']).exists():
        model, ckpt = load_checkpoint(repo_path(oc['checkpoint']), device)
        kind = ckpt['model']
    else:
        print('no perception checkpoint: starting from ImageNet weights')
        kind = oc.get('model', 'cnn')
        model = make_model(kind, pretrained=True).to(device)
    if kind != 'cnn':
        print('note: control_aware trains on single frames; the CNN-LSTM is run frame by frame here')
    controller = make_controller(cfg['control']).to(device)
    reference = make_controller(dict(cfg['control'], **oc['reference'], checkpoint=None)).to(device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    w = oc['command_weights']

    train = Sequences(str(root), 'train', limit=oc.get('limit_sequences'))
    val = Sequences(str(root), 'val', limit=oc.get('limit_sequences'))
    rates = {**yaw_rates(root, train), **yaw_rates(root, val)}
    for seqs in (train, val):
        for s in seqs.seqs:
            s['yaw_rate'] = rates[s['name']].numpy()

    class WithRate(FrameDataset):
        def __getitem__(self, i):
            item = super().__getitem__(i)
            si, t = self.index[i]
            item['yaw_rate'] = torch.tensor(float(self.seqs[si]['yaw_rate'][t]))
            return item

    loader = DataLoader(WithRate(train), batch_size=oc['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(WithRate(val), batch_size=oc['batch_size'])
    opt = torch.optim.AdamW([{'params': model.parameters(), 'lr': oc['lr_perception']},
                             {'params': controller.parameters(), 'lr': oc['lr_controller']}])

    def step(batch, train_mode):
        batch = {k: v.to(device) for k, v in batch.items()}
        x = augment_batch(batch['image']) if train_mode else batch['image'].float() / 255.0
        out = model(x)
        l_perc, _ = multitask_loss(out, batch)
        conf = torch.sigmoid(out['line_vis_logit'].float()).mean(1)
        u_pred = controller(batch_obs(out['pose'].float(), batch['yaw_rate'], conf))
        u_gt = controller(batch_obs(batch['pose'], batch['yaw_rate'], torch.ones_like(conf)))
        u_ref = reference(batch_obs(batch['pose'], batch['yaw_rate'], torch.ones_like(conf)))
        l_ctrl = command_loss(u_pred, u_gt, w)
        l_ref = command_loss(u_gt, u_ref, w)
        loss = l_perc + oc['lambda_control'] * l_ctrl + oc['lambda_reference'] * l_ref
        return loss, {'perception': l_perc.item(), 'control': l_ctrl.item(), 'reference': l_ref.item()}

    history, best = [], float('inf')
    for epoch in range(oc['epochs']):
        model.train(); controller.train()
        agg = {}
        for batch in loader:
            loss, parts = step(batch, True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v / len(loader)
        model.eval(); controller.eval()
        vagg = {}
        with torch.no_grad():
            for batch in val_loader:
                _, parts = step(batch, False)
                for k, v in parts.items():
                    vagg[k] = vagg.get(k, 0.0) + v / len(val_loader)
        score = vagg['perception'] + oc['lambda_control'] * vagg['control']
        history.append({'epoch': epoch, 'train': agg, 'val': vagg, 'score': score})
        print(f"epoch {epoch}: train {agg} | val {vagg}", flush=True)
        if score < best:
            best = score
            torch.save({'model': kind, 'state_dict': model.state_dict(), 'epoch': epoch,
                        'args': {'source': 'optimize.py control_aware'}}, out_dir / 'perception.pt')
            torch.save(controller.state_dict(), out_dir / 'controller.pt')
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    return {'best_val_score': best}


def rl(cfg, out_dir):
    from farmsim_control.laws import make_controller
    from farmsim_control.rl_env import FarmRowEnv

    rc = cfg['optimize']['rl']
    controller = make_controller(cfg['control'])
    log_std = torch.nn.Parameter(torch.full((2,), float(rc['init_log_std'])))
    opt = torch.optim.Adam(list(controller.parameters()) + [log_std], lr=rc['lr'])
    env = FarmRowEnv(cfg, rc['path'], rc['checkpoint'], out_dir, dt=rc['dt'], max_steps=rc['max_steps'],
                     max_offset_m=rc['max_offset_m'])
    history, baseline = [], None
    try:
        for ep in range(rc['episodes']):
            obs = env.reset()
            logps, rewards = [], []
            done = False
            while not done:
                o = torch.tensor(obs)
                v, w = controller({'lateral_offset': o[0:1], 'heading_error': o[1:2], 'confidence': o[2:3],
                                   'yaw_rate': o[3:4], 'speed': o[4:5], 'dist_to_end': o[5:6]})
                dist = torch.distributions.Normal(torch.cat([v, w]), log_std.exp())
                a = dist.sample()
                logps.append(dist.log_prob(a).sum())
                obs, r, done, info = env.step(a.numpy())
                rewards.append(r)
            ret = torch.tensor(rewards).flip(0).cumsum(0).flip(0)
            baseline = ret.mean() if baseline is None else 0.9 * baseline + 0.1 * ret.mean()
            loss = -(torch.stack(logps) * (ret - baseline)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append({'episode': ep, 'return': float(sum(rewards)), 'steps': len(rewards),
                            'progress_m': info['ref']['s'], 'off_track': info['off_track']})
            print(history[-1], flush=True)
            torch.save(controller.state_dict(), out_dir / 'controller.pt')
            with open(out_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)
    finally:
        env.close()
    return {'episodes': len(history)}


def main():
    cfg, _ = parse(__doc__, 'optimize.yaml')
    method = cfg['optimize']['method']
    name = cfg['optimize'].get('name') or f"{timestamp()}_{method}_{cfg['control']['strategy']}"
    out_dir = MODELS_DIR / 'optimize' / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if method == 'control_aware':
        result = control_aware(cfg, out_dir)
    elif method == 'rl':
        result = rl(cfg, out_dir)
    else:
        raise SystemExit(f'unknown optimize.method {method!r}')
    trained = dict(cfg['control'], checkpoint=str(out_dir / 'controller.pt'))
    with open(out_dir / 'control.yaml', 'w') as f:
        yaml.safe_dump({'control': trained}, f, sort_keys=False)
    print(f'saved to {out_dir}: {result}')

    if method == 'control_aware' and cfg['optimize']['control_aware'].get('closed_loop_eval'):
        from farmsim_utils.config import load_yaml
        from run_control import run
        run_cfg = load_yaml('run_control.yaml')
        run_cfg['control'] = trained
        run_cfg['run']['checkpoint'] = str(out_dir / 'perception.pt')
        metrics = run(run_cfg, RUNS_DIR / f'{name}_closed_loop')
        print('closed loop:', metrics)
        with open(out_dir / 'closed_loop_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    sys.exit(main())

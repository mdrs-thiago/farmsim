"""
config.py

YAML configuration with command-line overrides. Every root script loads one
file from cfg/ and accepts `key.sub=value` pairs on top of it:

    python run_control.py --config cfg/run_control.yaml control.strategy=fuzzy sim.headless=true

Values are parsed as YAML scalars, so `true`, `0.3` and `[1, 2]` get their
natural types. A config may name other files under `include:` (relative to
cfg/); they are loaded first and the including file wins on conflicts.
"""

import argparse
import copy
from pathlib import Path

import yaml

from farmsim_utils.paths import CFG_DIR


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CFG_DIR / path
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    merged = {}
    for inc in data.pop('include', []) or []:
        merged = deep_merge(merged, load_yaml(inc))
    return deep_merge(merged, data)


def set_dotted(cfg, dotted, value):
    node = cfg
    keys = dotted.split('.')
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_overrides(cfg, overrides):
    for item in overrides:
        if '=' not in item:
            raise ValueError(f'override {item!r} is not key=value')
        key, raw = item.split('=', 1)
        set_dotted(cfg, key, yaml.safe_load(raw))
    return cfg


def parse(description, default_config, add_args=None, argv=None):
    """Standard CLI for root scripts: --config FILE plus key=value overrides.
    Returns (cfg, args)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--config', default=default_config, help='YAML file (absolute or relative to cfg/)')
    if add_args:
        add_args(ap)
    ap.add_argument('overrides', nargs='*', help='key.sub=value overrides applied after the file')
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_yaml(args.config), args.overrides)
    return cfg, args

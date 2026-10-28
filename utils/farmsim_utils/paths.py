"""
paths.py

Fixed locations inside the farmsim repository. Everything resolves from the
repo root, so scripts behave the same whatever directory they are started
from. Set FARMSIM_ROOT to point a colcon-installed node at a checkout.
"""

import os
from pathlib import Path

REPO = Path(os.environ.get('FARMSIM_ROOT', Path(__file__).resolve().parents[2]))

CFG_DIR = REPO / 'cfg'
FARM_DIR = REPO / 'farm'
FIELDS_DIR = FARM_DIR / 'fields'
DATASET_DIR = REPO / 'dataset'
MODELS_DIR = REPO / 'models'
OUTPUTS_DIR = REPO / 'outputs'

RECORDINGS_DIR = OUTPUTS_DIR / 'recordings'
GENERATED_WORLDS_DIR = OUTPUTS_DIR / 'generated_worlds'
RUNS_DIR = OUTPUTS_DIR / 'runs'

DEFAULT_DATASET = DATASET_DIR / 'row_perception_v1'


def field_dir(date):
    return FIELDS_DIR / str(date)

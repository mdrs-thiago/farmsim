"""
fields.py

One surveyed field = one folder in farm/fields/<date>/. This module resolves
a field's files by role, so code never hard-codes a survey date:

    f = Field('20250425')
    f.rows_file, f.world, f.label_mesh, f.has_ground_truth

Ground-truth assets (label mesh, vegetation/structure rasters, georef,
heightmap) exist only for surveyed fields that went through the
farm/tools/ pipeline; such a field can be driven but not used to record a
labelled dataset.
"""

import os
from pathlib import Path

import yaml

# FARMSIM_ROOT wins so a colcon-installed copy of this module still finds the
# checkout's farm/ folder; a --symlink-install build resolves it either way.
FARM_DIR = (Path(os.environ['FARMSIM_ROOT']) / 'farm' if 'FARMSIM_ROOT' in os.environ
            else Path(__file__).resolve().parents[1])
FIELDS_DIR = FARM_DIR / 'fields'
WORLDS_DIR = FARM_DIR / 'worlds'

GROUND_TRUTH_FILES = ('label_mesh.glb', 'vegetation_raster.png', 'structure_raster.png',
                      'georef.json', 'terrain_heightmap.npz', 'topdown.yaml')


SHARED_MODELS_DIR = FARM_DIR / 'models'   # props used by every field (weed meshes)


def model_resource_path():
    """GZ_SIM_RESOURCE_PATH entries: every field's photogrammetry model plus
    the shared farm/models (weeds that world_gen scatters into corridors)."""
    dirs = [p for p in sorted(FIELDS_DIR.glob('*/model')) if p.is_dir()] + [SHARED_MODELS_DIR]
    return os.pathsep.join(str(p) for p in dirs)


def available_fields():
    return sorted(p.name for p in FIELDS_DIR.iterdir() if p.is_dir())


class Field:

    def __init__(self, date, fields_dir=None):
        self.date = str(date)
        self.dir = Path(fields_dir or FIELDS_DIR) / self.date
        if not self.dir.is_dir():
            raise FileNotFoundError(f'no field {self.date} in {self.dir.parent} '
                                    f'(available: {available_fields()})')

    def path(self, name):
        return self.dir / name

    @property
    def rows_file(self):
        """Canonical row centre lines; falls back to the raw marked rows."""
        for name in ('rows_centerline.yaml', 'rows.yaml'):
            if self.path(name).is_file():
                return self.path(name)
        raise FileNotFoundError(f'field {self.date} has no rows file')

    @property
    def world(self):
        return WORLDS_DIR / f'vollebekk_{self.date}.sdf'

    @property
    def world_name(self):
        return f'vollebekk_farm_{self.date}'

    @property
    def model_dir(self):
        return self.path('model')

    @property
    def has_model(self):
        return self.model_dir.is_dir() and any(self.model_dir.iterdir())

    label_mesh = property(lambda self: self.path('label_mesh.glb'))
    vegetation_raster = property(lambda self: self.path('vegetation_raster.png'))
    structure_raster = property(lambda self: self.path('structure_raster.png'))
    georef = property(lambda self: self.path('georef.json'))
    heightmap = property(lambda self: self.path('terrain_heightmap.npz'))
    topdown_image = property(lambda self: self.path('topdown.png'))
    topdown_meta_file = property(lambda self: self.path('topdown.yaml'))

    @property
    def topdown_meta(self):
        with open(self.topdown_meta_file) as f:
            return yaml.safe_load(f)

    @property
    def missing_ground_truth(self):
        return [n for n in GROUND_TRUTH_FILES if not self.path(n).is_file()]

    @property
    def has_ground_truth(self):
        return not self.missing_ground_truth

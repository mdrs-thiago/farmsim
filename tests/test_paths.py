import math

import numpy as np
import pytest
import yaml

from farmsim_farm.fields import Field, available_fields
from farmsim_farm.paths import FieldPath, PathTracker, RowPass

ROWS = {'rows': [{'id': i, 'points': [[0.0, -1.5 * i], [60.0, -1.5 * i]]} for i in range(6)]}


@pytest.fixture
def rows_file(tmp_path):
    p = tmp_path / 'rows.yaml'
    p.write_text(yaml.safe_dump(ROWS))
    return p


def test_reference_orders_rows_and_headland(rows_file):
    path = FieldPath('x', [RowPass(1, 'west', 'east'), RowPass(3, 'east', 'west')], rows_file)
    kinds = [s.kind for s in path.reference()]
    assert kinds == ['lead', 'row', 'lead', 'headland', 'lead', 'row', 'lead']
    segs = path.reference()
    assert segs[1].a[0] < segs[1].b[0]      # row 1 driven west -> east
    assert segs[5].a[0] > segs[5].b[0]      # row 3 driven east -> west
    x, y, yaw = path.spawn_pose()
    assert (x, y) == pytest.approx((-2.0, -1.5)) and yaw == pytest.approx(0.0)


def test_same_end_and_unknown_row_rejected(rows_file):
    with pytest.raises(ValueError, match='same end'):
        FieldPath('x', [RowPass(1, 'west', 'west')], rows_file)
    with pytest.raises(ValueError, match='not in rows file'):
        FieldPath('x', [RowPass(42, 'west', 'east')], rows_file)
    with pytest.raises(ValueError):
        FieldPath('x', [RowPass(1, 'up', 'east')], rows_file)


def test_start_end_names(rows_file):
    path = FieldPath('x', [{'row': 2, 'enter': 'end', 'exit': 'start'}], rows_file)
    row = [s for s in path.reference() if s.kind == 'row'][0]
    assert row.a[0] == 60.0


def test_tracker_does_not_jump_to_neighbour_row(rows_file):
    path = FieldPath('x', [RowPass(1, 'west', 'east'), RowPass(2, 'east', 'west')], rows_file)
    t = PathTracker(path)
    # drive along row 1, 0.8 m off toward row 2: past half the 1.5 m spacing, so
    # plain nearest-row lookup would already pick row 2
    for x in np.linspace(-2, 30, 40):
        ref = t.update(x, -1.5 - 0.8, 0.0)
    assert ref['kind'] == 'row' and ref['row'] == 1
    assert ref['lateral_offset'] == pytest.approx(-0.8)
    assert ref['dist_to_end'] == pytest.approx(30.0, abs=1.0)


def test_tracker_completes_and_signs(rows_file):
    path = FieldPath('x', [RowPass(0, 'west', 'east')], rows_file)
    t = PathTracker(path)
    ref = t.update(10.0, 0.3, math.radians(10))
    assert ref['lateral_offset'] > 0          # left of the row
    assert ref['heading_error'] < 0           # must turn clockwise
    for x in np.arange(10.0, 62.5, 0.5):
        ref = t.update(x, 0.0, 0.0)
    assert ref['done']


def test_example_paths_are_valid():
    from farmsim_utils.paths import CFG_DIR
    for f in (CFG_DIR / 'paths').glob('*.yaml'):
        FieldPath.load(f)


def test_fields_resolve():
    assert '20250425' in available_fields()
    f = Field('20250425')
    assert f.rows_file.exists() and f.world.exists()
    assert f.has_ground_truth, f.missing_ground_truth
    assert not Field('20250624').has_ground_truth

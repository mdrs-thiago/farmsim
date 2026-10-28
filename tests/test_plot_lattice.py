import numpy as np
import pytest

from farmsim_farm.plot_lattice import (
    apply_global_offset_scale,
    apply_homography,
    apply_nudges,
    fit_block_homography,
    generate_plot_grid,
    plots_to_geojson,
    plots_to_pixel,
)

# Axis-aligned block: top-left (100, 50), 400px wide, 200px tall, in image
# pixel order [TL, TR, BR, BL].
_CORNERS = [(100.0, 50.0), (500.0, 50.0), (500.0, 250.0), (100.0, 250.0)]


def test_homography_round_trip_is_exact():
    H, H_inv = fit_block_homography(_CORNERS)
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.3, 0.7]])
    px = apply_homography(H, unit_square)
    back = apply_homography(H_inv, px)
    assert np.allclose(back, unit_square, atol=1e-9)


def test_homography_maps_corners_correctly():
    H, _ = fit_block_homography(_CORNERS)
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    px = apply_homography(H, unit_square)
    assert np.allclose(px, np.array(_CORNERS), atol=1e-6)


def test_homography_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        fit_block_homography(_CORNERS[:3])


def test_plot_grid_cell_count_and_alley_inset():
    plots = generate_plot_grid(n_rows=4, n_cols=8, alley_width_fraction=0.2)
    assert len(plots) == 32

    p0 = next(p for p in plots if p['row'] == 0 and p['col'] == 0)
    pitch_x, pitch_y = 1.0 / 8, 1.0 / 4
    inset_x, inset_y = 0.2 * pitch_x / 2, 0.2 * pitch_y / 2
    expected = np.array([[inset_x, inset_y], [pitch_x - inset_x, inset_y],
                          [pitch_x - inset_x, pitch_y - inset_y], [inset_x, pitch_y - inset_y]])
    assert np.allclose(p0['corners'], expected)


def test_plot_grid_zero_alley_touches_edge_to_edge():
    plots = generate_plot_grid(n_rows=2, n_cols=2, alley_width_fraction=0.0)
    p00 = next(p for p in plots if p['row'] == 0 and p['col'] == 0)
    p01 = next(p for p in plots if p['row'] == 0 and p['col'] == 1)
    # Right edge of col 0 must exactly meet the left edge of col 1.
    assert p00['corners'][1, 0] == pytest.approx(p01['corners'][0, 0])


@pytest.mark.parametrize('alley', [-0.1, 1.0, 1.5])
def test_plot_grid_rejects_invalid_alley_fraction(alley):
    with pytest.raises(ValueError):
        generate_plot_grid(n_rows=2, n_cols=2, alley_width_fraction=alley)


def test_row_nudge_only_perturbs_its_own_row():
    plots = generate_plot_grid(n_rows=3, n_cols=3, alley_width_fraction=0.1)
    nudged = apply_nudges(plots, row_nudges={1: 0.05}, col_nudges=None)

    for before, after in zip(plots, nudged):
        if before['row'] == 1:
            assert np.allclose(after['corners'], before['corners'] + np.array([0.0, 0.05]))
        else:
            assert np.allclose(after['corners'], before['corners'])


def test_col_nudge_only_perturbs_its_own_col():
    plots = generate_plot_grid(n_rows=3, n_cols=3, alley_width_fraction=0.1)
    nudged = apply_nudges(plots, row_nudges=None, col_nudges={2: -0.03})

    for before, after in zip(plots, nudged):
        if before['col'] == 2:
            assert np.allclose(after['corners'], before['corners'] + np.array([-0.03, 0.0]))
        else:
            assert np.allclose(after['corners'], before['corners'])


def test_nudges_do_not_mutate_input():
    plots = generate_plot_grid(n_rows=2, n_cols=2, alley_width_fraction=0.1)
    original = [p['corners'].copy() for p in plots]
    apply_nudges(plots, row_nudges={0: 0.5}, col_nudges={0: 0.5})
    for p, orig in zip(plots, original):
        assert np.array_equal(p['corners'], orig)


def test_global_offset_scale_centered_at_block_center():
    plots = generate_plot_grid(n_rows=1, n_cols=1, alley_width_fraction=0.0)
    # The single plot spans the whole unit square; scaling by 0.5 about the
    # center should shrink it symmetrically to [0.25, 0.75].
    scaled = apply_global_offset_scale(plots, offset=(0.0, 0.0), scale=0.5)
    assert np.allclose(scaled[0]['corners'],
                        [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])


def test_plots_to_pixel_and_geojson_round_trip():
    H, _ = fit_block_homography(_CORNERS)
    plots = generate_plot_grid(n_rows=2, n_cols=3, alley_width_fraction=0.1)
    plots_px = plots_to_pixel(plots, H)
    assert all('corners_px' in p for p in plots_px)

    geojson = plots_to_geojson(plots_px)
    assert geojson['type'] == 'FeatureCollection'
    assert len(geojson['features']) == 6
    feature = geojson['features'][0]
    assert feature['geometry']['type'] == 'Polygon'
    ring = feature['geometry']['coordinates'][0]
    assert ring[0] == ring[-1]  # closed ring
    assert feature['properties']['excluded'] is False


def test_excluded_flag_propagates_to_geojson():
    H, _ = fit_block_homography(_CORNERS)
    plots = generate_plot_grid(n_rows=1, n_cols=2, alley_width_fraction=0.1)
    plots[0]['excluded'] = True
    geojson = plots_to_geojson(plots_to_pixel(plots, H))
    assert geojson['features'][0]['properties']['excluded'] is True
    assert geojson['features'][1]['properties']['excluded'] is False

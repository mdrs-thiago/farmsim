"""
palettes.py

Single owner of every declared class-color palette used across the
label-rendering pipeline. Deliberately pure Python + numpy, no cv2/
trimesh/scipy import: this needs to be importable from BOTH offline
venvs (label_render_venv has cv2, label_render_venv_gl does not -- moderngl
and cv2 were never installed together, see render_labels.py's docstring)
and from the ROS-package side (export_recording.py), without dragging
any of those heavier deps along just for a handful of color tuples.

Colors are OpenCV/PNG-convention BGR uint8 tuples, matching every raster
this pipeline already writes with cv2.imwrite/Image.save. Palette lists
are indexed by class id -- class_id == index into the palette list -- so
`classid_to_rgb`/`rgb_to_classid` are simple array-indexing/lookup, and
"unknown color in a mask" is a hard, detectable error rather than a
silent misclassification.

"""

import numpy as np

# ---------------------------------------------------------------------------
# Vegetation channel: is this pixel showing live crop canopy or not, within
# whatever the structure channel already says is "captured". VOID = outside
# the birdseye photo's coverage entirely (kept distinct from "bare/soil" so
# a downstream consumer can tell "we have no data here" from "we looked and
# there's no vegetation here").
VOID_COLOR = (0, 0, 0)        # BGR black
NOT_VEG_COLOR = (255, 0, 0)   # BGR blue
VEG_COLOR = (0, 255, 0)       # BGR green

VEG_PALETTE = [VOID_COLOR, NOT_VEG_COLOR, VEG_COLOR]
VEG_VOID, VEG_NOT_VEG, VEG_VEG = range(len(VEG_PALETTE))

# ---------------------------------------------------------------------------
# Structure channel: purely geometric -- is this world-XY cell within the
# surveyed footprint, and if so is it a corridor (valley, where the robot
# drives) or a bed (ridge, where crops are planted). Independent of hue/
# vegetation state by construction (see build_structure_raster.py).
STRUCT_VOID_COLOR = (0, 0, 0)        # BGR black -- outside the mesh footprint
STRUCT_BED_COLOR = (0, 255, 0)       # BGR green -- ridge / crop bed
STRUCT_CORRIDOR_COLOR = (255, 0, 0)  # BGR blue -- valley / drivable corridor ("row")

STRUCT_PALETTE = [STRUCT_VOID_COLOR, STRUCT_BED_COLOR, STRUCT_CORRIDOR_COLOR]
STRUCT_VOID, STRUCT_BED, STRUCT_CORRIDOR = range(len(STRUCT_PALETTE))

# ---------------------------------------------------------------------------
# Fused channel: NOT independently rendered -- computed downstream from the
# structure and vegetation class-id arrays (see fuse_structure_vegetation),
# wherever a single semantic map is actually wanted (dataset export, QC).
# Kept as a derived view rather than a third rendered channel so the two
# underlying signals (one stable/geometric, one per-survey-tuned/color)
# stay independently inspectable -- fusing only at the point of use means
# the fusion RULE can change later without re-baking or re-rendering
# anything.
FUSED_BACKGROUND_COLOR = (0, 0, 0)      # BGR black -- void in either channel (not actually observed)
FUSED_CROP_COLOR = (0, 255, 0)          # BGR green -- bed AND vegetated
FUSED_ROW_COLOR = (255, 0, 0)           # BGR blue -- corridor, regardless of vegetation (a weedy
                                         # corridor is still structurally a corridor)
FUSED_BARE_BED_COLOR = (0, 220, 255)    # BGR amber -- bed, not vegetated, not corridor

FUSED_PALETTE = [FUSED_BACKGROUND_COLOR, FUSED_CROP_COLOR, FUSED_ROW_COLOR, FUSED_BARE_BED_COLOR]
FUSED_BACKGROUND, FUSED_CROP, FUSED_ROW, FUSED_BARE_BED = range(len(FUSED_PALETTE))


def fuse_structure_vegetation(struct_ids, veg_ids):
    """Combine the independently-derived structure (bed/corridor/void) and
    vegetation (veg/bare/void) class-id arrays into one 4-class semantic
    map: background, crop (vegetated bed), row (corridor -- structure
    wins over vegetation state there, since the physical corridor doesn't
    stop being a corridor just because a weed is growing in it), bare-bed
    (bed, no vegetation detected). A cell only counts as "observed" (i.e.
    not background) when BOTH channels agree it was actually captured --
    trusting a location where either signal says void would let one
    channel's coverage gaps silently leak into the other's classification.
    `struct_ids`/`veg_ids` must be same-shape class-id arrays (e.g. from
    rgb_to_classid against STRUCT_PALETTE/VEG_PALETTE respectively)."""
    if struct_ids.shape != veg_ids.shape:
        raise ValueError(f'shape mismatch: structure {struct_ids.shape} vs vegetation {veg_ids.shape}')

    fused = np.full(struct_ids.shape, FUSED_BACKGROUND, dtype=np.uint8)
    observed = (struct_ids != STRUCT_VOID) & (veg_ids != VEG_VOID)
    is_corridor = observed & (struct_ids == STRUCT_CORRIDOR)
    is_bed = observed & (struct_ids == STRUCT_BED)

    fused[is_corridor] = FUSED_ROW
    fused[is_bed & (veg_ids == VEG_VEG)] = FUSED_CROP
    fused[is_bed & (veg_ids == VEG_NOT_VEG)] = FUSED_BARE_BED
    return fused


def classid_to_rgb(class_ids, palette):
    """(H,W) uint8 class-id array -> (H,W,3) uint8 BGR array via `palette`
    (a list of BGR tuples, index == class id)."""
    lut = np.array(palette, dtype=np.uint8)
    return lut[class_ids]


def rgb_to_classid(rgb, palette):
    """(H,W,3) uint8 BGR array -> (H,W) uint8 class-id array via exact
    color match against `palette`. Raises on any pixel whose color isn't
    one of the declared palette entries -- silently mapping an unknown
    color to some default class would hide a real bug (e.g. GL
    antialiasing bleed, a stale/wrong raster) rather than surface it,
    exactly the kind of "unexpected colors" check blend_check.py already
    treats as a hard failure."""
    h, w = rgb.shape[:2]
    class_ids = np.full((h, w), -1, dtype=np.int16)
    for idx, color in enumerate(palette):
        matches = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
        class_ids[matches] = idx
    unknown = int((class_ids < 0).sum())
    if unknown:
        bad_colors = set(map(tuple, np.unique(rgb[class_ids < 0].reshape(-1, 3), axis=0).tolist()))
        raise ValueError(
            f'{unknown} pixel(s) had a color not in the declared palette: {bad_colors}')
    return class_ids.astype(np.uint8)

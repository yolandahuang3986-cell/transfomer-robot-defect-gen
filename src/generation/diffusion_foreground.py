"""Foregrounds used by the diffusion-internal placement rules (placement_v1/v2).

Not shared QC. Shared QC keeps using src/generation/qc.py unchanged.
`core_thr3_fill_and_qc` = (QC background model, distance threshold x3, largest
component, holes filled) AND (QC foreground), estimated on the native-resolution
train/good image and nearest-resized to the generation grid with the QC helper.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes, label

from .qc import DEFAULT_QC_POLICY, _resize_binary_mask, estimate_object_foreground

VARIANTS = ('qc', 'core_thr3_fill_and_qc')


def _thresholded_foreground(image, multiplier, policy=DEFAULT_QC_POLICY):
    """Copy of qc.estimate_object_foreground with the final threshold scaled.

    multiplier=1 must reproduce the shared QC estimator exactly (checked below and in tests).
    """
    height, width = image.shape[1:]
    band = max(1, int(round(min(height, width) * policy.foreground_border_fraction)))
    hwc = np.moveaxis(image, 0, -1)
    border = np.zeros((height, width), dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    background_pixels = hwc[border]
    background = np.median(background_pixels, axis=0)
    distance = np.linalg.norm(background_pixels - background, axis=1)
    median = float(np.median(distance))
    mad = float(np.median(np.abs(distance - median)))
    threshold = max(policy.foreground_distance_floor,
                    median + policy.foreground_mad_multiplier * 1.4826 * mad) * multiplier
    components, count = label(np.linalg.norm(hwc - background, axis=2) > threshold,
                              structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        raise ValueError('Empty foreground at this threshold')
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def placement_foreground(image, variant, size=256, policy=DEFAULT_QC_POLICY):
    """Boolean size x size foreground for placement. `image`: float32 CHW in [0,1], native resolution."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown placement foreground variant {variant!r}')
    qc_fg = estimate_object_foreground(image, policy)
    if variant == 'core_thr3_fill_and_qc':
        if not np.array_equal(_thresholded_foreground(image, 1.0, policy), qc_fg):
            raise AssertionError('Copy of the shared QC foreground estimator drifted from qc.py')
        fg = binary_fill_holes(_thresholded_foreground(image, 3.0, policy)) & qc_fg
    else:
        fg = qc_fg
    fg = _resize_binary_mask(fg, (size, size))
    if not fg.any():
        raise ValueError('Placement foreground is empty on the generation grid')
    return fg

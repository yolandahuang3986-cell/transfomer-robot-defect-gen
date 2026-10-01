"""Shared in-memory interface: NCHW float32 RGB and N1HW binary masks."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class GeneratedBatch:
    images: np.ndarray
    masks: np.ndarray
    metadata: list[dict]


def validate_batch(batch, num_samples):
    if not isinstance(batch, GeneratedBatch):
        raise ValueError('Expected GeneratedBatch')
    x, y = batch.images, batch.masks
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError('Images/masks must be numpy arrays')
    if x.dtype != np.float32 or x.ndim != 4 or x.shape[:2] != (num_samples, 3):
        raise ValueError('Images must be float32 NCHW RGB')
    if min(x.shape[2:]) < 1 or y.shape != (num_samples, 1, *x.shape[2:]) or y.dtype != np.float32:
        raise ValueError('Masks must be float32 N1HW, spatially aligned')
    if not np.isfinite(x).all() or not ((x >= 0) & (x <= 1)).all():
        raise ValueError('Images must be finite and normalized to [0,1]')
    if not np.isin(y, [0., 1.]).all():
        raise ValueError('Masks must be binary')
    if len(batch.metadata) != num_samples:
        raise ValueError('One metadata record required per sample')
    for m in batch.metadata:
        for key in ('method', 'category', 'defect_type', 'source_image_id'):
            if not isinstance(m.get(key), str) or not m[key]:
                raise ValueError(f'Missing metadata string: {key}')
        if m['method'] not in {'dummy', 'procedural', 'gan', 'diffusion'}:
            raise ValueError('Unknown method')
        if type(m.get('seed')) is not int or not isinstance(m.get('generation_config'), dict):
            raise ValueError('Invalid seed/config')
        seconds = m.get('generation_seconds')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid generation runtime')
    return batch


class DefectGenerator(ABC):
    @abstractmethod
    def fit(self, dataset, config):
        ...

    @abstractmethod
    def generate(self, normal_images, num_samples, defect_type=None, seed=42, **kwargs):
        """Return GeneratedBatch. Metadata must identify each source image."""
        ...

"""Constant fixture patches for interface testing only; no synthesis baseline."""
import time
import numpy as np
from .base import DefectGenerator, GeneratedBatch, validate_batch


class DummyGenerator(DefectGenerator):
    def fit(self, dataset, config):
        return self

    def generate(self, normal_images, num_samples, defect_type=None, seed=42, **kwargs):
        start = time.perf_counter()
        normals = np.asarray(normal_images)
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError('num_samples must be positive')
        if normals.ndim != 4 or normals.shape[0] == 0 or normals.shape[1] != 3:
            raise ValueError('Expected nonempty NCHW normals')
        if normals.dtype != np.float32 or not np.isfinite(normals).all() or not ((normals >= 0) & (normals <= 1)).all():
            raise ValueError('Normals must be float32 in [0,1]')
        h, w = normals.shape[2:]
        if min(h, w) < 8:
            raise ValueError('Dummy fixture requires dimensions >= 8')
        ids = kwargs.get('source_image_ids', [f'fixture/{i}' for i in range(len(normals))])
        if len(ids) != len(normals):
            raise ValueError('Source IDs must align with normals')
        rng = np.random.default_rng(seed)
        x = normals[np.arange(num_samples) % len(normals)].copy()
        y = np.zeros((num_samples, 1, h, w), dtype=np.float32)
        for i in range(num_samples):
            top = int(rng.integers(0, h - h // 4 + 1))
            left = int(rng.integers(0, w - w // 4 + 1))
            y[i, :, top:top+h//4, left:left+w//4] = 1
            x[i, :, top:top+h//4, left:left+w//4] = 1
        seconds = (time.perf_counter() - start) / num_samples
        metadata = [dict(method='dummy', category=kwargs.get('category', 'fixture'),
                         defect_type=defect_type or 'fixture_patch', seed=seed,
                         source_image_id=ids[i % len(ids)],
                         generation_config={'fixture_only': True, 'patch_fraction': 0.25},
                         generation_seconds=seconds) for i in range(num_samples)]
        return validate_batch(GeneratedBatch(x, y, metadata), num_samples)

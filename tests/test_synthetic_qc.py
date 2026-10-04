import json

import numpy as np
import pytest

from src.generation.base import DefectGenerator, GeneratedBatch
from src.generation.qc import (
    QCUnderfilledError,
    estimate_object_foreground,
    filter_generated_batch,
)


def _metadata(count, method="procedural", source_id="metal_nut/train/good/001.png"):
    return [dict(method=method, category="metal_nut", defect_type="bent", seed=i,
                 source_image_id=source_id, generation_config={}, generation_seconds=0.01)
            for i in range(count)]


def _batch(masks, method="procedural"):
    masks = np.asarray(masks, dtype=np.float32)[:, None]
    images = np.full((len(masks), 3, *masks.shape[-2:]), 0.5, dtype=np.float32)
    return GeneratedBatch(images, masks, _metadata(len(masks), method=method))


def _object_mask(height=40, width=40):
    obj = np.zeros((height, width), dtype=bool)
    obj[4:36, 4:36] = True
    return obj


def test_foreground_estimator_finds_centered_object_from_normal_only():
    image = np.zeros((3, 40, 40), dtype=np.float32)
    image[:, 4:36, 4:36] = 0.6
    foreground = estimate_object_foreground(image)
    assert foreground[20, 20]
    assert not foreground[0, 0]
    assert foreground.sum() == 32 * 32


def test_filter_applies_shared_empty_area_and_object_overlap_rules():
    masks = np.zeros((5, 40, 40), dtype=np.float32)
    masks[0, 10:20, 10:20] = 1.0                # valid: 100 / 1024, fully on object
    # mask 1 remains empty
    masks[2, 10, 10] = 1.0                    # 1 / 1024 < 0.001
    masks[3, 10:25, 10:25] = 1.0             # 225 / 1024 > 0.20
    masks[4, 0:5, 10:20] = 1.0               # less than 90% lies on object
    batch = _batch(masks)

    result = filter_generated_batch(batch, {"metal_nut/train/good/001.png": _object_mask()})

    assert result.report.accepted_count == 1
    assert result.report.attempted_count == 5
    assert result.report.rejection_counts == {
        "empty_mask": 1,
        "mask_too_small": 1,
        "mask_too_large": 1,
        "low_foreground_overlap": 1,
    }
    assert result.batch.masks.shape == (1, 1, 40, 40)


def test_filter_rejects_unknown_source_foreground():
    with pytest.raises(ValueError, match="No training-normal foreground"):
        filter_generated_batch(_batch(np.ones((1, 40, 40))), {})


def test_source_foreground_is_nearest_resized_to_generated_output_grid():
    mask = np.zeros((1, 32, 32), dtype=np.float32)
    mask[0, 8:14, 8:14] = 1.0
    result = filter_generated_batch(
        _batch(mask), {"metal_nut/train/good/001.png": _object_mask(64, 64)}
    )
    assert result.report.accepted_count == 1


class FixedMaskGenerator(DefectGenerator):
    def __init__(self, valid):
        self.valid = valid

    def fit(self, dataset, config):
        return self

    def generate(self, normal_images, num_samples, defect_type=None, seed=42, **kwargs):
        h, w = normal_images.shape[-2:]
        images = np.full((num_samples, 3, h, w), 0.5, dtype=np.float32)
        masks = np.zeros((num_samples, 1, h, w), dtype=np.float32)
        if self.valid:
            masks[:, :, 10:20, 10:20] = 1.0
        ids = kwargs["source_image_ids"]
        metadata = [dict(method="gan", category="metal_nut", defect_type=defect_type or "bent",
                         seed=seed, source_image_id=ids[i % len(ids)],
                         generation_config={}, generation_seconds=0.01)
                    for i in range(num_samples)]
        return GeneratedBatch(images, masks, metadata)


def test_generator_base_entrypoint_reaches_target_via_shared_qc():
    normal = np.zeros((1, 3, 40, 40), dtype=np.float32)
    normal[:, :, 4:36, 4:36] = 0.6
    result = FixedMaskGenerator(valid=True).generate_with_qc(
        normal, ["metal_nut/train/good/001.png"], target_count=3, defect_type="bent", seed=10
    )
    assert result.report.target_met
    assert result.report.attempted_count == 3
    assert result.report.accepted_count == 3
    assert result.require_target().images.shape[0] == 3
    json.dumps(result.report.to_dict())


def test_generator_base_entrypoint_reports_underfill_at_ten_x_cap():
    normal = np.zeros((1, 3, 40, 40), dtype=np.float32)
    normal[:, :, 4:36, 4:36] = 0.6
    result = FixedMaskGenerator(valid=False).generate_with_qc(
        normal, ["metal_nut/train/good/001.png"], target_count=2, defect_type="bent", seed=10
    )
    assert not result.report.target_met
    assert result.report.attempted_count == 20
    assert result.report.rejection_counts["empty_mask"] == 20
    assert result.batch.images.shape[0] == 0
    assert result.report.rejected_samples[-1]["attempt_index"] == 19
    with pytest.raises(QCUnderfilledError):
        result.require_target()


def test_generator_base_entrypoint_rejects_non_training_normal_sources():
    normal = np.zeros((1, 3, 40, 40), dtype=np.float32)
    normal[:, :, 4:36, 4:36] = 0.6
    with pytest.raises(ValueError, match="train/good paths"):
        FixedMaskGenerator(valid=True).generate_with_qc(
            normal, ["metal_nut/test/good/001.png"], target_count=1
        )

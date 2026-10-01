import json
from pathlib import Path
import numpy as np
import pytest
from src.generation.base import GeneratedBatch, validate_batch
from src.generation.dummy import DummyGenerator
from src.inspection.results import validate_result
from src.metrics.segmentation import evaluate


def test_dummy_contract_is_reproducible_and_aligned():
    normals = np.zeros((2, 3, 32, 40), dtype=np.float32)
    g = DummyGenerator()
    a = g.generate(normals, 3, seed=42)
    b = g.generate(normals, 3, seed=42)
    np.testing.assert_array_equal(a.images, b.images)
    np.testing.assert_array_equal(a.masks, b.masks)
    assert len(a.metadata) == 3
    assert a.masks.sum() > 0


def test_contract_rejects_wrong_mask_shape():
    batch = DummyGenerator().generate(np.zeros((1, 3, 32, 32), np.float32), 1)
    with pytest.raises(ValueError, match='spatially aligned'):
        validate_batch(GeneratedBatch(batch.images, batch.masks[:, :, :-1], batch.metadata), 1)


def test_metrics_and_undefined_values():
    y = np.array([[[[0, 1], [0, 1]]]], dtype=np.float32)
    assert evaluate(y, y) == dict(pixel_auroc=1., iou=1., dice=1., defect_recall=1.)
    zeros = np.zeros_like(y)
    assert evaluate(zeros, zeros) == dict(pixel_auroc=None, iou=1., dice=1., defect_recall=None)


def test_result_schema_rejects_nonfinite_values():
    # A valid-shaped result with NaN metric cannot be serialized or accepted.
    result = dict(schema_version=1, experiment_id='x', category='fixture', method='dummy',
                  real_train_count=0, synthetic_count=1, seed=1,
                  generator={'generation_seconds_per_image': 0., 'config': {}},
                  inspection={'pixel_auroc': float('nan'), 'iou': 0., 'dice': 0., 'defect_recall': None},
                  training={'architecture':'unet','encoder':'resnet18','encoder_weights':None,
                            'image_size':64,'steps':1,'learning_rate':1e-4,'loss':1.},
                  evaluation={'source':'independent_fixture','count':1,'threshold':.5,
                              'smoke_only':True,'split_sha256':None},
                  provenance={'git_commit':'abc','git_dirty':False,'python':'3.10','torch':'2.8',
                              'cuda':None,'device':'cpu','runtime_seconds':0.,'packages':{}})
    with pytest.raises(ValueError):
        validate_result(result)

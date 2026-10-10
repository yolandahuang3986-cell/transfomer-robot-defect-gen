import copy
import json
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from src.data.integrity import sha256
from src.experiments.formal import fixed_step_loader, settings
from src.inspection.formal_bundle import load_formal_pool, require_acceptance
from src.metrics.segmentation import evaluate


@pytest.fixture
def pool(tmp_path, monkeypatch):
    # Small quota isolates admission rules without allocating a full production pool.
    import src.inspection.formal_bundle as module
    config = copy.deepcopy(settings())
    config['synthetic_counts']['metal_nut'] = 2
    config['split_sha256'] = 'frozen'
    monkeypatch.setattr(module, 'settings', lambda: config)
    source = 'metal_nut/train/good/000.png'
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[32:224, 32:224] = 200
    path = tmp_path / source
    path.parent.mkdir(parents=True)
    Image.fromarray(image).save(path)
    directory = tmp_path / 'bundle'; directory.mkdir()
    images = np.repeat((image.transpose(2, 0, 1).astype(np.float32) / 255)[None], 2, axis=0)
    masks = np.zeros((2, 1, 256, 256), dtype=np.float32)
    masks[:, :, 80:90, 80:90] = 1
    images[:, :, 80:90, 80:90] = .1
    records = [dict(method='procedural', category='metal_nut', defect_type='synthetic_patch',
                    seed=42, source_image_id=source, generation_seconds=.1,
                    generation_config={'donor_image_id': source}) for _ in range(2)]
    np.savez_compressed(directory / 'batch.npz', images=images, masks=masks)
    (directory / 'metadata.json').write_text(json.dumps(records))
    provenance = dict(category='metal_nut', split_sha256='frozen', seed=42, num_samples=2,
                      file_sha256={n: sha256(directory/n) for n in ('batch.npz', 'metadata.json')},
                      shared_qc=dict(policy_version='synthetic-qc-v1', target_met=True, target_count=2,
                                     accepted_count=2, attempted_count=3, generation_seconds=.3, qc_seconds=.1))
    (directory / 'provenance.json').write_text(json.dumps(provenance))
    manifest = dict(manifest_sha256='frozen', train_normal=[dict(category='metal_nut', image_path=source)],
                    support=[dict(category='metal_nut', defect_type='scratch')])
    def load():
        return load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path)
    return directory, provenance, load


def test_formal_pool_requires_qc_and_exact_quota(pool):
    directory, provenance, load = pool
    batch, identity, _ = load()
    assert len(batch.images) == 2 and len(identity['pool_sha256']) == 64
    provenance['shared_qc'] = None
    (directory/'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='QC report'):
        load()


@pytest.mark.parametrize('field,value', [('attempted_count', 21), ('target_met', False), ('policy_version', 'custom-qc')])
def test_formal_pool_rejects_invalid_qc_budget(pool, field, value):
    directory, provenance, load = pool
    provenance['shared_qc'][field] = value
    (directory/'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        load()


def test_claimed_qc_pass_cannot_admit_empty_mask(pool):
    directory, provenance, load = pool
    with np.load(directory/'batch.npz') as z:
        images, masks = z['images'], z['masks']
    masks[:] = 0
    np.savez_compressed(directory/'batch.npz', images=images, masks=masks)
    provenance['file_sha256']['batch.npz'] = sha256(directory/'batch.npz')
    (directory/'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='recheck failed'):
        load()


def test_formal_pool_rejects_test_source(pool):
    directory, provenance, load = pool
    rows = json.loads((directory/'metadata.json').read_text())
    rows[0]['source_image_id'] = 'metal_nut/test/scratch/000.png'
    (directory/'metadata.json').write_text(json.dumps(rows))
    provenance['file_sha256']['metadata.json'] = sha256(directory/'metadata.json')
    (directory/'provenance.json').write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match='train/good'):
        load()


def test_acceptance_binds_exact_pool_and_explicit_human_review(pool):
    directory, _, load = pool
    _, identity, _ = load()
    path = directory/'acceptance.json'
    record = dict(identity, status='pending_review', reviewer='A', notes='Checked', reviewed_at='2026-10-10',
                  training_provenance_verified=True)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Explicit'):
        require_acceptance(path, identity)
    record['status'] = 'accepted'
    path.write_text(json.dumps(record))
    require_acceptance(path, identity)
    record['pool_sha256'] = 'changed'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='stale'):
        require_acceptance(path, identity)


def test_fixed_step_budget_independent_of_dataset_size_and_reproducible():
    for count in (7, 100):
        data = TensorDataset(torch.arange(count))
        a = list(fixed_step_loader(data, 9, 8, 42))
        b = list(fixed_step_loader(data, 9, 8, 42))
        assert len(a) == 9 and all(len(x[0]) == 8 for x in a)
        assert torch.equal(torch.cat([x[0] for x in a]), torch.cat([x[0] for x in b]))


def test_false_positive_metric_exposes_any_pixel_recall_limitation():
    y = np.zeros((3, 1, 2, 2), dtype=np.float32)
    y[0, 0, 0, 0] = 1
    p = np.zeros_like(y)
    p[0, 0, 1, 1] = 1  # Wrong location still counts as image-level recall.
    p[1, 0, 0, 0] = 1
    result = evaluate(p, y)
    assert result['defect_recall'] == 1
    assert result['image_false_positive_rate'] == .5
    assert result['dice'] == 0
    assert evaluate(p[:1], y[:1])['image_false_positive_rate'] is None


@pytest.mark.parametrize('method', ['real_only', 'procedural'])
def test_formal_trainer_writes_schema_valid_fixed_step_result(tmp_path, monkeypatch, method):
    from types import SimpleNamespace
    from tests.test_dataset_integrity import _dataset
    from src.data.integrity import audit, digest, freeze_manifest
    from src.generation.base import GeneratedBatch
    import scripts.train_inspector as trainer
    _dataset(tmp_path)
    rows, _ = audit(tmp_path, ['metal_nut'])
    defects = [r for r in rows if r['defect_type'] == 'scratch']
    body = dict(schema_version=1, status='candidate', categories=['metal_nut'],
                n_support_requested_per_defect_type=5, seed=42,
                protocol_name='5-shot per defect type, seed 42',
                train_normal=[r for r in rows if r['split'] == 'train'],
                test_normal=[r for r in rows if r['split'] == 'test' and r['defect_type'] == 'good'],
                support=defects[:5], heldout=defects[5:])
    path = tmp_path/'split.json'
    path.write_text(json.dumps(dict(body, manifest_sha256=digest(body))))
    manifest, _ = freeze_manifest(path, tmp_path, 'A', 'Fixture only')
    config = copy.deepcopy(settings())
    config['split_sha256'] = manifest['manifest_sha256']
    config['optimizer_steps']['metal_nut'] = 2
    monkeypatch.setattr(trainer, 'settings', lambda: config)
    monkeypatch.setattr(trainer, 'build_model', lambda: torch.nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(trainer.subprocess, 'check_output', lambda argv, **kw: '' if '--porcelain' in argv else 'fixture-commit')
    acceptance = tmp_path/'acceptance.json'; acceptance.write_text('{}')
    if method == 'procedural':
        batch = GeneratedBatch(np.zeros((2,3,256,256),np.float32), np.zeros((2,1,256,256),np.float32),
                               [dict(method='procedural', category='metal_nut', defect_type='synthetic_patch',
                                     source_image_id='metal_nut/train/good/0.png', seed=42,
                                     generation_seconds=.1, generation_config={}) for _ in range(2)])
        monkeypatch.setattr(trainer, 'load_formal_pool', lambda *a: (batch, {'pool_sha256':'fixture'}, [{'generation_seconds':.2}]))
        monkeypatch.setattr(trainer, 'require_acceptance', lambda *a: {})
    args = SimpleNamespace(root=tmp_path, manifest=path, category='metal_nut', method=method, formal=True,
                           bundle=['fixture'] if method == 'procedural' else [],
                           acceptance=acceptance if method == 'procedural' else None,
                           synthetic_count=0, seed=42, image_size=256, batch_size=8, learning_rate=1e-4,
                           epochs=50, device='cpu', num_threads=1, num_workers=0,
                           out=tmp_path/'result.json', checkpoint=tmp_path/'weights.pt')
    result = trainer.train(args)
    assert result['training']['steps'] == 2
    assert result['training']['sampler'] == 'uniform_with_replacement'
    assert result['synthetic_count'] == (2 if method == 'procedural' else 0)
    assert result['evaluation']['count'] == 2
    assert result['inspection']['image_false_positive_rate'] is not None
    assert args.out.exists() and args.checkpoint.exists()


def test_summary_paired_deltas_and_missing_runs():
    from scripts.summarize_formal_results import summarize
    config = settings()
    rows = []
    for category in config['optimizer_steps']:
        for method in ['real_only', 'procedural']:
            for seed in config['downstream_seeds']:
                rows.append(dict(schema_version=1, experiment_id=f'{category}-{method}-{seed}', category=category,
                                 method=method, seed=seed, real_train_count=235,
                                 synthetic_count=0 if method=='real_only' else config['synthetic_counts'][category],
                                 generator={'generation_seconds_per_image':0., 'config':{'pool':{'pool_sha256':'same'}}},
                                 inspection=dict(pixel_auroc=.5 if method=='real_only' else .75, dice=.4, iou=.25,
                                                 defect_recall=.9, image_false_positive_rate=.1),
                                 training=dict(architecture='unet',encoder='resnet18',encoder_weights=None,image_size=256,
                                               steps=config['optimizer_steps'][category],learning_rate=1e-4,loss=.1,
                                               sampler='uniform_with_replacement',batch_size=8),
                                 evaluation=dict(source='heldout_real',count=77,threshold=.5,smoke_only=False,
                                                 split_sha256=config['split_sha256'],formal_protocol=config['version']),
                                 provenance=dict(git_commit='test',git_dirty=False,python='3.11',torch='2.8',cuda=None,
                                                 device='cpu',runtime_seconds=1.,packages={})))
    summary=summarize(rows)
    procedural=summary['methods'][1]
    assert procedural['categories']['metal_nut']['paired_delta_vs_real_only']['pixel_auroc']['mean']==.25
    assert procedural['equal_category_macro']['pixel_auroc']['mean']==.75
    assert summary['methods'][2]['equal_category_macro'] is None
    assert summary['methods'][2]['categories']['screw']['status']=='incomplete'
    with pytest.raises(ValueError,match='Duplicate'):
        summarize(rows+[rows[0]])

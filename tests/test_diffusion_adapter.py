"""Diffusion adapter contract tests with a fake worker (no GPU, no upstream code)."""
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.generation.base import validate_batch
from src.generation.diffusion_adapter import DiffusionDefectGenerator, DiffusionUnderfilledError
from src.generation.diffusion_foreground import _thresholded_foreground, placement_foreground
from src.generation.qc import estimate_object_foreground

ROOT = Path(__file__).resolve().parents[1]

FAKE = r'''#!{python}
import json, os, sys
import numpy as np
from PIL import Image
script, arg = sys.argv[1], sys.argv[-1]
if script.endswith("support_stats.py"):
    print(json.dumps({{"area_pct_256": [0.2, 0.3, 0.25, 0.4, 0.22], "latent_cells": [3, 4, 3, 6, 3], "torchvision": "fake"}}))
    sys.exit(0)
req = json.load(open(arg))
out = req["out_dir"]
mode = os.environ.get("FAKE_MODE", "ok")
os.makedirs(os.path.join(out, "images")); os.makedirs(os.path.join(out, "masks"))
samples = req["samples"] if mode == "ok" else req["samples"][:1]
recs = []
for s in samples:
    fg = np.asarray(Image.open(s["foreground_png"])) > 127
    ys, xs = np.nonzero(fg); cy, cx = int(ys.mean()), int(xs.mean())
    m = np.zeros((256, 256), np.uint8); m[cy - 3:cy + 3, cx - 3:cx + 3] = 255
    src = np.asarray(Image.open(os.path.join(req["mvtec_root"], s["source_image_id"])).convert("RGB").resize((256, 256)))
    img = src.copy(); img[m > 0] = 0
    Image.fromarray(img).save(os.path.join(out, "images", "%05d.png" % s["index"]))
    Image.fromarray(m).save(os.path.join(out, "masks", "%05d.png" % s["index"]))
    recs.append({{"index": s["index"], "source_image_id": s["source_image_id"], "mask_seconds": 1.0,
                 "inpaint_seconds": 0.5, "generation_seconds": 1.5, "mask_attempt": 3,
                 "placement_tries": 1, "placement_overlap": 1.0, "latent_cells": 4}})
json.dump({{"name": "x", "n": len(recs), "requested": len(req["samples"]),
           "status": "ok" if mode == "ok" else "underfilled", "mask_attempts": 30,
           "stage1_upstream_check_mask_rejected": 2,
           "diffusion_internal_filter": {{"rules_version": "mask_filter_v2", "rejected": {{"inverted": 5}}}},
           "mask_sampling_seconds_total": 10.0, "mask_model_load_seconds": 1.0, "image_model_load_seconds": 1.0,
           "provenance": {{"worker_sha256": "fake"}}, "samples": recs}}, open(os.path.join(out, "result.json"), "w"))
'''


def _object_image(path, size=300, shift=0):
    img = np.full((size, size, 3), 200, np.uint8)
    img[100 + shift:200 + shift, 120:180] = 60
    Image.fromarray(img).save(path)


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / 'mvtec'
    good = data / 'screw/train/good'; good.mkdir(parents=True)
    gt = data / 'screw/ground_truth/thread_side'; gt.mkdir(parents=True)
    normals = []
    for i in range(4):
        _object_image(good / f'{i:03d}.png', shift=i * 5)
        normals.append({'category': 'screw', 'image_path': f'screw/train/good/{i:03d}.png'})
    support = []
    for i in range(5):
        Image.fromarray(np.zeros((300, 300), np.uint8)).save(gt / f'{i:03d}_mask.png')
        support.append({'category': 'screw', 'defect_type': 'thread_side',
                        'mask_path': f'screw/ground_truth/thread_side/{i:03d}_mask.png'})
    manifest = {'support': support, 'train_normal': normals}
    ckpt = tmp_path / 'ckpt'; ckpt.mkdir()
    for name in ('ldm.ckpt', 'emb.pt', 'enc.pt', 'screw-thread_side.pt'):
        (ckpt / name).write_bytes(b'x')
    fake = tmp_path / 'fake_python'
    fake.write_text(FAKE.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    config = {'upstream': str(tmp_path), 'python': str(fake), 'root': str(data),
              'work_dir': str(tmp_path / 'raw'),
              'checkpoints': {'ldm': str(ckpt / 'ldm.ckpt'), 'anomaly_embeddings': str(ckpt / 'emb.pt'),
                              'spatial_encoder': str(ckpt / 'enc.pt'),
                              'mask_embeddings_template': str(ckpt / '{category}-{defect_type}.pt')},
              'sampling': {'mask_ddim_steps': 100, 'image_ddim_steps': 200, 'eta': 1.0, 'adaptive_mask': True,
                           'batch_size': 8, 'mask_attempt_factor': 50},
              'mask_filter_rules': 'configs/diffusion/mask_filter_v2.json',
              'placement_rules': 'configs/diffusion/placement_v2.json', 'save_rejects': True}
    images = np.stack([np.asarray(Image.open(data / r['image_path']).convert('RGB').resize((256, 256)),
                                  np.float32).transpose(2, 0, 1) / 255 for r in normals])
    return config, manifest, images, [r['image_path'] for r in normals], tmp_path


def test_generate_with_shared_qc_contract_and_provenance(setup):
    config, manifest, images, ids, _ = setup
    g = DiffusionDefectGenerator(config, manifest)
    result = g.generate_with_qc(images, ids, target_count=3, defect_type='thread_side', seed=42, category='screw')
    batch = validate_batch(result.require_target(), 3)
    meta = batch.metadata[0]
    assert meta['method'] == 'diffusion' and meta['defect_type'] == 'thread_side'
    assert meta['source_image_id'] in ids and meta['generation_seconds'] == 1.5
    cfg = meta['generation_config']
    assert cfg['placement'] == 'v2' and cfg['placement_foreground'] == 'core_thr3_fill_and_qc'
    assert cfg['mask_filter']['min_latent_cells'] == 3          # max(2, min support cells 3)
    assert cfg['stage1']['mask_attempts'] == 30 and cfg['stage1']['internal_rejected'] == {'inverted': 5}
    request = json.loads((Path(cfg['raw_output_dir']) / 'request.json').read_text())
    assert request['save_rejects'] is True
    assert request['placement_rules']['rules_version'] == 'placement_v2'
    assert len(g.runs) == 1 and result.report.accepted_count == 3


def test_flip_and_foreign_sources_are_rejected(setup):
    config, manifest, images, ids, _ = setup
    g = DiffusionDefectGenerator(config, manifest)
    with pytest.raises(ValueError, match='Out-of-scope'):
        g.generate(images, 1, defect_type='flip', category='metal_nut', source_image_ids=ids)
    bad = list(ids); bad[0] = 'screw/test/good/000.png'
    with pytest.raises(ValueError, match='train/good'):
        g.generate(images, 1, defect_type='thread_side', category='screw', source_image_ids=bad)


def test_underfilled_worker_raises_and_keeps_evidence(setup, monkeypatch):
    config, manifest, images, ids, tmp = setup
    monkeypatch.setenv('FAKE_MODE', 'underfill')
    g = DiffusionDefectGenerator(config, manifest)
    with pytest.raises(DiffusionUnderfilledError) as err:
        g.generate(images, 3, defect_type='thread_side', seed=7, category='screw', source_image_ids=ids)
    assert (err.value.run_dir / 'result.json').exists() and g.runs[0]['produced'] == 1


def test_draft_rules_and_in_repo_outputs_are_refused(setup):
    config, manifest, _, _, tmp = setup
    draft = tmp / 'mask_filter_draft.json'
    rules = json.loads((ROOT / 'configs/diffusion/mask_filter_v2.json').read_text())
    rules['status'] = 'DRAFT - not frozen'
    draft.write_text(json.dumps(rules))
    with pytest.raises(ValueError, match='DRAFT'):
        DiffusionDefectGenerator({**config, 'mask_filter_rules': str(draft)}, manifest)
    with pytest.raises(ValueError, match='outside Git'):
        DiffusionDefectGenerator({**config, 'work_dir': str(ROOT / 'tmp_raw')}, manifest)


def test_foreground_copy_matches_shared_qc_and_core_is_subset(setup):
    _, manifest, _, _, tmp = setup
    image = np.asarray(Image.open(tmp / 'mvtec/screw/train/good/000.png').convert('RGB'),
                       np.float32).transpose(2, 0, 1) / 255
    assert np.array_equal(_thresholded_foreground(image, 1.0), estimate_object_foreground(image))
    qc = placement_foreground(image, 'qc')
    core = placement_foreground(image, 'core_thr3_fill_and_qc')
    assert core.shape == (256, 256) and core.any() and not (core & ~qc).any()

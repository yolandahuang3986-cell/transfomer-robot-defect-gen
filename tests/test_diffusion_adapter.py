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
from src.generation.diffusion_adapter import DiffusionDefectGenerator, DiffusionUnderfilledError, summarize_runs
from src.generation.diffusion_foreground import _thresholded_foreground, placement_foreground
from src.generation.qc import estimate_object_foreground

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/diffusion'))
from stage1 import run_stage1, stage1_result

FAKE = r'''#!{python}
import json, os, sys
import numpy as np
from PIL import Image
script, arg = sys.argv[1], sys.argv[-1]
if script.endswith("support_stats.py"):
    print(json.dumps({{"area_pct_256": [0.2, 0.3, 0.25, 0.4, 0.22], "latent_cells": [3, 4, 3, 6, 3], "torchvision": "fake"}}))
    sys.exit(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
from stage1 import stage1_result                     # the real statistics schema, not a copy
req = json.load(open(arg))
out = req["out_dir"]
mode = os.environ.get("FAKE_MODE", "ok")
os.makedirs(os.path.join(out, "images")); os.makedirs(os.path.join(out, "masks"))
samples = {{"underfill": req["samples"][:1], "zero": []}}.get(mode, req["samples"])
recs = []
for s in samples:
    fg = np.asarray(Image.open(s["foreground_png"])) > 127
    ys, xs = np.nonzero(fg); cy, cx = int(ys.mean()), int(xs.mean())
    m = np.zeros((256, 256), np.uint8); m[cy - 3:cy + 3, cx - 3:cx + 3] = 255
    if mode == "qc_reject" and req["seed"] == 42 and s["index"] == 0:
        m[:] = 0                                      # shared QC rejects this pair as empty_mask
    src = np.asarray(Image.open(os.path.join(req["mvtec_root"], s["source_image_id"])).convert("RGB").resize((256, 256)))
    img = src.copy(); img[m > 0] = 0
    Image.fromarray(img).save(os.path.join(out, "images", "%05d.png" % s["index"]))
    Image.fromarray(m).save(os.path.join(out, "masks", "%05d.png" % s["index"]))
    recs.append({{"index": s["index"], "source_image_id": s["source_image_id"], "mask_seconds": 1.0,
                 "inpaint_seconds": 0.5, "generation_seconds": 1.5, "mask_attempt": 3,
                 "placement_tries": 1, "placement_overlap": 1.0, "latent_cells": 4}})
n = len(req["samples"])
s1 = {{"masks": [None] * len(recs), "attempts": 30, "budget": 50 * n, "surplus": 2,
      "seconds": 1.0 * len(recs) if recs else 2.0, "reasons": {{"inverted": 5, "check_mask": 2}}}}
stats = stage1_result(n, s1, 1.0, 1.0 if recs else 0.0, 0.5 * len(recs), {{"rules_version": "mask_filter_v2"}})
json.dump({{"name": "x", **stats, "provenance": {{"worker_sha256": "fake"}}, "samples": recs}},
          open(os.path.join(out, "result.json"), "w"))
sys.exit(2 if not recs else 0)
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


# ---------- review fixes (PR #5) ----------
def _batches(size, seconds=2.0):
    calls = []

    def sample():
        calls.append(1)
        return list(range(size)), seconds
    return sample, calls


def test_stage1_budget_counts_judged_masks_not_batches():
    """Budget 50 with batches of 8 used to judge 56 masks (7 full batches)."""
    sample, calls = _batches(8)
    s1 = run_stage1(sample, lambda o, a, k: (None, 'inverted', {}), n=1, max_attempts=50)
    assert s1['attempts'] == 50 and s1['reasons'] == {'inverted': 50}
    assert len(calls) == 7 and s1['surplus'] == 6          # 56 sampled, 6 never judged
    assert s1['seconds'] == 14.0                           # the surplus was still paid for


def test_stage1_stops_at_target_and_counts_surplus():
    sample, calls = _batches(8)
    judge = lambda o, a, k: (a, None if a % 4 == 0 else 'too_small_or_empty', {'mask_attempt': a})
    s1 = run_stage1(sample, judge, n=3, max_attempts=150)
    assert s1['masks'] == [4, 8, 12] and s1['attempts'] == 12 and s1['surplus'] == 4
    assert len(calls) == 2


def test_zero_output_result_has_the_same_statistics_as_a_normal_run():
    rules = {'rules_version': 'mask_filter_v2', 'rules_sha256': 'abc'}
    empty = {'masks': [], 'infos': [], 'attempts': 50, 'budget': 50, 'surplus': 6, 'seconds': 14.0,
             'reasons': {'inverted': 40, 'check_mask': 10}}
    zero = stage1_result(1, empty, 30.0, mask_filter=rules)
    normal = stage1_result(1, {**empty, 'masks': [1], 'attempts': 7}, 30.0, 40.0, 5.0, rules)
    assert zero.keys() == normal.keys()
    assert zero['status'] == 'underfilled' and zero['mask_model_load_seconds'] == 30.0
    assert zero['mask_reject_reasons'] == {'inverted': 40, 'check_mask': 10}
    assert zero['diffusion_internal_filter']['rejected'] == {'inverted': 40}
    assert zero['generation_seconds_total'] == 14.0 and normal['generation_seconds_total'] == 19.0


def test_zero_output_run_keeps_reasons_and_load_cost(setup, monkeypatch):
    config, manifest, images, ids, _ = setup
    monkeypatch.setenv('FAKE_MODE', 'zero')
    g = DiffusionDefectGenerator(config, manifest)
    with pytest.raises(DiffusionUnderfilledError):
        g.generate(images, 2, defect_type='thread_side', seed=7, category='screw', source_image_ids=ids)
    run = g.runs[0]
    assert run['produced'] == 0 and run['model_load_seconds'] == 1.0
    assert run['rejected_by_reason'] == {'inverted': 5, 'check_mask': 2}
    totals = summarize_runs(g.runs)
    assert totals['rejected_by_reason'] == {'inverted': 5, 'check_mask': 2}
    assert totals['model_load_seconds_total'] == 1.0 and totals['generation_seconds_total'] == 2.0


def test_cost_counts_pairs_rejected_by_shared_qc(setup, monkeypatch):
    """3 pairs at 1.5 s, one rejected by shared QC: cost is 4.5 s, not the 3 s of the accepted metadata."""
    config, manifest, images, ids, _ = setup
    monkeypatch.setenv('FAKE_MODE', 'qc_reject')
    g = DiffusionDefectGenerator(config, manifest)
    result = g.generate_with_qc(images, ids, target_count=3, defect_type='thread_side', seed=42, category='screw')
    assert result.report.rejection_counts['empty_mask'] == 1 and result.report.accepted_count == 3
    first_call = g.runs[0]
    assert first_call['produced'] == 3 and first_call['generation_seconds_total'] == 4.5
    totals = summarize_runs(g.runs)                         # second call (seed 43) refills one pair
    assert totals['worker_calls'] == 2 and totals['generation_seconds_total'] == 6.0
    assert totals['generation_seconds_total'] == result.report.generation_seconds
    assert sum(m['generation_seconds'] for m in result.batch.metadata) == 4.5   # what the old summary used

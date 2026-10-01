import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from src.data.integrity import audit, digest, freeze_manifest, validate_manifest, validate_freeze_attestation, write_new_json
from src.data.mvtec import scan_dataset
from src.data.splits import build_fewshot_manifest


def _dataset(root):
    base = root / 'metal_nut'
    (base / 'train/good').mkdir(parents=True)
    (base / 'test/good').mkdir(parents=True)
    (base / 'test/scratch').mkdir(parents=True)
    (base / 'ground_truth/scratch').mkdir(parents=True)
    for i in range(6):
        arr = np.full((8, 8, 3), i + 1, dtype=np.uint8)
        Image.fromarray(arr).save(base / f'test/scratch/{i}.png')
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        Image.fromarray(mask).save(base / f'ground_truth/scratch/{i}_mask.png')
    for value, split in ((250, 'train'), (240, 'test')):
        Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(base / f'{split}/good/0.png')


def test_audit_and_manifest_verify_content_and_freeze_overwrite(tmp_path):
    _dataset(tmp_path)
    rows, counts = audit(tmp_path, ['metal_nut'])
    assert len(rows) == 8
    records = scan_dataset(tmp_path, ['metal_nut'])
    split = build_fewshot_manifest(records, n_support=5, seed=42)
    assert (len(split['support']), len(split['heldout'])) == (5, 1)
    by_path = {r['image_path']: r for r in rows}
    body = {'schema_version': 1,
            'train_normal': [r for r in rows if r['split'] == 'train'],
            'test_normal': [r for r in rows if r['split'] == 'test' and r['defect_type'] == 'good'],
            'support': [by_path[Path(r['image_path']).relative_to(tmp_path).as_posix()] for r in split['support']],
            'heldout': [by_path[Path(r['image_path']).relative_to(tmp_path).as_posix()] for r in split['heldout']]}
    manifest = {**body, 'manifest_sha256': digest(body)}
    validate_manifest(manifest, tmp_path)
    image = tmp_path / rows[0]['image_path']
    image.write_bytes(image.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Data content changed'):
        validate_manifest(manifest, tmp_path)


def test_candidate_write_is_exclusive(tmp_path):
    p = tmp_path / 'candidate.json'
    write_new_json(p, {'status': 'candidate'})
    with pytest.raises(FileExistsError):
        write_new_json(p, {'status': 'replacement'})
    assert json.loads(p.read_text()) == {'status': 'candidate'}


def test_freeze_requires_matching_manifest_and_reviewer():
    manifest = {'status': 'frozen', 'manifest_sha256': 'a' * 64}
    good = {'status': 'frozen', 'manifest_sha256': 'a' * 64,
            'reviewer': 'reviewer', 'reason': 'counts checked'}
    assert validate_freeze_attestation(manifest, good) == good
    with pytest.raises(ValueError, match='not attested'):
        validate_freeze_attestation(manifest, {**good, 'status': 'candidate'})
    with pytest.raises(ValueError, match='does not match'):
        validate_freeze_attestation(manifest, {**good, 'manifest_sha256': 'b' * 64})


def test_freeze_records_protocol_and_atomically_updates_checksum(tmp_path):
    _dataset(tmp_path)
    rows, _ = audit(tmp_path, ['metal_nut'])
    by_path = {r['image_path']: r for r in rows}
    records = scan_dataset(tmp_path, ['metal_nut'])
    split = build_fewshot_manifest(records, 5, 42)
    body = {'schema_version': 1, 'status': 'candidate',
            'n_support_requested_per_defect_type': 5, 'seed': 42,
            'protocol_name': '5-shot per defect type, seed 42',
            'train_normal': [r for r in rows if r['split'] == 'train'],
            'test_normal': [r for r in rows if r['split'] == 'test' and r['defect_type'] == 'good'],
            'support': [by_path[Path(r['image_path']).relative_to(tmp_path).as_posix()] for r in split['support']],
            'heldout': [by_path[Path(r['image_path']).relative_to(tmp_path).as_posix()] for r in split['heldout']]}
    path = tmp_path / 'split.json'
    path.write_text(json.dumps({**body, 'manifest_sha256': digest(body)}))
    frozen, attestation = freeze_manifest(path, tmp_path, 'Member A', 'coverage approved')
    assert frozen['status'] == 'frozen'
    assert frozen['protocol_name'] == '5-shot per defect type, seed 42'
    assert json.loads(path.read_text())['manifest_sha256'] == frozen['manifest_sha256']
    validate_freeze_attestation(frozen, attestation)
    assert path.with_suffix('.freeze.json').exists()

"""Purpose admission with complete quotas and checksum-bound legacy reviews."""
import copy
import json
import shutil

import numpy as np
import pytest

from src.data.integrity import sha256
from src.inspection.formal_bundle import load_formal_pool, require_acceptance
from tests.test_formal_evaluation import pool


@pytest.fixture(params=['procedural', 'gan', 'diffusion'])
def method_pool(pool, request, monkeypatch):
    import src.inspection.formal_bundle as module
    directory, provenance, _ = pool
    method = request.param
    config = copy.deepcopy(module.settings())
    config['conditional_samples_per_defect'] = 2
    monkeypatch.setattr(module, 'settings', lambda: config)
    rows = json.loads((directory / 'metadata.json').read_text())
    for row in rows:
        row['method'] = method
        row['defect_type'] = 'synthetic_patch' if method == 'procedural' else 'scratch'
    (directory / 'metadata.json').write_text(json.dumps(rows))
    provenance['file_sha256']['metadata.json'] = sha256(directory / 'metadata.json')
    (directory / 'provenance.json').write_text(json.dumps(provenance))
    manifest = dict(manifest_sha256='frozen',
                    train_normal=[dict(category='metal_nut', image_path=rows[0]['source_image_id'])],
                    support=[dict(category='metal_nut', defect_type='scratch')])

    def load(path=directory):
        return load_formal_pool([path], manifest, 'metal_nut', method, directory.parent)

    return method, directory, provenance, load


def write_provenance(directory, provenance):
    (directory / 'provenance.json').write_text(json.dumps(provenance))


def write_legacy_review(directory, **overrides):
    review = dict(schema_version=1, purpose='formal', reviewer='A', reviewed_at='2026-10-15',
                  notes='Fixture: original use reviewed; not a development or validation pool.',
                  file_sha256={n: sha256(directory / n) for n in
                               ('batch.npz', 'metadata.json', 'provenance.json')})
    review.update(overrides)
    (directory / 'legacy_purpose_review.json').write_text(json.dumps(review))
    return review


def test_formal_purpose_enters_existing_pool_checks(method_pool):
    _, _, _, load = method_pool
    batch, identity, _ = load()
    assert len(batch.images) == identity['synthetic_count'] == 2


@pytest.mark.parametrize('purpose', ['validation', 'unknown', None, ['formal']])
def test_rejects_explicit_nonformal_purpose_even_with_legacy_review(method_pool, purpose):
    _, directory, provenance, load = method_pool
    provenance['purpose'] = purpose
    write_provenance(directory, provenance)
    write_legacy_review(directory)
    with pytest.raises(ValueError, match='purpose'):
        load()


def test_missing_purpose_is_not_automatically_accepted(method_pool):
    _, directory, provenance, load = method_pool
    del provenance['purpose']
    write_provenance(directory, provenance)
    with pytest.raises(ValueError, match='purpose'):
        load()


def test_legacy_review_only_admits_b_formats(method_pool):
    method, directory, provenance, load = method_pool
    del provenance['purpose']
    write_provenance(directory, provenance)
    original = {n: (directory / n).read_bytes() for n in
                ('batch.npz', 'metadata.json', 'provenance.json')}
    write_legacy_review(directory)
    if method == 'diffusion':
        with pytest.raises(ValueError, match='purpose'):
            load()
    else:
        batch, identity, _ = load()
        assert len(batch.images) == 2
        assert identity['artifacts'][0]['legacy_purpose_review.json'] == sha256(directory / 'legacy_purpose_review.json')
    assert all((directory / n).read_bytes() == content for n, content in original.items())


@pytest.mark.parametrize('method_pool', ['procedural', 'gan'], indirect=True)
@pytest.mark.parametrize('changes', [dict(reviewer='C'), dict(notes='  '), dict(reviewed_at=''),
                                     dict(schema_version=True), dict(purpose='validation')])
def test_legacy_review_requires_explicit_a_decision(method_pool, changes):
    _, directory, provenance, load = method_pool
    del provenance['purpose']
    write_provenance(directory, provenance)
    write_legacy_review(directory, **changes)
    with pytest.raises(ValueError, match='purpose review'):
        load()


@pytest.mark.parametrize('method_pool', ['procedural', 'gan'], indirect=True)
@pytest.mark.parametrize('changed_file', ['batch.npz', 'metadata.json', 'provenance.json'])
def test_legacy_review_cannot_be_reused_after_bundle_changes(method_pool, changed_file):
    _, directory, provenance, load = method_pool
    del provenance['purpose']
    write_provenance(directory, provenance)
    write_legacy_review(directory)
    # Change an otherwise valid bundle, keeping its ordinary checksums current.
    if changed_file == 'batch.npz':
        with np.load(directory / changed_file) as arrays:
            images, masks = arrays['images'], arrays['masks']
        images[:, :, 80:90, 80:90] = .2
        np.savez_compressed(directory / changed_file, images=images, masks=masks)
    elif changed_file == 'metadata.json':
        rows = json.loads((directory / changed_file).read_text())
        rows[0]['generation_config']['note'] = 'Changed configuration'
        (directory / changed_file).write_text(json.dumps(rows))
    else:
        provenance['note'] = 'Changed provenance'
    if changed_file != 'provenance.json':
        provenance['file_sha256'][changed_file] = sha256(directory / changed_file)
    write_provenance(directory, provenance)
    with pytest.raises(ValueError, match='purpose review.*checksums'):
        load()


@pytest.mark.parametrize('method_pool', ['procedural', 'gan'], indirect=True)
def test_legacy_review_is_portable_and_bound_to_whole_pool_acceptance(method_pool, tmp_path):
    _, directory, provenance, load = method_pool
    del provenance['purpose']
    write_provenance(directory, provenance)
    write_legacy_review(directory)
    _, identity, _ = load()
    copied = tmp_path / 'copied_bundle'
    shutil.copytree(directory, copied)
    assert load(copied)[1] == identity
    acceptance = directory / 'acceptance.json'
    record = dict(identity, status='pending_review', reviewer='A', reviewed_at=None, notes='',
                  training_provenance_verified=False)
    acceptance.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Explicit Member A'):
        require_acceptance(acceptance, identity)
    record.update(status='accepted', reviewed_at='2026-10-15', notes='Fixture quality and provenance review.',
                  training_provenance_verified=True)
    acceptance.write_text(json.dumps(record))
    require_acceptance(acceptance, identity)
    write_legacy_review(directory, notes='Fixture purpose review revised.')
    _, revised, _ = load()
    assert revised['pool_sha256'] != identity['pool_sha256']
    with pytest.raises(ValueError, match='stale or mismatched'):
        require_acceptance(acceptance, revised)

"""ratio_tier_count: admits a secondary ratio-analysis quota without ever
changing what the primary (unparameterized) call enforces."""
import copy
import json

import numpy as np
import pytest
from PIL import Image

from src.data.integrity import sha256
from src.experiments.formal import settings
from src.inspection.formal_bundle import load_formal_pool, require_acceptance


def _build_bundle(tmp_path, count, monkeypatch, frozen_count=2, seed=42, name='bundle'):
    import src.inspection.formal_bundle as module
    config = copy.deepcopy(settings())
    config['synthetic_counts']['metal_nut'] = frozen_count
    config['split_sha256'] = 'frozen'
    monkeypatch.setattr(module, 'settings', lambda: config)
    source = 'metal_nut/train/good/000.png'
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[32:224, 32:224] = 200
    path = tmp_path / source
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
    directory = tmp_path / name
    directory.mkdir()
    images = np.repeat((image.transpose(2, 0, 1).astype(np.float32) / 255)[None], count, axis=0)
    masks = np.zeros((count, 1, 256, 256), dtype=np.float32)
    masks[:, :, 80:90, 80:90] = 1
    images[:, :, 80:90, 80:90] = .1
    records = [dict(method='procedural', category='metal_nut', defect_type='synthetic_patch',
                    seed=seed, source_image_id=source, generation_seconds=.1,
                    generation_config={'donor_image_id': source}) for _ in range(count)]
    np.savez_compressed(directory / 'batch.npz', images=images, masks=masks)
    (directory / 'metadata.json').write_text(json.dumps(records))
    provenance = dict(purpose='formal', category='metal_nut', split_sha256='frozen', seed=seed, num_samples=count,
                      file_sha256={n: sha256(directory / n) for n in ('batch.npz', 'metadata.json')},
                      shared_qc=dict(policy_version='synthetic-qc-v1', target_met=True, target_count=count,
                                     accepted_count=count, attempted_count=count + 1,
                                     generation_seconds=.3, qc_seconds=.1))
    (directory / 'provenance.json').write_text(json.dumps(provenance))
    manifest = dict(manifest_sha256='frozen', train_normal=[dict(category='metal_nut', image_path=source)],
                    support=[dict(category='metal_nut', defect_type='scratch')])
    return directory, manifest


def test_default_still_enforces_the_frozen_primary_quota(tmp_path, monkeypatch):
    # count=3 but the frozen (monkeypatched) primary quota is 2: must fail exactly as before.
    directory, manifest = _build_bundle(tmp_path, 3, monkeypatch, frozen_count=2)
    with pytest.raises(ValueError, match='Formal quota mismatch'):
        load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path)


def test_ratio_tier_count_admits_a_matching_non_primary_quota(tmp_path, monkeypatch):
    directory, manifest = _build_bundle(tmp_path, 3, monkeypatch, frozen_count=2)
    batch, identity, _ = load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path,
                                          ratio_tier_count=3)
    assert len(batch.images) == 3
    assert identity['ratio_tier_count'] == 3


def test_ratio_tier_count_still_rejects_a_mismatched_count(tmp_path, monkeypatch):
    directory, manifest = _build_bundle(tmp_path, 3, monkeypatch, frozen_count=2)
    with pytest.raises(ValueError, match='Formal quota mismatch'):
        load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path, ratio_tier_count=5)


@pytest.mark.parametrize('bad', [0, -1, 1.5, '3'])
def test_ratio_tier_count_must_be_a_positive_int(tmp_path, monkeypatch, bad):
    directory, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2)
    with pytest.raises(ValueError, match='ratio_tier_count must be a positive int'):
        load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path, ratio_tier_count=bad)


def test_primary_pool_identity_and_hash_are_unaffected_by_the_new_parameter(tmp_path, monkeypatch):
    # Regression guard: adding ratio_tier_count must not change pool_sha256 (and
    # therefore must not invalidate) any already-signed primary acceptance.json.
    directory, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2)
    _, identity, _ = load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path)
    assert 'ratio_tier_count' not in identity
    directory2, manifest2 = _build_bundle(tmp_path.parent / 'other', 2, monkeypatch, frozen_count=2)
    _, identity2, _ = load_formal_pool([directory2], manifest2, 'metal_nut', 'procedural', tmp_path.parent / 'other')
    assert identity['pool_sha256'] == identity2['pool_sha256']


def test_preexisting_primary_acceptance_without_the_new_field_still_validates(tmp_path, monkeypatch):
    # Regression guard: B's and C's already-signed acceptance.json files (written
    # before this parameter existed) have no "ratio_tier_count" key at all.
    directory, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2)
    _, identity, _ = load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path)
    record = {k: v for k, v in identity.items()}  # no ratio_tier_count key, same as pre-patch records
    record.update(status='accepted', reviewer='A', reviewed_at='2026-10-17', notes='fixture',
                 training_provenance_verified=True)
    path = tmp_path / 'acceptance.json'
    path.write_text(json.dumps(record))
    require_acceptance(path, identity)  # must not raise


def test_acceptance_is_bound_to_its_own_ratio_tier(tmp_path, monkeypatch):
    directory, manifest = _build_bundle(tmp_path, 3, monkeypatch, frozen_count=2)
    _, identity_r3, _ = load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path,
                                         ratio_tier_count=3)
    record = dict(identity_r3, status='accepted', reviewer='A', reviewed_at='2026-10-18',
                 notes='fixture', training_provenance_verified=True)
    path = tmp_path / 'acceptance.json'
    path.write_text(json.dumps(record))
    # Matches its own tier.
    require_acceptance(path, identity_r3)
    # A primary-tier identity (no ratio_tier_count) must not accept it.
    primary_identity = {k: v for k, v in identity_r3.items() if k != 'ratio_tier_count'}
    primary_identity['pool_sha256'] = 'different-pool-entirely'
    with pytest.raises(ValueError, match='stale or mismatched'):
        require_acceptance(path, primary_identity)


# ---------- ratio_append_seeds: the frozen "r=10 keeps the primary pool and
# appends under a fresh, disclosed seed" construction (protocol §4) ----------

def test_default_seed_check_is_unaffected_by_the_new_parameter(tmp_path, monkeypatch):
    # A seed-142 bundle must still be rejected exactly as before when
    # ratio_append_seeds is not passed at all.
    directory, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2, seed=142)
    with pytest.raises(ValueError, match='base generation seed 42'):
        load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path)


def test_ratio_append_seeds_requires_ratio_tier_count(tmp_path, monkeypatch):
    directory, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2, seed=142)
    with pytest.raises(ValueError, match='only valid with an explicit ratio_tier_count'):
        load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path,
                         ratio_append_seeds=[142])


@pytest.mark.parametrize('bad', [[], [1.5], ['x']])
def test_ratio_append_seeds_must_be_nonempty_ints(tmp_path, monkeypatch, bad):
    directory, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2)
    with pytest.raises(ValueError, match='nonempty list of ints'):
        load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path,
                         ratio_tier_count=2, ratio_append_seeds=bad)


def test_ratio_append_seeds_admits_primary_plus_a_declared_extra_seed_bundle(tmp_path, monkeypatch):
    # The real construction: keep the primary (seed-42) pool unchanged, append
    # new samples generated under a fresh, disclosed seed, combine via two bundles.
    primary_dir, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2, seed=42, name='primary')
    append_dir, _ = _build_bundle(tmp_path, 3, monkeypatch, frozen_count=2, seed=142, name='append')
    batch, identity, _ = load_formal_pool([primary_dir, append_dir], manifest, 'metal_nut', 'procedural', tmp_path,
                                          ratio_tier_count=5, ratio_append_seeds=[142])
    assert len(batch.images) == 5
    assert identity['ratio_append_seeds'] == [142]


def test_ratio_append_seeds_rejects_an_undeclared_seed(tmp_path, monkeypatch):
    primary_dir, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2, seed=42, name='primary')
    append_dir, _ = _build_bundle(tmp_path, 3, monkeypatch, frozen_count=2, seed=142, name='append')
    # Declaring a different seed than the one the append bundle actually used must still fail.
    with pytest.raises(ValueError, match='base generation seed 42 or a declared ratio_append_seed'):
        load_formal_pool([primary_dir, append_dir], manifest, 'metal_nut', 'procedural', tmp_path,
                         ratio_tier_count=5, ratio_append_seeds=[999])


def test_primary_pool_identity_unaffected_by_ratio_append_seeds_being_available(tmp_path, monkeypatch):
    # Regression guard, same spirit as the ratio_tier_count one: not using the
    # parameter must not change a primary pool's identity shape or pool_sha256.
    directory, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2)
    _, identity, _ = load_formal_pool([directory], manifest, 'metal_nut', 'procedural', tmp_path)
    assert 'ratio_append_seeds' not in identity


def test_acceptance_is_bound_to_its_declared_append_seeds(tmp_path, monkeypatch):
    primary_dir, manifest = _build_bundle(tmp_path, 2, monkeypatch, frozen_count=2, seed=42, name='primary')
    append_dir, _ = _build_bundle(tmp_path, 3, monkeypatch, frozen_count=2, seed=142, name='append')
    _, identity, _ = load_formal_pool([primary_dir, append_dir], manifest, 'metal_nut', 'procedural', tmp_path,
                                      ratio_tier_count=5, ratio_append_seeds=[142])
    record = dict(identity, status='accepted', reviewer='A', reviewed_at='2026-10-19',
                 notes='fixture', training_provenance_verified=True)
    path = tmp_path / 'acceptance.json'
    path.write_text(json.dumps(record))
    require_acceptance(path, identity)
    # An identity for the same pool_sha256 but a different declared seed set must not match
    # (constructed here as a stale record claiming no append seeds were used).
    stale = {k: v for k, v in identity.items() if k != 'ratio_append_seeds'}
    with pytest.raises(ValueError, match='stale or mismatched'):
        require_acceptance(path, stale)

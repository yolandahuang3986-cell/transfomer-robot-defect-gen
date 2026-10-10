"""Fail-closed, method-neutral admission of frozen synthetic pools."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

from src.data.integrity import sha256
from src.experiments.formal import ROOT, settings
from src.experiments.scope import filter_in_scope, require_in_scope
from src.generation.base import GeneratedBatch, validate_batch
from src.generation.qc import DEFAULT_QC_POLICY, estimate_object_foreground, filter_generated_batch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def policy_hashes():
    return {name: sha256(ROOT / name) for name in
            ('src/generation/qc.py', 'configs/experiment_scope.json', 'configs/experiments.yaml')}


def _integer(value):
    return type(value) is int


def load_formal_pool(directories, manifest, category, method, root):
    """Validate artifacts and re-run shared QC; never silently filter the pool.

    Every directory contains batch.npz, metadata.json and provenance.json.
    The provenance follows B's bundle format, extended to any generator method.
    Sources use the exporter's RGB bilinear resize to the native generated grid.
    """
    protocol = settings()
    if category not in protocol['synthetic_counts'] or method not in {'procedural', 'gan', 'diffusion'}:
        raise ValueError('Unsupported formal category/method')
    if manifest['manifest_sha256'] != protocol['split_sha256']:
        raise ValueError('Formal pool requires the approved frozen manifest')
    if not directories:
        raise ValueError('Formal synthetic experiments require bundles')
    allowed = {r['image_path']: r for r in manifest['train_normal'] if r['category'] == category}
    strata = sorted({r['defect_type'] for r in filter_in_scope(manifest['support'])
                     if r['category'] == category})
    if len({Path(d).resolve() for d in directories}) != len(directories):
        raise ValueError('Duplicate bundle directory')
    images, masks, metadata, artifacts, reports = [], [], [], [], []
    foregrounds = {}
    for directory in sorted({Path(d).resolve() for d in directories},
                            key=lambda d: sha256(d / 'provenance.json')):
        provenance = json.loads((directory / 'provenance.json').read_text())
        if provenance.get('split_sha256') != manifest['manifest_sha256'] or provenance.get('category') != category:
            raise ValueError('Bundle split/category mismatch')
        if provenance.get('seed') != protocol['generation_seed']:
            raise ValueError('Bundle must record base generation seed 42')
        for name in ('batch.npz', 'metadata.json'):
            if sha256(directory / name) != provenance.get('file_sha256', {}).get(name):
                raise ValueError('Bundle checksum mismatch')
        with np.load(directory / 'batch.npz', allow_pickle=False) as arrays:
            batch = GeneratedBatch(arrays['images'], arrays['masks'],
                                   json.loads((directory / 'metadata.json').read_text()))
        count = provenance.get('num_samples')
        if not _integer(count) or count < 1:
            raise ValueError('Positive bundle sample count required')
        validate_batch(batch, count)
        if batch.images.shape[2:] != (256, 256):
            raise ValueError('Formal export requires 256x256 images and masks')
        report = provenance.get('shared_qc')
        if not isinstance(report, dict) or report.get('policy_version') != DEFAULT_QC_POLICY.version:
            raise ValueError('Missing or incompatible shared QC report')
        attempted = report.get('attempted_count')
        if (report.get('target_met') is not True or report.get('accepted_count') != count
                or report.get('target_count') != count or not _integer(attempted)
                or not count <= attempted <= count * protocol['max_shared_qc_attempt_multiplier']):
            raise ValueError('Underfilled or invalid QC attempt budget')
        for name in ('generation_seconds', 'qc_seconds'):
            value = report.get(name)
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError('Missing or invalid QC cost record')
        for row in batch.metadata:
            if row['category'] != category or row['method'] != method:
                raise ValueError('Unexpected sample method/category')
            defect = row['defect_type']
            require_in_scope(category, defect)
            if defect not in (['synthetic_patch'] if method == 'procedural' else strata):
                raise ValueError('Unexpected formal defect type')
            source = row['source_image_id']
            if source not in allowed:
                raise ValueError('Source is not frozen train/good')
            donor = row['generation_config'].get('donor_image_id')
            if (method == 'procedural' and donor not in allowed) or (donor is not None and donor not in allowed):
                raise ValueError('Donor is not frozen train/good')
            if source not in foregrounds:
                with Image.open(Path(root) / source) as im:
                    array = np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
                foregrounds[source] = estimate_object_foreground(array.transpose(2, 0, 1) / 255)
        if method != 'procedural' and (len({r['defect_type'] for r in batch.metadata}) != 1
                or count != protocol['conditional_samples_per_defect']):
            raise ValueError('Conditional exports require one quota-sized bundle per defect type')
        checked = filter_generated_batch(batch, foregrounds)
        if checked.report.accepted_count != count:
            raise ValueError('Shared QC recheck failed; whole bundle must be reviewed')
        images.append(batch.images); masks.append(batch.masks); metadata.extend(batch.metadata)
        artifacts.append({name: sha256(directory / name) for name in
                          ('batch.npz', 'metadata.json', 'provenance.json')})
        reports.append(report)
    counts = Counter(row['defect_type'] for row in metadata)
    expected = ({'synthetic_patch': protocol['synthetic_counts'][category]} if method == 'procedural'
                else {name: protocol['conditional_samples_per_defect'] for name in strata})
    if dict(counts) != expected:
        raise ValueError(f'Formal quota mismatch: expected {expected}, got {dict(counts)}')
    batch = GeneratedBatch(np.concatenate(images), np.concatenate(masks), metadata)
    # Canonical artifact order makes identity independent of mount paths/argument order.
    artifacts.sort(key=fingerprint)
    identity = dict(protocol_version=protocol['version'], category=category, method=method,
                    split_sha256=manifest['manifest_sha256'], artifacts=artifacts,
                    policy_hashes=policy_hashes(), synthetic_count=len(metadata))
    identity['pool_sha256'] = fingerprint(identity)
    return batch, identity, reports


def require_acceptance(path, identity):
    record = json.loads(Path(path).read_text())
    for key in ('pool_sha256', 'protocol_version', 'category', 'method', 'policy_hashes'):
        if record.get(key) != identity[key]:
            raise ValueError(f'Acceptance is stale or mismatched: {key}')
    if record.get('status') != 'accepted' or record.get('reviewer') != 'A':
        raise ValueError('Explicit Member A whole-pool acceptance required')
    if not record.get('reviewed_at') or not record.get('notes'):
        raise ValueError('Acceptance requires review date and findings')
    if record.get('training_provenance_verified') is not True:
        raise ValueError('A must verify generator training provenance, including checkpoint leakage')
    return record

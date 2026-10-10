"""Verify the training-only GAN stage against the frozen manifest."""
import json
from pathlib import Path
from src.data.integrity import validate_manifest, validate_freeze_attestation, sha256


def validate_stage(stage, manifest_path, root):
    stage, manifest_path = Path(stage), Path(manifest_path)
    manifest = validate_manifest(json.loads(manifest_path.read_text()), root)
    validate_freeze_attestation(manifest, json.loads(manifest_path.with_suffix('.freeze.json').read_text()))
    p = json.loads((stage/'provenance.json').read_text())
    if p['split_sha256'] != manifest['manifest_sha256']:
        raise ValueError('GAN stage split mismatch')
    expected = {}
    for row in manifest['train_normal']:
        if row['category'] == p['category']:
            expected[row['image_path']] = ('A', row)
    support = [x for x in manifest['support'] if x['category']==p['category'] and x['defect_type']==p['defect_type']]
    if len(support) != manifest['n_support_requested_per_defect_type']:
        raise ValueError('Incorrect GAN support count')
    for row in support: expected[row['image_path']] = ('B', row)
    if p['support_count'] != len(support) or p['normal_count'] != len(expected)-len(support):
        raise ValueError('GAN stage declared counts mismatch')
    actual_sources = [x['source_image_id'] for x in p['mapping']]
    if len(set(actual_sources)) != len(actual_sources) or set(actual_sources) != set(expected):
        raise ValueError('GAN sources differ from frozen normal/support rows')
    targets = set()
    for entry in p['mapping']:
        domain, row = expected[entry['source_image_id']]
        target = Path(entry['target'])
        if target.is_absolute() or '..' in target.parts or target.parent.as_posix() != 'train/'+domain:
            raise ValueError('Invalid GAN stage path')
        if sha256(stage/target) != row['image_sha256']:
            raise ValueError('GAN image checksum mismatch')
        targets.add(target.as_posix())
        if domain == 'B':
            mask_target = Path(entry['mask_target'])
            if mask_target.as_posix() != 'train/mask/'+target.stem+'_mask.png':
                raise ValueError('GAN mask ordering mismatch')
            if entry['source_mask_id'] != row['mask_path'] or sha256(stage/mask_target) != row['mask_sha256']:
                raise ValueError('GAN mask checksum mismatch')
            targets.add(mask_target.as_posix())
    files = {f.relative_to(stage).as_posix() for f in (stage/'train').rglob('*') if f.is_file()}
    if files != targets:
        raise ValueError('Unexpected or missing GAN training files')
    return p

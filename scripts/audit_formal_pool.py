#!/usr/bin/env python3
"""Recheck a candidate synthetic pool and prepare A's human review record."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.integrity import validate_manifest, validate_freeze_attestation
from src.inspection.formal_bundle import load_formal_pool


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--manifest', default='data/splits/fewshot_n5_seed42.json')
    p.add_argument('--category', required=True, choices=['metal_nut', 'screw'])
    p.add_argument('--method', required=True, choices=['procedural', 'gan', 'diffusion'])
    p.add_argument('--bundle', required=True, action='append')
    p.add_argument('--out', required=True, help='New review directory outside Git')
    p.add_argument('--ratio-tier-count', type=int, default=None,
                   help='Secondary ratio-analysis quota to admit instead of the frozen primary quota. '
                        'For procedural this is the CATEGORY total (e.g. 15 or 150 for metal_nut); for '
                        'gan/diffusion it is the PER-DEFECT-TYPE count (e.g. 5 or 50), summed across '
                        'every --bundle for that type when --ratio-append-seed is also given')
    p.add_argument('--ratio-append-seed', type=int, action='append', default=None,
                   help='Disclosed extra generation seed(s) a ratio-tier append bundle is allowed to use '
                        '(only valid together with --ratio-tier-count); repeat for multiple seeds')
    args = p.parse_args()
    out = Path(args.out).resolve()
    if out.is_relative_to(ROOT) or out.exists():
        p.error('Review directory must be new and outside the repository')
    path = Path(args.manifest)
    manifest = validate_manifest(json.loads(path.read_text()), args.root)
    validate_freeze_attestation(manifest, json.loads(path.with_suffix('.freeze.json').read_text()))
    batch, identity, reports = load_formal_pool(args.bundle, manifest, args.category, args.method, args.root,
                                                ratio_tier_count=args.ratio_tier_count,
                                                ratio_append_seeds=args.ratio_append_seed)
    groups = defaultdict(list)
    for index, row in enumerate(batch.metadata):
        groups[row['defect_type']].append(index)
    rng = np.random.default_rng(42)
    for indices in groups.values():
        rng.shuffle(indices)
    selected = []
    while len(selected) < min(20, len(batch.images)):
        for name in sorted(groups):
            if groups[name] and len(selected) < 20:
                selected.append(groups[name].pop())
    out.mkdir(parents=True)
    for rank, index in enumerate(selected):
        rgb = batch.images[index].transpose(1, 2, 0)
        mask = batch.masks[index, 0].astype(bool)
        overlay = rgb.copy()
        overlay[mask] = .5 * overlay[mask] + .5 * np.array([1., 0., 0.])
        with Image.open(Path(args.root) / batch.metadata[index]['source_image_id']) as im:
            source = np.asarray(im.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        grid = np.concatenate([source, rgb, np.repeat(mask[:, :, None], 3, axis=2), overlay], axis=1)
        Image.fromarray(np.rint(grid * 255).astype(np.uint8)).save(out / f'{rank:02d}_sample_{index:04d}.png')
    record = dict(identity, status='pending_review', reviewer='A', reviewed_at=None, notes='',
                  training_provenance_verified=False,
                  audit_samples=[dict(index=i, metadata=batch.metadata[i]) for i in selected],
                  qc_reports=reports)
    (out / 'acceptance.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(f'Automated checks passed; human review PENDING: {out / "acceptance.json"}')


if __name__ == '__main__':
    main()

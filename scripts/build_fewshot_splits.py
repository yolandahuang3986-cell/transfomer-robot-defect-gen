#!/usr/bin/env python3
"""Build a reviewed-later candidate; never overwrite a split."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.integrity import audit, digest, validate_manifest, write_new_json
from src.data.mvtec import scan_dataset
from src.data.splits import build_fewshot_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--categories', nargs='+', required=True)
    p.add_argument('--n-support', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default='data/splits')
    args = p.parse_args()
    root = Path(args.root).resolve()
    rows, counts = audit(root, args.categories)
    split = build_fewshot_manifest(scan_dataset(root, args.categories), args.n_support, args.seed)
    by_path = {r['image_path']: r for r in rows}
    manifest = {k: v for k, v in split.items() if k not in ('support', 'heldout')}
    manifest.update(schema_version=1, status='candidate', categories=sorted(args.categories),
                    counts=counts, dataset_fingerprint=digest(rows))
    for part in ('support', 'heldout'):
        manifest[part] = [by_path[Path(r['image_path']).relative_to(root).as_posix()] for r in split[part]]
    manifest['train_normal'] = [r for r in rows if r['split'] == 'train']
    manifest['test_normal'] = [r for r in rows if r['split'] == 'test' and r['defect_type'] == 'good']
    manifest['manifest_sha256'] = digest(manifest)
    validate_manifest(manifest)
    out = Path(args.out_dir) / f'fewshot_n{args.n_support}_seed{args.seed}.json'
    write_new_json(out, manifest)
    print({p: len(manifest[p]) for p in ('train_normal', 'test_normal', 'support', 'heldout')})
    print(f'Candidate saved: {out}; team count/coverage review required before freeze')


if __name__ == '__main__':
    main()

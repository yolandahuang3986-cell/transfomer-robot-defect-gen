#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec import scan_dataset
from src.data.splits import build_fewshot_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--n-support", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/splits")
    args = parser.parse_args()

    records = scan_dataset(args.root, args.categories)
    manifest = build_fewshot_manifest(records, args.n_support, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fewshot_n{args.n_support}_seed{args.seed}.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Support defect images: {len(manifest['support'])}")
    print(f"Held-out defect images: {len(manifest['heldout'])}")
    print(f"Saved split manifest: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.mvtec import scan_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--out", default="results/metrics/dataset_audit.csv")
    args = parser.parse_args()

    records = scan_dataset(args.root, args.categories)
    rows = [
        {
            "category": r.category,
            "split": r.split,
            "defect_type": r.defect_type,
            "image_path": str(r.image_path),
            "has_mask": r.mask_path is not None,
            "mask_path": str(r.mask_path) if r.mask_path else "",
        }
        for r in records
    ]

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = (
        df.groupby(["category", "split", "defect_type"], dropna=False)
        .agg(images=("image_path", "count"), masks=("has_mask", "sum"))
        .reset_index()
    )

    print(summary.to_string(index=False))
    print(f"\nSaved audit: {out}")


if __name__ == "__main__":
    main()

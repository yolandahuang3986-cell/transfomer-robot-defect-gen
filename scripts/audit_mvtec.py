#!/usr/bin/env python3
"""Audit every image/mask before allowing split construction."""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.integrity import audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--categories', nargs='+', required=True)
    p.add_argument('--out', default='results/metrics/dataset_audit.csv')
    args = p.parse_args()
    rows, counts = audit(args.root, args.categories)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in counts:
        print(row)
    print(f'AUDIT PASS: {len(rows)} images; saved {out}')


if __name__ == '__main__':
    main()

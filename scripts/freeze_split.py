#!/usr/bin/env python3
"""Record an actual review without mutating the candidate manifest."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.integrity import freeze_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--reviewer', required=True)
    p.add_argument('--reason', required=True, help='Actual count/coverage review decision')
    args = p.parse_args()
    if not args.reviewer.strip() or not args.reason.strip():
        p.error('Reviewer and review reason must not be blank')
    path = Path(args.manifest)
    manifest, record = freeze_manifest(path, args.root, args.reviewer, args.reason)
    print(f"Frozen manifest: {path}")
    print(f"Protocol: {manifest['protocol_name']}")
    print(f"Manifest SHA-256: {manifest['manifest_sha256']}")
    print(f"Freeze attestation: {path.with_suffix('.freeze.json')}")


if __name__ == '__main__':
    main()

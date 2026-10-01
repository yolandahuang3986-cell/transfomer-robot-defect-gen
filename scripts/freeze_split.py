#!/usr/bin/env python3
"""Record an actual review without mutating the candidate manifest."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.integrity import validate_manifest, write_new_json


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
    manifest = validate_manifest(json.loads(path.read_text()), args.root)
    record = dict(manifest_sha256=manifest['manifest_sha256'], status='frozen',
                  reviewer=args.reviewer, reason=args.reason,
                  reviewed_at=datetime.now(timezone.utc).isoformat())
    write_new_json(path.with_suffix('.freeze.json'), record)
    print('Freeze attestation saved; commit candidate and attestation together')


if __name__ == '__main__':
    main()

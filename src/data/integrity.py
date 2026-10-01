"""Strict, portable MVTec audit and manifest integrity checks."""
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .mvtec import scan_dataset


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def audit(root, categories):
    root = Path(root).resolve()
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("Categories must be nonempty and unique")
    records = scan_dataset(root, categories)
    rows, counts = [], Counter()
    for r in records:
        if r.split == "train" and r.defect_type != "good":
            raise ValueError(f"Training data must be normal-only: {r.image_path}")
        with Image.open(r.image_path) as image:
            image.load()
            size = image.size
        if r.defect_type != "good":
            if r.mask_path is None:
                raise ValueError(f"Missing mask: {r.image_path}")
            with Image.open(r.mask_path) as mask:
                mask.load()
                values = np.asarray(mask)
                if mask.size != size or values.ndim != 2:
                    raise ValueError(f"Invalid mask dimensions: {r.mask_path}")
                if not set(np.unique(values)).issubset({0, 255}) or not values.any():
                    raise ValueError(f"Invalid binary defect mask: {r.mask_path}")
        counts[(r.category, r.split, r.defect_type)] += 1
        rows.append({"category": r.category, "split": r.split,
                     "defect_type": r.defect_type,
                     "image_path": r.image_path.relative_to(root).as_posix(),
                     "mask_path": r.mask_path.relative_to(root).as_posix() if r.mask_path else None,
                     "image_sha256": sha256(r.image_path),
                     "mask_sha256": sha256(r.mask_path) if r.mask_path else None,
                     "width": size[0], "height": size[1]})
    for c in categories:
        if not counts[c, "train", "good"] or not counts[c, "test", "good"]:
            raise ValueError(f"Missing normal train/test images: {c}")
        if not any(k[0] == c and k[1] == "test" and k[2] != "good" for k in counts):
            raise ValueError(f"Missing test defects: {c}")
        expected = {r.mask_path.resolve() for r in records if r.category == c and r.mask_path}
        actual = {p.resolve() for p in (root / c / "ground_truth").rglob("*.png")}
        if expected != actual:
            raise ValueError(f"Orphan or missing masks: {c}")
    return rows, [{"category": c, "split": s, "defect_type": d, "count": n}
                  for (c, s, d), n in sorted(counts.items())]


PARTITIONS = ("train_normal", "test_normal", "support", "heldout")


def validate_manifest(manifest, root=None):
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if manifest.get("manifest_sha256") != digest(body):
        raise ValueError("Manifest checksum mismatch")
    if "protocol_name" in manifest:
        expected = (f"{manifest['n_support_requested_per_defect_type']}-shot per defect type, "
                    f"seed {manifest['seed']}")
        if manifest["protocol_name"] != expected:
            raise ValueError("Protocol name does not match split parameters")
    paths, hashes = set(), {}
    for partition in PARTITIONS:
        if not manifest.get(partition):
            raise ValueError(f"Empty partition: {partition}")
        for row in manifest[partition]:
            path = row["image_path"]
            if path in paths:
                raise ValueError(f"Duplicate or overlapping image: {path}")
            paths.add(path)
            is_defect = partition in ("support", "heldout")
            if is_defect != (row["defect_type"] != "good"):
                raise ValueError("Incorrect partition label")
            expected_split = "train" if partition == "train_normal" else "test"
            if row["split"] != expected_split:
                raise ValueError("Incorrect official split")
            expected_prefix = (row["category"], expected_split, row["defect_type"])
            if PurePosixPath(path).parts[:-1] != expected_prefix:
                raise ValueError("Incorrect image path semantics")
            if is_defect and not row["mask_path"]:
                raise ValueError("Defect mask required")
            if not is_defect and row["mask_path"] is not None:
                raise ValueError("Normal images must not have masks")
            prior = hashes.setdefault(row["image_sha256"], partition)
            if prior != partition:
                raise ValueError("Identical image content crosses partitions")
            for kind in ("image", "mask"):
                rel = row[f"{kind}_path"]
                if rel is None:
                    continue
                p = PurePosixPath(rel)
                if p.is_absolute() or ".." in p.parts:
                    raise ValueError("Manifest paths must be dataset-relative")
                if root is not None:
                    full = (Path(root) / rel).resolve()
                    if not full.is_relative_to(Path(root).resolve()):
                        raise ValueError("Path escapes dataset root")
                    if sha256(full) != row[f"{kind}_sha256"]:
                        raise ValueError(f"Data content changed: {rel}")
    return manifest


def validate_freeze_attestation(manifest, attestation):
    if manifest.get("status") != "frozen":
        raise ValueError("Manifest status is not frozen")
    if attestation.get("status") != "frozen":
        raise ValueError("Split is not attested as frozen")
    if attestation.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("Freeze attestation does not match manifest")
    if not str(attestation.get("reviewer", "")).strip() or not str(attestation.get("reason", "")).strip():
        raise ValueError("Freeze attestation must record reviewer and reason")
    return attestation


def freeze_manifest(path, root, reviewer, reason):
    """Validate, atomically mark a candidate frozen, and write its attestation."""
    import os
    from datetime import datetime, timezone

    path = Path(path)
    attestation_path = path.with_suffix(".freeze.json")
    if attestation_path.exists():
        raise FileExistsError(attestation_path)
    manifest = validate_manifest(json.loads(path.read_text(encoding="utf-8")), root)
    if manifest.get("status") != "candidate":
        raise ValueError("Only a candidate manifest may be frozen")
    if not reviewer.strip() or not reason.strip():
        raise ValueError("Reviewer and freeze reason must not be blank")
    frozen = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    frozen["status"] = "frozen"
    frozen["manifest_sha256"] = digest(frozen)
    # Stage beside the manifest and atomically replace only in explicit freeze action.
    temp_path = path.with_name(path.name + ".freeze-tmp")
    with temp_path.open("x", encoding="utf-8") as stream:
        json.dump(frozen, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)
    record = dict(manifest_sha256=frozen["manifest_sha256"], status="frozen",
                  reviewer=reviewer, reason=reason,
                  reviewed_at=datetime.now(timezone.utc).isoformat())
    write_new_json(attestation_path, record)
    return frozen, record


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silent regeneration of any existing artifact.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

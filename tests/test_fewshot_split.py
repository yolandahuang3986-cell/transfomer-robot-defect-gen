from pathlib import Path
from src.data.mvtec import MVTecImage
from src.data.splits import build_fewshot_manifest
import pytest


def _record(category, defect_type, idx):
    return MVTecImage(
        category=category,
        split="test",
        defect_type=defect_type,
        image_path=Path(f"{category}/test/{defect_type}/{idx:03d}.png"),
        mask_path=Path(f"{category}/ground_truth/{defect_type}/{idx:03d}_mask.png"),
    )


def test_support_and_heldout_do_not_overlap():
    records = []
    for defect in ["scratch", "bent"]:
        records.extend(_record("metal_nut", defect, i) for i in range(12))

    manifest = build_fewshot_manifest(records, n_support=5, seed=42)

    support = {x["image_path"] for x in manifest["support"]}
    heldout = {x["image_path"] for x in manifest["heldout"]}

    assert support.isdisjoint(heldout)
    assert len(support) == 10
    assert len(heldout) == 14


def test_split_is_deterministic_for_same_seed():
    records = [_record("screw", "scratch", i) for i in range(20)]

    a = build_fewshot_manifest(records, n_support=5, seed=42)
    b = build_fewshot_manifest(records, n_support=5, seed=42)

    assert a == b


def test_good_images_never_enter_defect_support():
    good = MVTecImage(
        category="screw",
        split="test",
        defect_type="good",
        image_path=Path("screw/test/good/000.png"),
        mask_path=None,
    )
    defects = [_record("screw", "scratch", i) for i in range(6)]

    manifest = build_fewshot_manifest([good, *defects], n_support=5, seed=42)

    support_paths = {x["image_path"] for x in manifest["support"]}
    heldout_paths = {x["image_path"] for x in manifest["heldout"]}
    assert str(good.image_path) not in support_paths
    assert str(defects[0].image_path) in support_paths | heldout_paths


def test_requires_nonempty_heldout_and_mask():
    with pytest.raises(ValueError, match="Insufficient held-out"):
        build_fewshot_manifest([_record("screw", "scratch", 0)], 1, 42)
    record = _record("screw", "scratch", 0)
    record = MVTecImage(record.category, record.split, record.defect_type, record.image_path, None)
    with pytest.raises(ValueError, match="Missing defect mask"):
        build_fewshot_manifest([record, _record("screw", "scratch", 1)], 1, 42)

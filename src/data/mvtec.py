from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MVTecImage:
    category: str
    split: str
    defect_type: str
    image_path: Path
    mask_path: Path | None


def _pngs(path: Path):
    return sorted(p for p in path.glob("*.png") if p.is_file())


def scan_category(root, category):
    root = Path(root)
    base = root / category
    if not base.exists():
        raise FileNotFoundError(base)

    records = []

    train = base / "train"
    if train.exists():
        for defect_dir in sorted(x for x in train.iterdir() if x.is_dir()):
            for image in _pngs(defect_dir):
                records.append(
                    MVTecImage(
                        category=category,
                        split="train",
                        defect_type=defect_dir.name,
                        image_path=image,
                        mask_path=None,
                    )
                )

    test = base / "test"
    gt = base / "ground_truth"
    if test.exists():
        for defect_dir in sorted(x for x in test.iterdir() if x.is_dir()):
            for image in _pngs(defect_dir):
                mask = None
                if defect_dir.name != "good":
                    candidate = gt / defect_dir.name / f"{image.stem}_mask.png"
                    mask = candidate if candidate.exists() else None

                records.append(
                    MVTecImage(
                        category=category,
                        split="test",
                        defect_type=defect_dir.name,
                        image_path=image,
                        mask_path=mask,
                    )
                )

    return records


def scan_dataset(root, categories):
    records = []
    for category in categories:
        records.extend(scan_category(root, category))
    return records

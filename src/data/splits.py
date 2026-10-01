import random
from collections import defaultdict


def build_fewshot_manifest(records, n_support, seed):
    """Build a leakage-safe defect support/held-out split.

    Only real defective images are eligible for support/held-out splitting.
    Sampling is stratified by category and defect type.
    """
    rng = random.Random(seed)
    if not isinstance(n_support, int) or isinstance(n_support, bool) or n_support < 1:
        raise ValueError("n_support must be a positive integer")
    groups = defaultdict(list)
    seen = set()

    for record in records:
        path = str(record.image_path)
        if path in seen:
            raise ValueError(f"Duplicate record: {path}")
        seen.add(path)
        if record.split == "test" and record.defect_type != "good":
            if record.mask_path is None:
                raise ValueError(f"Missing defect mask: {path}")
            groups[(record.category, record.defect_type)].append(record)

    if not groups:
        raise ValueError("No real defects found")

    support = []
    heldout = []

    for (category, defect_type), items in sorted(groups.items()):
        items = sorted(items, key=lambda item: str(item.image_path))
        rng.shuffle(items)
        if len(items) <= n_support:
            raise ValueError(f"Insufficient held-out defects: {category}/{defect_type}")
        k = n_support

        for bucket, subset in ((support, items[:k]), (heldout, items[k:])):
            bucket.extend(
                {
                    "category": item.category,
                    "defect_type": item.defect_type,
                    "image_path": str(item.image_path),
                    "mask_path": str(item.mask_path) if item.mask_path else None,
                }
                for item in subset
            )

    return {
        "protocol": "stratified_real_defect_fewshot_support",
        "seed": seed,
        "n_support_requested_per_defect_type": n_support,
        "support": support,
        "heldout": heldout,
    }

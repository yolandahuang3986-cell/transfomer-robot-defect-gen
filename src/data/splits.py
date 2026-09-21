import random
from collections import defaultdict


def build_fewshot_manifest(records, n_support, seed):
    """Build a leakage-safe defect support/held-out split.

    Only real defective images are eligible for support/held-out splitting.
    Sampling is stratified by category and defect type.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)

    for record in records:
        if record.split == "test" and record.defect_type != "good":
            groups[(record.category, record.defect_type)].append(record)

    support = []
    heldout = []

    for (category, defect_type), items in sorted(groups.items()):
        items = list(items)
        rng.shuffle(items)
        k = min(n_support, len(items))

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

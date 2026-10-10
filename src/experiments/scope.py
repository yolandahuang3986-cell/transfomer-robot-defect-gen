"""Pre-registered defect strata included in the core comparison."""
import json
from pathlib import Path


_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "experiment_scope.json"
_PROTOCOL = json.loads(_CONFIG.read_text())
PROTOCOL_VERSION = _PROTOCOL["protocol_version"]
EXCLUDED_STRATA = {
    (row["category"], row["defect_type"]): row["reason"]
    for row in _PROTOCOL["excluded_strata"]
}


def stratum_exclusion(category, defect_type):
    """Return the frozen exclusion reason, or None when the stratum is in scope."""
    return EXCLUDED_STRATA.get((category, defect_type))


def require_in_scope(category, defect_type):
    reason = stratum_exclusion(category, defect_type)
    if reason is not None:
        raise ValueError(
            f"Out-of-scope defect stratum {category}/{defect_type} "
            f"under {PROTOCOL_VERSION}: {reason}"
        )


def filter_in_scope(rows):
    """Filter manifest records by the pre-registered category/type scope."""
    return [
        row for row in rows
        if stratum_exclusion(row["category"], row["defect_type"]) is None
    ]


def excluded_rows(rows):
    """Return rows omitted by the frozen scope, for result provenance."""
    return [
        row for row in rows
        if stratum_exclusion(row["category"], row["defect_type"]) is not None
    ]

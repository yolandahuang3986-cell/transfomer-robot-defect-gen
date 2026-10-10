import pytest

from src.experiments.scope import PROTOCOL_VERSION, filter_in_scope, require_in_scope


def test_metal_nut_flip_is_excluded_from_shared_core_scope():
    rows = [
        {"category": "metal_nut", "defect_type": "flip", "id": 1},
        {"category": "metal_nut", "defect_type": "color", "id": 2},
        {"category": "screw", "defect_type": "thread_side", "id": 3},
    ]
    assert [row["id"] for row in filter_in_scope(rows)] == [2, 3]
    with pytest.raises(ValueError, match=PROTOCOL_VERSION):
        require_in_scope("metal_nut", "flip")


def test_color_remains_in_scope_pending_support_mask_audit():
    require_in_scope("metal_nut", "color")

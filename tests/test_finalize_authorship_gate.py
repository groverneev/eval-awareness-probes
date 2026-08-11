import copy

import pytest

from vnaa.finalize_authorship_gate import validate_cells


def complete_cells():
    cells = []
    for condition in (
        "target_on_policy", "same_family_off_policy", "cross_family_off_policy"
    ):
        prefix = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": condition},
        ]
        for status, final in (("eval", "e"), ("deploy", "d")):
            cells.append({
                "item_id": "i1", "history_authorship": condition,
                "status": status,
                "conversation": copy.deepcopy(prefix)
                + [{"role": "user", "content": final}],
                "terminated_naturally": True,
            })
    return cells


def test_validate_cells_accepts_complete_factorial():
    assert validate_cells(complete_cells()) == ["i1"]


def test_validate_cells_rejects_prefix_drift():
    cells = complete_cells()
    cells[1]["conversation"][1]["content"] = "drift"
    with pytest.raises(ValueError, match="prefixes differ"):
        validate_cells(cells)


def test_validate_cells_rejects_nontermination():
    cells = complete_cells()
    cells[0]["terminated_naturally"] = False
    with pytest.raises(ValueError, match="did not terminate"):
        validate_cells(cells)

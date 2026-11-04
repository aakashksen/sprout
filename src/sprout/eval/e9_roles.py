"""Which model E9 scores as primary and which beside it. E9 chooses them and check_run chooses them again from the
committed decision, with this one function, so a record cannot name another pairing.

The decision (`decision_file` of configs/e9.json, written by T4.9) names the chosen checkpoint and its sha256. If the
chosen checkpoint is one of the config's candidates, that candidate is the primary. Otherwise it is a new model, called
`e15`, and the decision must name its calibration file. A decision that names `calibration` also replaces the chosen
candidate's own calibration file, because T4.9 refits E5 on the chosen model.
"""

from __future__ import annotations

from typing import Any

NEW_MODEL = "e15"


def candidate_specs(candidates: dict[str, Any], decision: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The candidates by key: the config's, then the decision's own model when it is none of them."""
    specs = {k: dict(v) for k, v in candidates.items()}
    chosen = [k for k, v in specs.items() if v.get("checkpoint") == decision.get("chosen_model")]
    if not chosen:
        if not decision.get("calibration"):
            raise ValueError(f"the decision chose {decision.get('chosen_model')!r}, which is neither of the config's "
                             f"candidates {[v.get('checkpoint') for v in specs.values()]} and names no calibration file")
        specs[NEW_MODEL] = {"checkpoint": decision["chosen_model"], "calibration": decision["calibration"]}
    elif decision.get("calibration"):
        specs[chosen[0]]["calibration"] = decision["calibration"]
    return specs


def roles(candidates: dict[str, Any], decision: dict[str, Any]) -> tuple[str, list[str]]:
    """(primary key, alternate keys in the config's order)."""
    specs = candidate_specs(candidates, decision)
    primary = next(k for k, v in specs.items() if v["checkpoint"] == decision["chosen_model"])
    return primary, [k for k in specs if k != primary]


def system_of(index: int) -> str:
    """The system name of the alternate at this position."""
    return "alternate" if index == 0 else f"alternate_{index + 1}"

"""Which model E9 scores as primary and which beside it. E9 chooses them and check_run chooses them again from the
committed decision, with this one function, so a record cannot name another pairing.

The decision (`decision_file` of configs/e9.json, written by T4.9) names the chosen checkpoint and its sha256; the chosen
model is the primary and it must be one of the config's candidates (E4, E12 and T4.9's own model), whose calibration is
the refit T4.9 made on the chosen model. Every other candidate is an alternate, in the config's order.
"""

from __future__ import annotations

from typing import Any

DECISION_KEYS = ("chosen_checkpoint", "chosen_checkpoint_sha256", "classes")


def normalized(raw: dict[str, Any], e15_config: dict[str, Any]) -> dict[str, Any]:
    """T4.9's results/e15/decision.json (`chosen`, `chosen_checkpoint`, `chosen_checkpoint_sha256`, `classes`) in the names
    the rest of E9 uses. The calibration is the refit T4.9 defines in configs/e15.json: `<refit.out_dir>/chosen.json`."""
    missing = [k for k in DECISION_KEYS if k not in raw]
    if missing:
        raise ValueError(f"the decision lacks {missing}")
    try:
        calibration = f"{e15_config['refit']['out_dir']}/chosen.json"
    except (KeyError, TypeError) as exc:
        raise ValueError("configs/e15.json has no refit.out_dir, so the refit calibration cannot be located") from exc
    return {"chosen": raw.get("chosen"), "chosen_model": raw["chosen_checkpoint"],
            "chosen_model_sha256": raw["chosen_checkpoint_sha256"], "class_names": raw["classes"],
            "calibration": calibration}


def candidate_specs(candidates: dict[str, Any], decision: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The candidates by key, the chosen one with the refit calibration the decision points to."""
    specs = {k: dict(v) for k, v in candidates.items()}
    chosen = [k for k, v in specs.items() if v.get("checkpoint") == decision.get("chosen_model")]
    if not chosen:
        raise ValueError(f"the decision chose {decision.get('chosen_model')!r}, which is none of the config's candidates "
                         f"{[v.get('checkpoint') for v in specs.values()]}")
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

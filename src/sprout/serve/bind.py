"""Gives E5's calibration file the checksum of the checkpoint it belongs to, so the Identifier can check the pairing.

A refit (E12's model) writes `model_sha256` itself, and this only confirms it. E4's `results/e5/chosen.json` has none,
because E5 did not record one when it was fitted. For that file the checkpoint cannot be proven by its bytes, so
the binding rests on three things that can be checked: the checkpoint is the path `configs/e5.json` says E5 used, a
committed E5 record names the same E4 record and holds the same temperature and threshold as the file, and the file's
own numbers are copied through unchanged. The output says which of the two it was (`bound_by`).

    python -m sprout.serve.bind --checkpoint models/e4/convnext_joint/best.pt --out models/serve/chosen.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from sprout.atomic import atomic_write_text

ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_calibration(checkpoint: Path, root: Path, chosen: str = "results/e5/chosen.json",
                      e5_config: str = "configs/e5.json") -> dict:
    """The calibration file's contents with the checkpoint's name and hash added, or ValueError if the pairing cannot
    be shown. Writes nothing."""
    checkpoint = Path(checkpoint).resolve()
    sha = _sha256(checkpoint)
    fitted = json.loads((root / chosen).read_text(encoding="utf-8"))
    rel = checkpoint.relative_to(root.resolve()).as_posix() if checkpoint.is_relative_to(root.resolve()) else str(checkpoint)
    if fitted.get("model_sha256"):
        if fitted["model_sha256"] != sha:
            raise ValueError(f"{chosen} was fitted for checkpoint {fitted['model_sha256']}, not {sha} ({checkpoint}): "
                             "that is another model")
        bound_by = "refit: the file names this checkpoint's sha256"
    else:
        model = json.loads((root / e5_config).read_text(encoding="utf-8"))["model"]
        if (root / model["checkpoint"]).resolve() != checkpoint:
            raise ValueError(f"{checkpoint} is not the checkpoint E5 used ({model['checkpoint']} in {e5_config})")
        shown = [
            p for p in sorted((root / "results/runs").glob("E5-*.json"))
            if (c := json.loads(p.read_text(encoding="utf-8")).get("config", {}))
            and c.get("model_record") == model["e4_record"]
            and c.get("temperature") == fitted["temperature"] and c.get("abstain_threshold") == fitted["abstain_threshold"]
        ]
        if not shown:
            raise ValueError(f"no E5 record of E4 record {model['e4_record']} shows the temperature and threshold in "
                             f"{chosen}, so the file cannot be tied to that model")
        bound_by = (f"path: {rel} is the checkpoint configs/e5.json names, and {shown[0].name} shows these numbers "
                    f"for E4 record {model['e4_record']}; E5 stored no checksum when it was fitted")
    return {**fitted, "model_checkpoint": rel, "model_sha256": sha, "bound_by": bound_by}


def bind_calibration(checkpoint: Path, root: Path, out: Path, chosen: str = "results/e5/chosen.json",
                     e5_config: str = "configs/e5.json") -> Path:
    bound = bound_calibration(checkpoint, root, chosen, e5_config)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(bound, indent=1) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="the checkpoint the served model file was exported from")
    ap.add_argument("--out", default="models/serve/chosen.json")
    ap.add_argument("--chosen", default="results/e5/chosen.json")
    a = ap.parse_args(argv)
    try:
        print(bind_calibration(ROOT / a.checkpoint, ROOT, ROOT / a.out, a.chosen))
    except (ValueError, OSError, KeyError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

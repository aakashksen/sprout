"""List the extra photos to remove when PlantDoc is held out (gate G1 of docs/DECISIONS.md, configs/e12_gates.json).

Run from the repo root, with -I so a planted file next to the photos cannot load:
    py -3.12 -I tools/extra_trim.py [--device cpu|cuda] [--workers 4]
Embeds PlantDoc's field_dev photos as scored (cleaning applied), its field_train photos, and all extras with their mirrors
(DINOv2 ViT-S/14 class token), then writes
    results/e12/plantdoc_extra_trim.json              the counts and the extras at cosine 0.75 or above
    datasets/manifests/extra_train_trim_plantdoc.csv  extra_train.csv without those rows, which the trimmed run trains on
Embeddings are cached in models/e12/trim_cache (written atomically), so a stopped run continues. field_test is never read.
Uses the GPU when one is visible, else the CPU (about 4,000 photos twice; run it when the GPU is in a window).
Commit both files after checking the printed counts against the verifier's.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import sprout.data.splits as splits  # noqa: E402
from sprout.atomic import atomic_write_text  # noqa: E402
from sprout.baselines.dino import load_dino_encoder  # noqa: E402
from sprout.data import near_extras as ne  # noqa: E402
from sprout.experiments import e12_more_sources as e12  # noqa: E402

GATES = "configs/e12_gates.json"


def write_outputs(result: dict, cfg: dict, root: Path) -> None:
    trim = cfg["gates"]["g1"]["trim"]
    extra = splits.load_manifest(cfg["extra"]["manifest"])
    kept = ne.trimmed_manifest(extra, result["remove"])
    out = root / "datasets" / "manifests" / f"{trim['manifest']}.csv"
    atomic_write_text(out, kept.to_csv(index=False))
    (root / trim["list"]).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(root / trim["list"], ne.dumps(result))


def make(cfg: dict, root: Path, model, device, workers: int) -> dict:
    trim = cfg["gates"]["g1"]["trim"]
    e12.check_extra_frozen(cfg)
    spec = e12.run_spec(cfg, cfg["gates"]["g1"]["run"])
    _, scored, _, _ = e12.dev_rows(cfg, spec, root)
    field = pd.read_csv(root / "datasets" / "manifests" / "field.csv")
    train = ne.source_rows(field, spec["held_out_source"], "field_train")
    result = ne.build(spec["held_out_source"], scored, train, splits.load_manifest(cfg["extra"]["manifest"]), model, device,
                      root, threshold=trim["threshold"], workers=workers, cache=root / trim["cache"])
    expected = trim["verifier"]
    got = {k: result[k] for k in expected}
    result["verifier_counts"] = expected
    result["reproduces_verifier"] = got == expected
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    cfg = e12.load_config(ROOT / GATES)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    result = make(cfg, ROOT, load_dino_encoder(), device, a.workers)
    write_outputs(result, cfg, ROOT)
    print(json.dumps({k: v for k, v in result.items() if k != "remove"}, indent=1))
    return 0 if result["reproduces_verifier"] else 2


if __name__ == "__main__":
    sys.exit(main())

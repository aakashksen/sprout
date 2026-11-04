import hashlib
import json

import pytest

from sprout.serve import bind

CHECKPOINT = b"weights of the model E5 was fitted on"
SHA = hashlib.sha256(CHECKPOINT).hexdigest()
FITTED = {"temperature": 1.19, "abstain_threshold": 0.885377, "gate": {"chosen": "green_rule", "settings": {}}}


def _world(tmp_path, chosen=None, record=None, config=None):
    """A results/e5/chosen.json, its E5 record, configs/e5.json and the checkpoint it names, in a temp root."""
    (tmp_path / "results/e5").mkdir(parents=True)
    (tmp_path / "results/runs").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "models/e4").mkdir(parents=True)
    (tmp_path / "models/e4/best.pt").write_bytes(CHECKPOINT)
    (tmp_path / "results/e5/chosen.json").write_text(json.dumps(chosen or FITTED))
    (tmp_path / "configs/e5.json").write_text(json.dumps(
        config or {"model": {"checkpoint": "models/e4/best.pt", "e4_record": "E4-1"}}))
    rec = {"experiment": "E5", "config": {"model_record": "E4-1", "temperature": 1.19, "abstain_threshold": 0.885377}}
    (tmp_path / "results/runs/E5-1.json").write_text(json.dumps(record or rec))
    return tmp_path


def _bind(root, checkpoint="models/e4/best.pt"):
    return bind.bind_calibration(root / checkpoint, root, root / "models/serve/chosen.json")


def test_an_e4_calibration_gets_the_checkpoint_hash_and_keeps_its_numbers(tmp_path):
    root = _world(tmp_path)
    out = _bind(root)
    bound = json.loads(out.read_text())
    assert bound["model_sha256"] == SHA and bound["model_checkpoint"] == "models/e4/best.pt"
    assert {k: v for k, v in bound.items() if k in FITTED} == FITTED  # E5's numbers are untouched
    assert "model_sha256" not in json.loads((root / "results/e5/chosen.json").read_text())  # and so is its file
    assert "path" in bound["bound_by"]


def test_a_refit_calibration_is_checked_against_the_checkpoint_it_names(tmp_path):
    root = _world(tmp_path, chosen={**FITTED, "model_sha256": SHA, "model_checkpoint": "models/e4/best.pt"})
    assert json.loads(_bind(root).read_text())["model_sha256"] == SHA
    root2 = _world(tmp_path / "other", chosen={**FITTED, "model_sha256": "0" * 64})
    with pytest.raises(ValueError, match="another model"):
        _bind(root2)


def test_refuses_a_checkpoint_other_than_the_one_e5_used(tmp_path):
    root = _world(tmp_path)
    (root / "models/e4/other.pt").write_bytes(b"different weights")
    with pytest.raises(ValueError, match="not the checkpoint"):
        _bind(root, "models/e4/other.pt")
    assert not (root / "models/serve/chosen.json").exists()


def test_refuses_numbers_that_no_e5_record_shows(tmp_path):
    root = _world(tmp_path, chosen={**FITTED, "temperature": 2.0})
    with pytest.raises(ValueError, match="no E5 record"):
        _bind(root)


def test_refuses_a_record_of_another_e4_model(tmp_path):
    rec = {"experiment": "E5", "config": {"model_record": "E4-2", "temperature": 1.19, "abstain_threshold": 0.885377}}
    with pytest.raises(ValueError, match="no E5 record"):
        _bind(_world(tmp_path, record=rec))

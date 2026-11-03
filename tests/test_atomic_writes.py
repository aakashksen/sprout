import os
import subprocess
import threading
import time

import numpy as np
import pytest
import torch

import sprout.atomic as atomic
from sprout.atomic import atomic_write_text, temp_path
from sprout.eval.check_run import check
from sprout.eval.predict import Predictions
from sprout.eval.records import split_info, write_run
from sprout.train.checkpoint import load_checkpoint, save_checkpoint
from sprout.train.config import TrainConfig


def sh(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def root(tmp_path):
    lab = ["path,label,group,split,sha1"] + [f"i/{i}.jpg,{'AB'[i % 2]},g{i},lab_test,s{i}" for i in range(4)]
    for rel, text in {"datasets/manifests/lab.csv": lab, "orchestration/tasks.json": ["{}"],
                      ".gitignore": ["*.tmp"]}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("\n".join(text) + "\n")
    sh(tmp_path, "init", "-q")
    sh(tmp_path, "config", "user.email", "t@t")
    sh(tmp_path, "config", "user.name", "t")
    sh(tmp_path, "add", "-A")
    sh(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def run(root):
    info = split_info("lab_test", root)
    labels = np.array([0, 1, 0, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    return write_run(
        "E1", "lab_test", {"class_names": ["A", "B"]}, {"classifier": "resnet-test"},
        {"a": {"accuracy": 1.0}}, {}, [{"setting": "default", "result": "ok"}],
        {"a": Predictions(list(info.ids), labels, probs)}, info.hash, root=root,
        measurements={"latency_ms": [1.0, 2.0]},
    )


def test_failure_while_writing_the_record_leaves_no_record(root, monkeypatch):
    real = os.replace

    def replace(src, dst):
        if str(dst).endswith(".json") and ".measurements." not in str(dst):
            raise OSError("power cut")
        real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", replace)
    with pytest.raises(OSError, match="power cut"):
        run(root)
    runs = root / "results" / "runs"
    assert not [p for p in runs.glob("E1-*.json") if ".measurements." not in p.name]
    assert not list(runs.glob("*.tmp"))
    assert list(runs.glob("*.predictions.jsonl")) and list(runs.glob("*.measurements.json"))


def test_stale_temp_files_do_not_break_check_run_or_dirty_the_tree(root):
    path = run(root)
    for target in (path, path.with_name(path.stem + ".a.predictions.jsonl")):
        temp_path(target).write_text("{ truncated")
    stray = temp_path(path.with_name("E1-20990101T000000000000Z.json"))
    stray.write_text("{")
    assert "." in stray.name and not stray.name.endswith(".json")
    assert check("E1", root=root, min_n=0) == []
    second = run(root)
    assert second != path
    out = subprocess.run(["git", "status", "--porcelain", "-uall"], cwd=root, capture_output=True, text=True)
    assert ".tmp" not in out.stdout


def checkpoint(path, marker):
    class Obj:
        def state_dict(self):
            return {"marker": torch.tensor(marker)}

    save_checkpoint(path, model=Obj(), optimizer=Obj(), scheduler=Obj(), scaler=Obj(), epoch=marker,
                    best_metric=0.0, class_map={"a": 0}, config=TrainConfig(manifest="m", out_dir="o"))


def test_failed_checkpoint_write_keeps_the_old_latest(tmp_path, monkeypatch):
    latest = tmp_path / "latest.pt"
    checkpoint(latest, 1)

    def broken_save(obj, f):
        f.write(b"partial")
        raise OSError("disk gone")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk gone"):
        checkpoint(latest, 2)
    assert load_checkpoint(latest)["epoch"] == 1
    assert not temp_path(latest).exists()


def test_replace_waits_for_a_reader_that_lets_go(tmp_path):
    latest = tmp_path / "latest.pt"
    atomic_write_text(latest, "old")
    reader = open(latest)
    threading.Timer(0.3, reader.close).start()
    atomic_write_text(latest, "new")
    assert latest.read_text() == "new"


def test_replace_gives_a_clear_error_when_the_reader_never_lets_go(tmp_path):
    latest = tmp_path / "latest.pt"
    atomic_write_text(latest, "old")
    with open(latest) as reader:
        start = time.time()
        with pytest.raises(OSError, match="latest.pt"):
            atomic_write_text(latest, "new")
        assert time.time() - start < 10
        if os.name == "nt":
            assert reader.read() == "old"
    assert latest.read_text() == "old"
    assert not temp_path(latest).exists()

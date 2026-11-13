import dataclasses
import pickle
import random

import pandas as pd
import pytest
import torch
from PIL import Image

import sprout.data.splits as splits
from sprout.data.splits import dataset_hash
from sprout.preprocess import preprocess
from sprout.train.checkpoint import load_checkpoint, save_checkpoint
from sprout.train.config import TrainConfig
from sprout.train.dataset import (
    build_datasets,
    build_loaders,
    make_class_map,
    train_transform,
    val_transform,
)
from sprout.train.loop import build_model, train


def _noise_image(seed: int) -> Image.Image:
    g = torch.Generator().manual_seed(seed)
    data = (torch.rand(3, 64, 64, generator=g) * 255).to(torch.uint8)
    return Image.fromarray(data.permute(1, 2, 0).numpy())


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    rows = []
    for i in range(12):
        label = "red" if i % 2 == 0 else "blue"
        color = (200, 30, 30) if label == "red" else (30, 30, 200)
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (64, 64), color).save(p)
        split = "lab_train" if i < 8 else "lab_dev"
        rows.append((str(p), label, f"g{i}", split, f"sha{i}"))
    df = pd.DataFrame(rows, columns=["path", "label", "group", "split", "sha1"])
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    df.to_csv(mdir / "tiny.csv", index=False)
    monkeypatch.setattr(splits, "MANIFEST_DIR", mdir)
    return df


def _config(tmp_path, **kw) -> TrainConfig:
    base = dict(
        manifest="tiny",
        out_dir=str(tmp_path / "run"),
        model="tiny_cnn",
        seed=1,
        lr=0.05,
        batch_size=4,
        epochs=2,
    )
    base.update(kw)
    return TrainConfig(**base)


def test_train_transform_differs_from_val_transform():
    img = _noise_image(0)
    tfs = TrainConfig(manifest="m", out_dir="o").transforms
    random.seed(0)
    outs = [train_transform(img, tfs) for _ in range(4)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])
    assert torch.equal(val_transform(img), val_transform(img))
    assert torch.equal(val_transform(img), preprocess(img))


def test_val_dataset_has_no_random_augmentation(manifest, tmp_path):
    cfg = _config(tmp_path)
    train_ds, val_ds = build_datasets(manifest, cfg, make_class_map(manifest))
    assert len(train_ds) == 8 and len(val_ds) == 4
    assert val_ds.transform is val_transform
    a, _ = val_ds[0]
    b, _ = val_ds[0]
    assert torch.equal(a, b)


def _interrupt_after(monkeypatch, epoch):
    """Make train() stop (as a killed job would) once the checkpoint for `epoch` is on disk."""
    import sprout.train.loop as loop

    real = loop.save_checkpoint

    def save(path, **state):
        real(path, **state)
        if state["epoch"] == epoch and path.name == "latest.pt":
            raise KeyboardInterrupt

    monkeypatch.setattr(loop, "save_checkpoint", save)
    return lambda: monkeypatch.setattr(loop, "save_checkpoint", real)


def test_resume_requires_explicit_path(manifest, tmp_path, monkeypatch):
    cfg = _config(tmp_path, epochs=2)
    undo = _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    undo()
    with pytest.raises(FileExistsError):
        train(cfg)  # checkpoints exist, but resume was not asked for
    with pytest.raises(FileNotFoundError):
        train(_config(tmp_path, out_dir=str(tmp_path / "other")), resume=tmp_path / "nope.pt")
    out = train(cfg, resume=tmp_path / "run" / "latest.pt")
    assert out.epochs_done == 2 and len(out.history) == 1


@pytest.mark.parametrize("change", [dict(lr=0.001), dict(batch_size=8), dict(epochs=4), dict(seed=9), dict(transforms=())])
def test_resume_refuses_changed_training_settings(manifest, tmp_path, monkeypatch, change):
    cfg = _config(tmp_path, epochs=2)
    undo = _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    undo()
    with pytest.raises(ValueError, match="settings"):
        train(dataclasses.replace(cfg, **change), resume=tmp_path / "run" / "latest.pt")


def test_resume_restores_optimizer_and_scheduler(manifest, tmp_path, monkeypatch):
    straight = train(_config(tmp_path, out_dir=str(tmp_path / "straight"), transforms=()))
    cfg = _config(tmp_path, transforms=())
    undo = _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    undo()
    resumed = train(cfg, resume=tmp_path / "run" / "latest.pt")
    a, b = load_checkpoint(straight.latest_path), load_checkpoint(resumed.latest_path)
    assert b["scheduler"] == a["scheduler"]
    assert b["optimizer"]["param_groups"] == a["optimizer"]["param_groups"]
    assert b["optimizer"]["state"][0]["step"] == a["optimizer"]["state"][0]["step"]
    for x, y in zip(a["model"].values(), b["model"].values(), strict=False):
        assert torch.allclose(x, y, atol=1e-5)


def test_checkpoint_round_trip_includes_class_map_and_optimizer(tmp_path):
    cfg = TrainConfig(manifest="m", out_dir="o", model="tiny_cnn", manifest_hash="abc")
    model = build_model("tiny_cnn", 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model(torch.zeros(1, 3, 224, 224)).sum().backward()
    opt.step()
    p = tmp_path / "c.pt"
    save_checkpoint(
        p,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler,
        epoch=3,
        best_metric=0.5,
        class_map={"a": 0, "b": 1},
        config=cfg,
    )
    ck = load_checkpoint(p)
    assert ck["class_map"] == {"a": 0, "b": 1}
    assert ck["epoch"] == 3 and ck["best_metric"] == 0.5
    assert ck["config"] == cfg
    assert ck["optimizer"]["state"]  # AdamW moments saved
    assert "scheduler" in ck and "scaler" in ck
    m2 = build_model("tiny_cnn", 2)
    m2.load_state_dict(ck["model"])
    for a, b in zip(model.parameters(), m2.parameters(), strict=False):
        assert torch.equal(a, b)


def test_best_and_latest_are_separate(manifest, tmp_path):
    out = train(_config(tmp_path))
    assert out.best_path != out.latest_path
    assert out.best_path.is_file() and out.latest_path.is_file()
    assert load_checkpoint(out.latest_path)["epoch"] == 2
    assert load_checkpoint(out.best_path)["best_metric"] == out.best_metric
    assert load_checkpoint(out.latest_path)["config"].manifest_hash == dataset_hash(manifest)


def test_two_step_run_on_tiny_model_reduces_loss_on_cpu(manifest, tmp_path):
    out = train(_config(tmp_path, transforms=()))
    assert out.history[1]["train_loss"] < out.history[0]["train_loss"]


def test_same_seed_same_split_and_first_batch(manifest, tmp_path):
    cfg = _config(tmp_path)
    cm = make_class_map(manifest)
    a_tr, a_val = build_loaders(manifest, cfg, cm)
    xa, ya = next(iter(a_tr))
    b_tr, b_val = build_loaders(manifest, cfg, cm)
    xb, yb = next(iter(b_tr))
    assert a_tr.dataset.paths == b_tr.dataset.paths
    assert a_val.dataset.paths == b_val.dataset.paths
    assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_convnext_tiny_head_is_resized_and_trains_on_cpu():
    model = build_model("convnext_tiny", 7, pretrained=False)
    assert model.classifier[2].out_features == 7
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, 7)
    out.sum().backward()
    assert model.classifier[2].weight.grad is not None


@pytest.mark.real
def test_convnext_tiny_default_weights_download():
    model = build_model("convnext_tiny", 38, pretrained=True)
    assert model.classifier[2].out_features == 38
    assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 38)


def test_class_map_comes_from_manifest(manifest, tmp_path):
    assert make_class_map(manifest) == {"blue": 0, "red": 1}
    out = train(_config(tmp_path, epochs=1))
    assert load_checkpoint(out.latest_path)["class_map"] == {"blue": 0, "red": 1}


def test_first_batch_is_seed_stable_workers_are_configured_and_datasets_pickle(manifest, tmp_path):
    # Runs at 0 workers (spawned workers fail with WinError 6 on a busy machine). Not proved here: that
    # worker processes start, re-seed from the loader's generator, and return the same batch as 0 workers.
    cm = make_class_map(manifest)
    first = []
    for _ in range(2):
        tr, _val = build_loaders(manifest, _config(tmp_path), cm)
        first.append(next(iter(tr)))
    assert torch.equal(first[0][0], first[1][0]) and torch.equal(first[0][1], first[1][1])
    tr, val = build_loaders(manifest, _config(tmp_path, num_workers=2), cm)  # built, never iterated
    assert tr.num_workers == val.num_workers == 2 and tr.persistent_workers and val.persistent_workers
    for loader in (tr, val):  # what a spawned worker receives must survive pickling
        copy = pickle.loads(pickle.dumps(loader.dataset))
        assert copy.paths == loader.dataset.paths and copy.targets == loader.dataset.targets
    assert torch.equal(pickle.loads(pickle.dumps(val.dataset))[0][0], val.dataset[0][0])


def test_a_run_records_seconds_and_memory_columns(manifest, tmp_path):
    out = train(_config(tmp_path))
    assert out.epochs_done == 2
    assert all(row["seconds"] > 0 for row in out.history)
    assert {"peak_mem_allocated", "peak_mem_reserved"} <= set(out.history[0])


def test_worker_count_does_not_block_a_resume(manifest, tmp_path, monkeypatch):
    # The settings check is what is tested; the loaders are forced to 0 workers so no process is spawned.
    # Not proved here: that a resumed run can iterate with real workers.
    import sprout.train.loop as loop

    cfg = _config(tmp_path, epochs=2)
    undo = _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    undo()
    real = loop.build_loaders
    monkeypatch.setattr(
        loop, "build_loaders", lambda m, c, cm: real(m, dataclasses.replace(c, num_workers=0), cm)
    )
    out = train(dataclasses.replace(cfg, num_workers=2), resume=tmp_path / "run" / "latest.pt")
    assert out.epochs_done == 2
    assert load_checkpoint(out.latest_path)["config"].num_workers == 2


def test_nonfinite_train_loss_stops_the_run_without_checkpoints(manifest, tmp_path, monkeypatch):
    import sprout.train.loop as loop

    def broken(name, num_classes, pretrained=True):
        model = build_model(name, num_classes, pretrained=False)
        with torch.no_grad():
            model[-1].weight.fill_(float("nan"))
        return model

    monkeypatch.setattr(loop, "build_model", broken)
    cfg = _config(tmp_path)
    with pytest.raises(RuntimeError, match=r"epoch 1.*train loss.*not finite"):
        train(cfg)
    assert not (tmp_path / "run" / "best.pt").exists() and not (tmp_path / "run" / "latest.pt").exists()


def test_nonfinite_validation_loss_stops_before_best_is_saved(manifest, tmp_path, monkeypatch):
    import sprout.train.loop as loop

    calls = []

    def evaluate(model, loader, loss_fn, device):
        calls.append(1)
        return (float("nan"), 0.9) if len(calls) == 2 else (0.5, 0.5)

    monkeypatch.setattr(loop, "_evaluate", evaluate)
    cfg = _config(tmp_path)
    with pytest.raises(RuntimeError, match=r"epoch 2.*validation loss.*not finite"):
        train(cfg)
    assert load_checkpoint(tmp_path / "run" / "best.pt")["epoch"] == 1
    assert load_checkpoint(tmp_path / "run" / "latest.pt")["epoch"] == 1


def test_stale_temp_checkpoint_does_not_break_train_or_resume(manifest, tmp_path, monkeypatch):
    from sprout.atomic import temp_path

    cfg = _config(tmp_path, epochs=2)
    out = tmp_path / "run"
    out.mkdir()
    for name in ("latest.pt", "best.pt"):
        temp_path(out / name).write_bytes(b"half a checkpoint")
    undo = _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)  # leftover temp files do not count as checkpoints
    undo()
    assert train(cfg, resume=out / "latest.pt").epochs_done == 2


# Real worker processes die with WinError 6 on this machine when it is busy, so these run only on request
# (pytest -m real) and are not retried.
class _PidProbe(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        import os

        return os.getpid()


@pytest.mark.real
def test_real_workers_are_separate_processes():
    import os

    pids = {int(p) for batch in torch.utils.data.DataLoader(_PidProbe(), batch_size=1, num_workers=2) for p in batch}
    assert pids and os.getpid() not in pids


@pytest.mark.real
def test_real_workers_reseed_from_the_loader_generator(manifest, tmp_path):
    cm = make_class_map(manifest)

    def first(seed):
        tr, _ = build_loaders(manifest, _config(tmp_path, num_workers=2, seed=seed), cm)
        it = iter(tr)
        batch = next(it)[0]
        it._shutdown_workers()  # leave no worker holding handles when the next loader starts
        return batch

    assert torch.equal(first(1), first(1))  # random augmentation repeats because workers are seeded from gen
    assert not torch.equal(first(1), first(2))


@pytest.mark.real
def test_real_workers_give_the_zero_worker_first_batch_without_augmentation(manifest, tmp_path):
    cm = make_class_map(manifest)
    batches = []
    for workers in (0, 2):
        tr, _ = build_loaders(manifest, _config(tmp_path, num_workers=workers, transforms=()), cm)
        batches.append(next(iter(tr)))
    assert torch.equal(batches[0][0], batches[1][0]) and torch.equal(batches[0][1], batches[1][1])


@pytest.mark.real
def test_a_resumed_run_iterates_with_real_workers(manifest, tmp_path, monkeypatch):
    cfg = _config(tmp_path, epochs=2)
    undo = _interrupt_after(monkeypatch, 1)
    with pytest.raises(KeyboardInterrupt):
        train(cfg)
    undo()
    out = train(dataclasses.replace(cfg, num_workers=2), resume=tmp_path / "run" / "latest.pt")
    assert out.epochs_done == 2

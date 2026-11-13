import numpy as np


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1
    return cm


def per_class_report(y_true, y_pred, class_names: list[str]) -> dict[str, dict]:
    """Precision, recall, F1 and support for each class that appears in y_true."""
    cm = confusion(y_true, y_pred, len(class_names))
    report = {}
    for i, name in enumerate(class_names):
        support = int(cm[i].sum())
        if support == 0:
            continue
        tp = cm[i, i]
        predicted = cm[:, i].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
    return report


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean F1 over classes present in y_true."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    num = int(max(y_true.max(), y_pred.max())) + 1
    report = per_class_report(y_true, y_pred, [str(i) for i in range(num)])
    return float(np.mean([r["f1"] for r in report.values()]))


def ece(probs, labels, bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def selective_metrics(probs, labels, threshold: float, class_names=None, healthy_threshold=None) -> dict[str, float]:
    """Answer when top confidence >= threshold; with `class_names` and a healthy threshold, the rule serving applies
    (a healthy top label must also reach the healthy bar). Rates are nan when nothing is answered."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    if healthy_threshold is not None:
        from sprout.serve.calibrate import answered_mask

        answered = answered_mask(probs, class_names, threshold, healthy_threshold)
    else:
        answered = probs.max(axis=1) >= threshold
    if not answered.any():
        return {
            "coverage": 0.0,
            "selective_accuracy": float("nan"),
            "confident_error_rate": float("nan"),
        }
    acc = float((probs[answered].argmax(axis=1) == labels[answered]).mean())
    return {
        "coverage": float(answered.mean()),
        "selective_accuracy": acc,
        "confident_error_rate": 1.0 - acc,
    }


def recall_at_k(ranks, k: int) -> float:
    """`ranks` holds the 1-based rank of the gold item per query, or None if not retrieved."""
    return float(np.mean([r is not None and r <= k for r in ranks]))


def mrr(ranks) -> float:
    return float(np.mean([0.0 if r is None else 1.0 / r for r in ranks]))


def top_k_accuracy(probs, labels, k: int) -> float:
    """Share of rows whose true class is among the k highest probabilities. Ties go to the lower index,
    as in argmax, so k=1 equals accuracy exactly."""
    if k < 1:
        raise ValueError("k must be at least 1")
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels, dtype=int)
    true_p = probs[np.arange(len(labels)), labels][:, None]
    before = (probs > true_p).sum(axis=1) + ((probs == true_p) & (np.arange(probs.shape[1]) < labels[:, None])).sum(axis=1)
    return float((before < k).mean())


def plant_of(class_name: str) -> str | None:
    """The plant is the part of the class name before `___`; None when the name has no plant part."""
    return class_name.split("___", 1)[0] if "___" in class_name else None


def options_per_plant(class_names: list[str]) -> dict[str, int]:
    """How many classes each plant has among the model's outputs."""
    counts: dict[str, int] = {}
    for name in class_names:
        plant = plant_of(name)
        if plant is not None:
            counts[plant] = counts.get(plant, 0) + 1
    return counts


def plant_given_top_k(probs, labels, class_names: list[str], k: int) -> float:
    """Top-k accuracy when each row's outputs are restricted to the classes of its true plant.
    nan (undefined) when any true label has no plant, as for a non-plant image."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels, dtype=int)
    plants = [plant_of(n) for n in class_names]
    if probs.shape[1] != len(class_names) or any(plants[y] is None for y in labels):
        return float("nan")
    column_plant = np.array([p if p is not None else "" for p in plants])
    keep = column_plant[None, :] == column_plant[labels][:, None]
    return top_k_accuracy(np.where(keep, probs, -np.inf), labels, k)

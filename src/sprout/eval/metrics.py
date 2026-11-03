import numpy as np


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
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


def selective_metrics(probs, labels, threshold: float) -> dict[str, float]:
    """Answer when top confidence >= threshold. Rates are nan when nothing is answered."""
    probs, labels = np.asarray(probs), np.asarray(labels)
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

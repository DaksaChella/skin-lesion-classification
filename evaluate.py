"""
Regenerate the full metric set (balanced acc, macro F1, macro AUC,
confusion matrix) from saved best checkpoints — no retraining.

Looks for checkpoints at results/{S1,S2,S3}_best.pth and the training
history embedded in results/{S1,S2,S3}_metrics.json (if present), then
overwrites the metrics JSON with the expanded payload.

Run from project root: `python evaluate.py`
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # save figures without opening a blocking GUI window

import torch

from data.preprocessing import get_data_loaders
from evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    print_classification_report,
    print_metrics,
)
from evaluation.plots import plot_strategy_comparison, plot_training_curves
from models.train import STRATEGY_BUILDERS, evaluate_on_test


def evaluate_strategy(strategy_name, test_loader, device):
    ckpt_path = f"results/{strategy_name}_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"[skip] {ckpt_path} not found — train {strategy_name} first.")
        return None

    model = STRATEGY_BUILDERS[strategy_name]()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)

    y_true, y_pred, y_prob = evaluate_on_test(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print_metrics(metrics, strategy_name)
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, strategy_name)

    metrics_path = f"results/{strategy_name}_metrics.json"
    history = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                history = json.load(f).get("history")
        except (json.JSONDecodeError, OSError):
            history = None

    if history and all(k in history for k in ("train_loss", "val_loss", "train_acc", "val_acc")):
        plot_training_curves(history, strategy_name)

    payload = {**metrics, "history": history} if history else dict(metrics)
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"Metrics written to {metrics_path}")

    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, _ = get_data_loaders()

    summary = {}
    for name in ("S1", "S2", "S3"):
        metrics = evaluate_strategy(name, test_loader, device)
        if metrics is not None:
            summary[name] = metrics

    if summary:
        plot_strategy_comparison(summary)
        write_combined_metrics()


def write_combined_metrics(strategies=("S1", "S2", "S3"), out_path="results/all_metrics.json"):
    """Aggregate per-strategy metrics JSONs into one file."""
    combined = {}
    for name in strategies:
        path = f"results/{name}_metrics.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            combined[name] = json.load(f)
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=4)
    print(f"Combined metrics written to {out_path}")


if __name__ == "__main__":
    main()

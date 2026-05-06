import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.preprocessing import get_data_loaders
from evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    print_classification_report,
    print_metrics,
)
from evaluation.plots import plot_training_curves
from models.model import (
    build_s1_from_scratch,
    build_s2_full_freeze,
    build_s3_gradual_unfreeze,
    unfreeze_block,
)


STRATEGY_BUILDERS = {
    "S1": build_s1_from_scratch,
    "S2": build_s2_full_freeze,
    "S3": build_s3_gradual_unfreeze,
}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


def train(
    model,
    train_loader,
    val_loader,
    strategy_name,
    class_weights=None,
    num_epochs=5,
    unfreeze_every=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    weight_tensor = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
    )

    print(f"\nTraining {strategy_name} for {num_epochs} epochs...")
    print("=" * 60)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0
    os.makedirs("results", exist_ok=True)

    for epoch in range(num_epochs):
        # S3 — unfreeze one EfficientNet feature block every `unfreeze_every` epochs.
        if strategy_name == "S3" and epoch > 0 and epoch % unfreeze_every == 0:
            block_index = epoch // unfreeze_every
            total_blocks = len(list(model.features.children()))
            if block_index < total_blocks:
                unfreeze_block(model, block_index)
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=1e-4,
                )

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"results/{strategy_name}_best.pth")

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    print(f"\nBest validation accuracy for {strategy_name}: {best_val_acc:.4f}")
    return model, history


def evaluate_on_test(model, test_loader, device):
    model.eval()
    y_true, y_pred, y_prob_chunks = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob_chunks.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.concatenate(y_prob_chunks, axis=0)
    return y_true, y_pred, y_prob


def run_strategy(strategy_name, num_epochs=5, batch_size=32, unfreeze_every=1):
    """
    End-to-end pipeline for one freezing strategy:
    load data → build model → train → evaluate on test → save plots/metrics.
    Returns a dict with the trained model, training history, and test metrics.
    """
    if strategy_name not in STRATEGY_BUILDERS:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(STRATEGY_BUILDERS)}.")

    train_loader, val_loader, test_loader, class_weights = get_data_loaders(batch_size=batch_size)

    builder = STRATEGY_BUILDERS[strategy_name]
    model = builder()

    model, history = train(
        model,
        train_loader,
        val_loader,
        strategy_name=strategy_name,
        class_weights=class_weights,
        num_epochs=num_epochs,
        unfreeze_every=unfreeze_every,
    )

    device = next(model.parameters()).device
    y_true, y_pred, y_prob = evaluate_on_test(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, strategy_name)
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, strategy_name)
    plot_training_curves(history, strategy_name)

    os.makedirs("results", exist_ok=True)
    with open(f"results/{strategy_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return {"model": model, "history": history, "metrics": metrics}

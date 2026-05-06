# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Skin lesion classification on the HAM10000 dataset (7 classes: mel, nv, bcc, akiec, bkl, df, vasc). The work is structured as a comparison of three transfer-learning strategies on two backbones:

- **S1 — From scratch**: random-init weights, all layers trainable
- **S2 — Full freeze**: ImageNet-pretrained, only the final classification head trained
- **S3 — Gradual unfreeze**: ImageNet-pretrained, starts frozen; blocks unfrozen progressively during training

Backbones: EfficientNet-B0 (`models/model.py`) and ResNet18 (`models/resnet_model.py`). Both expose `build_<backbone>_s{1,2,3}_*` builders and an `unfreeze_*` helper used during S3 training.

## Data flow

1. Raw HAM10000 (metadata CSV + two image folders) is expected at `HAM10000/` at the project root — gitignored.
2. `data/export_splits.py` reads the metadata, joins with image paths, stratified-splits 80/10/10, computes balanced class weights, and writes `preprocessed_output/{train,val,test}_split.csv` + `preprocessing_summary.json`. Run this once after placing the dataset.
3. `data/preprocessing.py` is the runtime data pipeline — defines `SkinDataset`, transforms (resize→224, ImageNet normalization, train-only augmentation), and `get_data_loaders(splits_dir="preprocessed_output", batch_size=32)` which **reads the exported CSVs** (it does not re-split) and returns `(train_loader, val_loader, test_loader, class_weights)`.
4. Class imbalance is severe (NV ≈ 67% of train). Balanced class weights are computed in `get_data_loaders` and threaded through to `CrossEntropyLoss(weight=...)` inside `train(...)`.

## Training & evaluation

- **Entry points (project root):** `train_s1.py`, `train_s2.py`, `train_s3.py` for individual strategies; `main.py` runs all three sequentially and emits the strategy comparison chart. All thin wrappers around `models.train.run_strategy(name, num_epochs=5, ...)`.
- `models/train.py:run_strategy` is the orchestrator: loads data → builds the right model → trains → evaluates on test → saves training-curve PNG, confusion matrix PNG, best checkpoint (`results/{strategy}_best.pth`), and metrics JSON (`results/{strategy}_metrics.json`).
- `models/train.py:train` returns a `(model, history)` tuple where `history` has keys `train_loss`, `val_loss`, `train_acc`, `val_acc` — the format `evaluation/plots.plot_training_curves` expects.
- The S3 branch unfreezes one EfficientNet feature block every `unfreeze_every` epochs and rebuilds the optimizer to pick up the newly trainable params. The unfreeze logic is EfficientNet-specific (`model.features.children()`); for ResNet S3 use `unfreeze_resnet_block(model, "layer4")` etc. from `resnet_model.py` instead.
- `evaluation/metrics.py` computes balanced accuracy + macro AUC, prints classification reports, and saves a normalized confusion matrix to `figures/{strategy_name}_confusion_matrix.png`.
- `evaluation/plots.py` plots per-strategy training curves and a cross-strategy comparison bar chart.

## Known gotchas

- Imports use namespace packages (no `__init__.py` files). All entry points must be run from the **project root** so `from data.X` / `from models.X` / `from evaluation.X` resolve.
- `evaluation/metrics.py` and `evaluation/plots.py` call `plt.show()` after each save. In a headless environment switch the matplotlib backend (`matplotlib.use("Agg")`) before import.
- The `if __name__ == "__main__"` blocks in `models/model.py`, `models/resnet_model.py`, `evaluation/metrics.py`, and `evaluation/plots.py` use random dummy tensors — sanity checks, not real runs.

## Environment

Conda is the configured environment manager (`.vscode/settings.json`). Core stack: PyTorch + torchvision, scikit-learn, pandas, numpy, Pillow, matplotlib, seaborn. There is no `requirements.txt` or `environment.yml` checked in.

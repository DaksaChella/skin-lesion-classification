# Freezing Strategies in Transfer Learning for Skin Lesion Classification

**Deep Learning Project Report S26 — University of Bern**

| Name | Email |
|------|-------|
| Harsh Nayak | harsh.nayak@students.unibe.ch |
| Avin Pathak | avin.pathak@students.unibe.ch |
| Daksahaini Chellappah | daksahaini.chellappah@students.unibe.ch |
| Jasmin Injodikaran | jasmin.injodikaran@students.unibe.ch |

---

## Overview

This project investigates three transfer learning strategies for classifying skin lesions using the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). We compare how much of a pre-trained network's knowledge should be frozen or fine-tuned when adapting to a new medical imaging task.

**Three strategies are evaluated:**
- **S1 – From Scratch:** All weights randomly initialised; full network trained on HAM10000
- **S2 – Full Freeze:** Pre-trained backbone frozen; only classification head trained
- **S3 – Gradual Unfreeze:** Backbone progressively unfrozen layer by layer during training

**Two backbones are used:**
- EfficientNet-B0 (primary)
- ResNet18 (cross-architecture validation)

---

## Results

### EfficientNet-B0 (Test Set)

| Strategy | Balanced Acc. | Macro F1 | Macro AUC |
|----------|--------------|----------|-----------|
| S1 – From Scratch | 0.500 | 0.363 | 0.878 |
| S2 – Full Freeze | 0.618 | 0.461 | 0.912 |
| **S3 – Gradual Unfreeze** | **0.655** | **0.486** | **0.924** |

### ResNet18 (Test Set)

| Strategy | Balanced Acc. | Macro F1 | Macro AUC |
|----------|--------------|----------|-----------|
| S2 – Full Freeze | 0.584 | 0.423 | 0.900 |
| **S3 – Gradual Unfreeze** | **0.733** | **0.601** | **0.945** |

---

## Project Structure

```
skin-lesion-classification/
├── configs/                  # Configuration files
├── data/                     # Split CSVs and preprocessing scripts
│   └── export_splits.py      # Generates train/val/test splits
├── evaluation/               # Evaluation logic
├── figures/                  # Generated plots (confusion matrices, training curves)
├── models/                   # Model definitions
│   ├── model.py              # EfficientNet-B0 model
│   └── resnet_model.py       # ResNet18 model
├── preprocessed_output/      # Preprocessed data and split CSVs
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── report/                   # LaTeX source for the project report
├── results/                  # Saved outputs
│   └── final_test_results.csv
├── train/                    # Training scripts
│   ├── train_s1_efficientnet.py
│   ├── train_s2_efficientnet.py
│   ├── train_s3_efficientnet.py
│   ├── train_s2_resnet.py
│   └── train_s3_resnet.py
├── evaluate.py               # Evaluation and figure generation
├── main.py
└── run_preprocessing.py
```

---

## Reproducing the Experiments

### 1. Install dependencies

```bash
pip install torch torchvision efficientnet-pytorch scikit-learn pandas numpy matplotlib seaborn
```

### 2. Export splits (run once)

```bash
python data/export_splits.py
```

### 3. Train a strategy

```bash
# EfficientNet-B0
python train/train_s1_efficientnet.py   # From Scratch
python train/train_s2_efficientnet.py   # Full Freeze
python train/train_s3_efficientnet.py   # Gradual Unfreeze

# ResNet18
python train/train_s2_resnet.py
python train/train_s3_resnet.py
```

### 4. Evaluate and generate figures

```bash
python evaluate.py
```

Final test-set results are saved to `results/final_test_results.csv`.

---

## Dataset

We use the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (10,015 dermoscopic images, 7 classes). The dataset is **not included** in this repository due to its size. Download it separately and place it according to the paths in `configs/`.

---

## Report

The full project report is available in `report/`. It covers the methodology, results, and discussion in detail.

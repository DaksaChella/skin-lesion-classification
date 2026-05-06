import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# 7-class label mapping for HAM10000.
label_map = {
    "mel": 0,
    "nv": 1,
    "bcc": 2,
    "akiec": 3,
    "bkl": 4,
    "df": 5,
    "vasc": 6,
}


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "path"]
        label = int(self.df.loc[idx, "label"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(splits_dir="preprocessed_output", batch_size=32):
    """
    Build train/val/test DataLoaders from the exported split CSVs and compute
    balanced class weights from the training set.

    Args:
        splits_dir: directory containing train_split.csv, val_split.csv, test_split.csv
            (produced by data/export_splits.py).
        batch_size: number of images per batch.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
        class_weights is a torch.FloatTensor of length 7.
    """
    train_df = pd.read_csv(os.path.join(splits_dir, "train_split.csv"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val_split.csv"))
    test_df = pd.read_csv(os.path.join(splits_dir, "test_split.csv"))

    train_dataset = SkinDataset(train_df, train_transforms)
    val_dataset = SkinDataset(val_df, val_transforms)
    test_dataset = SkinDataset(test_df, val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(7),
        y=train_df["label"].values,
    )
    class_weights = torch.FloatTensor(class_weights)

    return train_loader, val_loader, test_loader, class_weights

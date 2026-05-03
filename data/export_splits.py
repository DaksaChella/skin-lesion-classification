import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "HAM10000"

metadata_path = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

image_folders = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]

label_map = {
    "mel": 0,
    "nv": 1,
    "bcc": 2,
    "akiec": 3,
    "bkl": 4,
    "df": 5,
    "vasc": 6,
}

def find_image_path(image_id):
    for folder in image_folders:
        path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None

df = pd.read_csv(metadata_path)

df["label"] = df["dx"].map(label_map)
df["path"] = df["image_id"].apply(find_image_path)

df = df.dropna(subset=["path"]).reset_index(drop=True)

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)

output_dir = "preprocessed_output"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val_split.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(7),
    y=train_df["label"].values,
)

summary = {
    "total_images": len(df),
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "split_ratio": {
        "train": "80%",
        "validation": "10%",
        "test": "10%",
    },
    "label_map": label_map,
    "class_counts_train": train_df["dx"].value_counts().to_dict(),
    "class_counts_val": val_df["dx"].value_counts().to_dict(),
    "class_counts_test": test_df["dx"].value_counts().to_dict(),
    "class_weights": {
        str(i): float(weight) for i, weight in enumerate(class_weights)
    },
}

with open(os.path.join(output_dir, "preprocessing_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

print("Done.")
print("Created:")
print("preprocessed_output/train_split.csv")
print("preprocessed_output/val_split.csv")
print("preprocessed_output/test_split.csv")
print("preprocessed_output/preprocessing_summary.json")

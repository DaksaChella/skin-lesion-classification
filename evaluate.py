import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '.')
from models.model import build_s1_from_scratch, build_s2_full_freeze, build_s3_gradual_unfreeze



# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)




# Data
DATA_PATH = "/Users/daksahainichellappah/Desktop/Semester/Deepl Learning/DL_Project/dataset_Skin Cancer MNIST_HAM10000"

val_df = pd.read_csv('preprocessed_output/val_split.csv')
val_df['path'] = DATA_PATH + '/' + val_df['path'].str.replace('HAM10000/', '', regex=False)

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        image = Image.open(self.df.loc[idx, 'path']).convert("RGB")
        label = self.df.loc[idx, 'label']
        if self.transform:
            image = self.transform(image)
        return image, label

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_loader = DataLoader(SkinDataset(val_df, val_transforms), batch_size=32, shuffle=False)

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

def evaluate_model(model, loader, strategy_name):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    auc      = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"\nResults for {strategy_name}:")
    print("=" * 40)
    print(f"Balanced Accuracy : {round(bal_acc, 4)}")
    print(f"Macro AUC         : {round(auc, 4)}")
    print(f"Macro F1          : {round(macro_f1, 4)}")
    print("=" * 40)
    return bal_acc, auc, macro_f1

def save_confusion_matrix(model, loader, strategy_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                cmap='Blues')
    plt.title(f'Confusion Matrix — {strategy_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{strategy_name}_confusion_matrix.png')
    plt.close()
    print(f"Saved: figures/{strategy_name}_confusion_matrix.png")

def save_training_curves(history, strategy_name):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'],   label='Val Loss',   marker='o')
    ax1.set_title(f'Loss — {strategy_name}')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(epochs, history['val_acc'],   label='Val Accuracy',   marker='o')
    ax2.set_title(f'Accuracy — {strategy_name}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/{strategy_name}_training_curves.png')
    plt.close()
    print(f"Saved: figures/{strategy_name}_training_curves.png")





# Load models
def build_resnet_s2():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    return model

def build_resnet_s3():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    return model

print("\nLoading saved models...")

model_eff_s1 = build_s1_from_scratch()
model_eff_s1.load_state_dict(torch.load('saved_models/efficientnet_s1.pth', map_location=device))
model_eff_s1 = model_eff_s1.to(device)

model_eff_s2 = build_s2_full_freeze()
model_eff_s2.load_state_dict(torch.load('saved_models/efficientnet_s2.pth', map_location=device))
model_eff_s2 = model_eff_s2.to(device)

model_eff_s3 = build_s3_gradual_unfreeze()
model_eff_s3.load_state_dict(torch.load('saved_models/efficientnet_s3.pth', map_location=device))
model_eff_s3 = model_eff_s3.to(device)

model_res_s2 = build_resnet_s2()
model_res_s2.load_state_dict(torch.load('saved_models/resnet18_s2.pth', map_location=device))
model_res_s2 = model_res_s2.to(device)

model_res_s3 = build_resnet_s3()
model_res_s3.load_state_dict(torch.load('saved_models/resnet18_s3.pth', map_location=device))
model_res_s3 = model_res_s3.to(device)

print("All models loaded!")




# Evaluate all models
print("\n" + "="*60)
print("EfficientNet-B0 Results")
print("="*60)
evaluate_model(model_eff_s1, val_loader, "EfficientNet_S1")
save_confusion_matrix(model_eff_s1, val_loader, "EfficientNet_S1")

evaluate_model(model_eff_s2, val_loader, "EfficientNet_S2")
save_confusion_matrix(model_eff_s2, val_loader, "EfficientNet_S2")

evaluate_model(model_eff_s3, val_loader, "EfficientNet_S3")
save_confusion_matrix(model_eff_s3, val_loader, "EfficientNet_S3")

print("\n" + "="*60)
print("ResNet18 Results")
print("="*60)
evaluate_model(model_res_s2, val_loader, "ResNet18_S2")
save_confusion_matrix(model_res_s2, val_loader, "ResNet18_S2")

evaluate_model(model_res_s3, val_loader, "ResNet18_S3")
save_confusion_matrix(model_res_s3, val_loader, "ResNet18_S3")




# Training curves
print("\nGenerating training curves...")
for name, path in [
    ("EfficientNet_S1", "saved_models/efficientnet_s1_history.pth"),
    ("EfficientNet_S2", "saved_models/efficientnet_s2_history.pth"),
    ("EfficientNet_S3", "saved_models/efficientnet_s3_history.pth"),
    ("ResNet18_S2",     "saved_models/resnet18_s2_history.pth"),
    ("ResNet18_S3",     "saved_models/resnet18_s3_history.pth"),
]:
    history = torch.load(path, map_location='cpu')
    save_training_curves(history, name)

print("\n All figures saved to figures/")
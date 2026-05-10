
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS GPU!")
else:
    device = torch.device("cpu")
    print("Using CPU")






# Data
DATA_PATH = "/Users/daksahainichellappah/Desktop/Semester/Deepl Learning/DL_Project/dataset_Skin Cancer MNIST_HAM10000"

train_df = pd.read_csv('preprocessed_output/train_split.csv')
val_df   = pd.read_csv('preprocessed_output/val_split.csv')

train_df['path'] = DATA_PATH + '/' + train_df['path'].str.replace('HAM10000/', '', regex=False)
val_df['path']   = DATA_PATH + '/' + val_df['path'].str.replace('HAM10000/', '', regex=False)

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

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(SkinDataset(train_df, train_transforms), batch_size=32, shuffle=True)
val_loader   = DataLoader(SkinDataset(val_df, val_transforms), batch_size=32, shuffle=False)

class_weights = compute_class_weight('balanced', classes=np.arange(7), y=train_df['label'].values)
class_weights = torch.FloatTensor(class_weights).to(device)






# Training functions
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

def unfreeze_resnet_block(model, block_index):
    blocks = ['layer1', 'layer2', 'layer3', 'layer4']
    if block_index < len(blocks):
        block = getattr(model, blocks[block_index])
        for param in block.parameters():
            param.requires_grad = True
        print(f"ResNet {blocks[block_index]} unfrozen")





# Build model
def build_resnet_s3():
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 7)
    print("ResNet18 S3 - Gradual Unfreeze: starts frozen")
    return model

model = build_resnet_s3()
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

NUM_EPOCHS = 10
UNFREEZE_EVERY = 3
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0

print(f"\nTraining ResNet18 S3 for {NUM_EPOCHS} epochs...")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    if epoch > 0 and epoch % UNFREEZE_EVERY == 0:
        block_index = epoch // UNFREEZE_EVERY - 1
        unfreeze_resnet_block(model, block_index)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc     = validate(model, val_loader, criterion)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

print(f"\nBest val accuracy: {best_val_acc:.4f}")





# Save
os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), 'saved_models/resnet18_s3.pth')
torch.save(history, 'saved_models/resnet18_s3_history.pth')
print(" Model saved to saved_models/resnet18_s3.pth")
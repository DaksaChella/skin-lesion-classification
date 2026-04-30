#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# In[27]:


df = pd.read_csv("HAM10000_metadata.csv")
print(df.head())


# In[28]:


#we map all our 7 labels with indices from 0-6
label_map = {
    'mel': 0,
    'nv': 1,
    'bcc': 2,
    'akiec': 3,
    'bkl': 4,
    'df': 5,
    'vasc': 6
}

df['label'] = df['dx'].map(label_map)


# In[29]:


IMAGE_FOLDERS = [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2"
]


# In[30]:


def find_image_path(image_id):
    for folder in IMAGE_FOLDERS:
        path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None


# In[31]:


df['path'] = df['image_id'].apply(find_image_path)

# remove missing images (safety step)
df = df.dropna(subset=['path'])


# In[32]:


df.head()
df.info()


# In[33]:


import os

missing = df['path'].apply(lambda x: not os.path.exists(x)).sum()
print(missing)


# In[34]:


from sklearn.model_selection import train_test_split


# In[47]:


train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df['label'],
    random_state=42
)


# In[48]:


val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df['label'],
    random_state=42
)


# In[49]:


print(len(train_df), len(val_df), len(test_df))


# In[50]:


from torchvision import transforms


# In[ ]:




train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          
    transforms.RandomHorizontalFlip(),       
    transforms.RandomRotation(10),          
    transforms.ToTensor(),                   
    transforms.Normalize(                    
        mean=[0.485, 0.456, 0.406],         
        std=[0.229, 0.224, 0.225]           

    )
])



# In[ ]:


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),           
    transforms.ToTensor(),                   
    transforms.Normalize(                    
        mean=[0.485, 0.456, 0.406],         # Mean per channel (RGB)
        std=[0.229, 0.224, 0.225]           # Std per channel (RGB)
    )
])


# In[53]:


from torch.utils.data import Dataset
from PIL import Image

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'path']
        label = self.df.loc[idx, 'label']

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# In[54]:


train_dataset = SkinDataset(train_df, train_transforms)
val_dataset = SkinDataset(val_df, val_transforms)
test_dataset = SkinDataset(test_df, val_transforms)


# In[ ]:


from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)



# Compute class weights to handle class imbalance
# NV dominates with ~67% of samples, so we need to balance the loss


train_labels = train_df['label'].values

class_weights = compute_class_weight(
    class_weight='balanced',   # Automatically balance all 7 classes
    classes=np.arange(7),      # Classes 0 to 6
    y=train_labels             # Training labels
)



# Convert to PyTorch tensor for use in loss function
class_weights = torch.FloatTensor(class_weights)



print("Class weights:", class_weights)


# In[56]:


images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


# In[57]:


print(labels[:10])


# In[58]:


import matplotlib.pyplot as plt

img = images[0].permute(1, 2, 0)  # CHW → HWC

plt.imshow(img)
plt.title(f"Label: {labels[0].item()}")
plt.axis("off")
plt.show()


# In[ ]:


def get_data_loaders(data_path, batch_size=32):


    """
    Returns train, validation and test data loaders plus class weights.

    Args:
        data_path: path to the HAM10000 dataset folder
        batch_size: number of images per batch (default: 32)

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """


    return train_loader, val_loader, test_loader, class_weights


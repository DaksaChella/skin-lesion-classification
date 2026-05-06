import torch

import torch.nn as nn

import torchvision.models as models

from torchvision.models import ResNet18_Weights

NUM_CLASSES = 7

def build_resnet_s1_from_scratch():

    # S1: ResNet18 with random weights, trained fully from scratch

    model = models.resnet18(weights=None)

    # Replace ImageNet's 1000-class layer with our 7 skin-lesion classes

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # Train all layers

    for param in model.parameters():

        param.requires_grad = True

    print("ResNet S1 - From Scratch: all layers trainable")

    return model

def build_resnet_s2_full_freeze():

    # S2: pretrained ResNet18, only final classification layer trained

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Freeze feature extractor

    for param in model.parameters():

        param.requires_grad = False

    # Replace final layer; new layer is trainable by default

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    print("ResNet S2 - Full Freeze: only final layer trainable")

    return model

def build_resnet_s3_gradual_unfreeze():

    # S3: pretrained ResNet18, starts frozen, layers unfrozen during training

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Freeze everything first

    for param in model.parameters():

        param.requires_grad = False

    # Replace final classification layer

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    print("ResNet S3 - Gradual Unfreeze: starts frozen")

    return model

def unfreeze_resnet_block(model, block_index):

    # Unfreezes a larger ResNet block, e.g. "layer4", "layer3", etc.

    block = getattr(model, block_index)

    for param in block.parameters():

        param.requires_grad = True

    print(f"{block_index} unfrozen")

def count_trainable_params(model):

    # Prints how much of the model is currently trainable

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable:,} out of {total:,} total")

def test_with_dummy_data():

    # 4 fake RGB images with same size as your HAM10000 preprocessing

    dummy_images = torch.randn(4, 3, 224, 224)

    for name, builder in [

        ("S1", build_resnet_s1_from_scratch),

        ("S2", build_resnet_s2_full_freeze),

        ("S3", build_resnet_s3_gradual_unfreeze),

    ]:

        print("=" * 50)

        print(f"TEST ResNet {name}")

        print("=" * 50)

        model = builder()

        count_trainable_params(model)

        output = model(dummy_images)

        print("Output shape:", output.shape)

        print()

    print("All ResNet models work!")

if __name__ == "__main__":

    test_with_dummy_data()
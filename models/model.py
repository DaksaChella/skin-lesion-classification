import torch
import torch.nn as nn
import torchvision.models as models



def build_s1_from_scratch():
    # S1 — random weights, nothing pretrained
    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        7
    )

    # All layers trainable
    for param in model.parameters():
        param.requires_grad = True

    print("S1 - From Scratch: all layers trainable")
    return model


def build_s2_full_freeze():
    # S2 — pretrained weights, only last layer trained
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace and unfreeze only the classification head
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        7
    )



    print("S2 - Full Freeze: only classification head trainable")
    return model





def build_s3_gradual_unfreeze():
    # S3 — pretrained weights, everything frozen at the start
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Replace and unfreeze only the classification head
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        7
    )

    print("S3 - Gradual Unfreeze: starts frozen, blocks unfrozen during training")
    return model




def unfreeze_block(model, block_index):
    # Called during training every few epochs
    # Unfreezes one block at a time from top to bottom
    blocks = list(model.features.children())
    block = blocks[block_index]

    for param in block.parameters():
        param.requires_grad = True

    print(f"Block {block_index} unfrozen")




def count_trainable_params(model):
    # Shows how many parameters are being trained
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} out of {total:,} total")




def test_with_dummy_data():
    # Dummy data — 4 random images (224x224 RGB)
    dummy_images = torch.randn(4, 3, 224, 224)

    print("=" * 50)
    print("TEST S1 — From Scratch")
    print("=" * 50)
    model_s1 = build_s1_from_scratch()
    count_trainable_params(model_s1)
    output = model_s1(dummy_images)
    print("Output shape:", output.shape)

    print()
    print("=" * 50)
    print("TEST S2 — Full Freeze")
    print("=" * 50)
    model_s2 = build_s2_full_freeze()
    count_trainable_params(model_s2)
    output = model_s2(dummy_images)
    print("Output shape:", output.shape)

    print()
    print("=" * 50)
    print("TEST S3 — Gradual Unfreeze")
    print("=" * 50)
    model_s3 = build_s3_gradual_unfreeze()
    count_trainable_params(model_s3)
    output = model_s3(dummy_images)
    print("Output shape:", output.shape)

    print()
    print("✅ All 3 models work!")



if __name__ == "__main__":
    test_with_dummy_data()
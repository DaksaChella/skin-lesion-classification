import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import build_s1_from_scratch, build_s2_full_freeze, build_s3_gradual_unfreeze, unfreeze_block


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





def train(model, train_loader, val_loader, strategy_name, num_epochs=5, unfreeze_every=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    print(f"\nTraining {strategy_name} for {num_epochs} epochs...")
    print("=" * 60)

    best_val_acc = 0

    for epoch in range(num_epochs):

        # Gradual unfreeze — unfreeze one block every few epochs
        if strategy_name == "S3" and epoch > 0 and epoch % unfreeze_every == 0:
            block_index = epoch // unfreeze_every
            total_blocks = len(list(model.features.children()))
            if block_index < total_blocks:
                unfreeze_block(model, block_index)
                # Update optimizer to include newly unfrozen params
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=1e-4
                )

        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"results/{strategy_name}_best.pth")
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    print(f"\nBest validation accuracy for {strategy_name}: {best_val_acc:.4f}")
    return model




def get_dummy_loaders():
    # Dummy data — 100 training images, 20 validation images
    train_images = torch.randn(100, 3, 224, 224)
    train_labels = torch.randint(0, 7, (100,))

    val_images = torch.randn(20, 3, 224, 224)
    val_labels = torch.randint(0, 7, (20,))

    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    return train_loader, val_loader



if __name__ == "__main__":
    train_loader, val_loader = get_dummy_loaders()

    # Test all 3 strategies
    print("TESTING S1 — From Scratch")
    model_s1 = build_s1_from_scratch()
    train(model_s1, train_loader, val_loader, strategy_name="S1", num_epochs=3)

    print("\nTESTING S2 — Full Freeze")
    model_s2 = build_s2_full_freeze()
    train(model_s2, train_loader, val_loader, strategy_name="S2", num_epochs=3)

    print("\nTESTING S3 — Gradual Unfreeze")
    model_s3 = build_s3_gradual_unfreeze()
    train(model_s3, train_loader, val_loader, strategy_name="S3", num_epochs=6)
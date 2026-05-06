from data.preprocessing import get_data_loaders


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights = get_data_loaders()

    print("Preprocessing works!")
    print(f"Train batches: {len(train_loader)} (samples: {len(train_loader.dataset)})")
    print(f"Val batches:   {len(val_loader)} (samples: {len(val_loader.dataset)})")
    print(f"Test batches:  {len(test_loader)} (samples: {len(test_loader.dataset)})")
    print(f"Class weights: {class_weights}")

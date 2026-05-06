from models.train import run_strategy


if __name__ == "__main__":
    run_strategy("S3", num_epochs=5, unfreeze_every=1)

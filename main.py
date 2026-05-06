from evaluation.plots import plot_strategy_comparison
from models.train import run_strategy


if __name__ == "__main__":
    results = {name: run_strategy(name, num_epochs=5) for name in ("S1", "S2", "S3")}
    plot_strategy_comparison({name: r["metrics"] for name, r in results.items()})

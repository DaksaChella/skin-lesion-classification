import matplotlib.pyplot as plt
import numpy as np
import os





os.makedirs('results', exist_ok=True)



def plot_training_curves(history, strategy_name):
    """
    Plots and saves training and validation loss and accuracy curves.

    Args:
        history: dictionary with keys:
                 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        strategy_name: name of the strategy e.g. 'S1', 'S2', 'S3'
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))



    # Loss curve
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    ax1.set_title(f'Loss — {strategy_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)




    # Accuracy curve
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', marker='o')
    ax2.set_title(f'Accuracy — {strategy_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'results/{strategy_name}_training_curves.png')
    plt.show()
    print(f"Training curves saved to results/{strategy_name}_training_curves.png")



def plot_strategy_comparison(results):
    """
    Plots a bar chart comparing Balanced Accuracy and AUC for all strategies.

    Args:
        results: dictionary like:
                 {
                   'S1': {'balanced_accuracy': 0.45, 'macro_auc': 0.70},
                   'S2': {'balanced_accuracy': 0.60, 'macro_auc': 0.80},
                   'S3': {'balanced_accuracy': 0.75, 'macro_auc': 0.88}
                 }
    """
    strategies = list(results.keys())
    bal_accs = [results[s]['balanced_accuracy'] for s in strategies]
    aucs = [results[s]['macro_auc'] for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width/2, bal_accs, width, label='Balanced Accuracy')
    bars2 = ax.bar(x + width/2, aucs, width, label='Macro AUC')

    ax.set_title('Strategy Comparison — S1 vs S2 vs S3')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, axis='y')



    # Add value labels on top of bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom', fontsize=10
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{bar.get_height():.2f}',
            ha='center', va='bottom', fontsize=10
        )

    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png')
    plt.show()
    print("Strategy comparison saved to results/strategy_comparison.png")





def test_with_dummy_data():
    # Dummy training history for one strategy
    dummy_history = {
        'train_loss': [2.1, 1.9, 1.7, 1.5, 1.3],
        'val_loss':   [2.2, 2.0, 1.9, 1.8, 1.7],
        'train_acc':  [0.15, 0.25, 0.35, 0.45, 0.55],
        'val_acc':    [0.12, 0.20, 0.28, 0.32, 0.38]
    }



    # Dummy results for all 3 strategies
    dummy_results = {
        'S1': {'balanced_accuracy': 0.45, 'macro_auc': 0.70},
        'S2': {'balanced_accuracy': 0.60, 'macro_auc': 0.80},
        'S3': {'balanced_accuracy': 0.75, 'macro_auc': 0.88}
    }

    print("Plotting training curves...")
    plot_training_curves(dummy_history, strategy_name="S1")

    print("Plotting strategy comparison...")
    plot_strategy_comparison(dummy_results)

    print("\n All plots work!")





if __name__ == "__main__":
    test_with_dummy_data()




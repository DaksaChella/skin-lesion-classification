import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns




# Class names for HAM10000
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']




def compute_metrics(y_true, y_pred, y_prob):
    """
    Computes all evaluation metrics.

    Args:
        y_true: true labels (list or numpy array)
        y_pred: predicted labels (list or numpy array)
        y_prob: predicted probabilities for each class (numpy array of shape [N, 7])

    Returns:
        dictionary with all metrics
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    metrics = {
        'balanced_accuracy': round(bal_acc, 4),
        'macro_auc': round(auc, 4),
    }

    return metrics




def print_metrics(metrics, strategy_name):
    """
    Prints metrics in a readable format.

    Args:
        metrics: dictionary returned by compute_metrics()
        strategy_name: name of the strategy e.g. 'S1', 'S2', 'S3'
    """
    print(f"\nResults for {strategy_name}:")
    print("=" * 40)
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']}")
    print(f"Macro AUC         : {metrics['macro_auc']}")
    print("=" * 40)




def plot_confusion_matrix(y_true, y_pred, strategy_name):
    """
    Plots and saves a normalized confusion matrix.

    Args:
        y_true: true labels
        y_pred: predicted labels
        strategy_name: name of the strategy e.g. 'S1', 'S2', 'S3'
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap='Blues'
    )
    plt.title(f'Confusion Matrix — {strategy_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{strategy_name}_confusion_matrix.png')
    plt.show()
    print(f"Confusion matrix saved to results/{strategy_name}_confusion_matrix.png")




def print_classification_report(y_true, y_pred):
    """
    Prints per-class precision, recall and F1 score.

    Args:
        y_true: true labels
        y_pred: predicted labels
    """
    print("\nPer-class Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))



def test_with_dummy_data():
    # Dummy data — 100 samples, 7 classes
    y_true = np.random.randint(0, 7, size=100)
    y_pred = np.random.randint(0, 7, size=100)
    y_prob = np.random.dirichlet(np.ones(7), size=100)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, strategy_name="S1")
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, strategy_name="S1")


if __name__ == "__main__":
    test_with_dummy_data()
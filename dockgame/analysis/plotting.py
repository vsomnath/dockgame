import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(
    y_pred: list, 
    y_true: list, 
    bins: int = 10, 
    alpha: float = 0.8, 
    show: bool = False
):
    fig, ax = plt.subplots(figsize=(8, 8))

    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()

    labels = ["pred", "true"]
    colors = ["b", "g"]

    ax.hist([y_pred, y_true], bins=bins, alpha=alpha, label=labels, color=colors)
    ax.set_title("Histograms of predicted (blue) and true (green) dock scores")
    ax.legend(loc="upper right")

    fig.tight_layout()

    # Display figure if needed
    if show:
        plt.show()
        return

    return fig


def plot_scores_vs_ids(
    y_pred: list, 
    y_true: list, 
    y_true_unord: list = None,  
    show: bool = False
):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x=np.arange(len(y_true)), y=y_pred, c="b", label="Pred")
    ax.scatter(x=np.arange(len(y_true)), y=y_true, c="g", label="True")
    if y_true_unord is not None:
        ax.scatter(x=np.arange(len(y_true_unord)), y=y_true_unord, c="black", label="True, unordered")
    ax.legend(loc="upper left")

    fig.tight_layout()
    if show:
        plt.show()
        return
    
    return fig


def plot_diff_scores_sign(
    y_pred: list, 
    y_true: list, 
    x_values: list, 
    show: bool = False
):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(x=x_values, y=y_pred, marker='x', c="b", label="Pred")
    ax.scatter(x=x_values, y=y_true, marker='o', facecolors='none', edgecolors='g', label="True")
    ax.legend(loc="upper left")

    fig.tight_layout()
    if show:
        plt.show()
        return
    
    return fig

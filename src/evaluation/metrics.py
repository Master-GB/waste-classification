import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
    class_names,
):
    """
    Calculate multiclass classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    return metrics, report, matrix


def collect_predictions(
    model,
    dataloader,
    device,
):
    """
    Collect true labels, predictions and probabilities.
    """

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.inference_mode():

        for images, labels in dataloader:

            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = probabilities.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities),
    )


"""Common, explicit action-recognition metrics for all Phase 1C models."""
import numpy as np

from e0.metrics import compute_metrics as e0_metrics


def compute_metrics(probabilities, labels, classes):
    result = e0_metrics(probabilities, labels, classes)
    confusion = np.asarray(result["confusion_matrix"], dtype=np.int64)
    support, predicted = confusion.sum(1), confusion.sum(0)
    correct = confusion.diagonal()
    recall = np.divide(correct, support, out=np.zeros(len(classes)), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(len(classes)), where=predicted > 0)
    f1 = np.asarray([row["f1"] for row in result["per_class"]])
    result.update(weighted_f1=float((f1 * support).sum() / support.sum()),
                  balanced_accuracy=float(recall[support > 0].mean()),
                  balanced_accuracy_all_selected_classes=float(recall.mean()),
                  classes_never_predicted=int((predicted == 0).sum()),
                  classes_zero_correct=int((correct == 0).sum()),
                  prediction_concentration_top10=float(np.sort(predicted)[-10:].sum() / support.sum()),
                  class_mapping=classes)
    for i, row in enumerate(result["per_class"]):
        row.update(predicted=int(predicted[i]), correct=int(correct[i]),
                   recall=float(recall[i]), precision=float(precision[i]))
    return result

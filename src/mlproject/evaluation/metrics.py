"""Classification metrics (W2, tasks Z2 and Z3). They stay in your project: you will use them to evaluate the networks.

Build them from sklearn.metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
"""

from sklearn import metrics


def binary_metrics(y_true, p, threshold):
    """Metrics when p >= threshold means class 1 (positive class 1).

    Returns a dict of floats: "accuracy", "precision", "recall", "f1", and the counts "tn", "fp", "fn", "tp".
    A ratio with a zero denominator is 0.0 (zero_division=0).
    """
    pred_class = (p >= threshold).astype(int)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, pred_class).ravel()

    accuracy = metrics.accuracy_score(y_true,pred_class)
    f1 = metrics.f1_score(y_true, pred_class, zero_division = 0)
    precision = metrics.precision_score(y_true, pred_class, zero_division = 0)
    recall = metrics.recall_score(y_true, pred_class, zero_division = 0) 
    # TODO (Z2)
    return {
	"accuracy": accuracy,
	"precision": precision,
	"recall": recall,
	"f1": f1,
	"tn": tn,
	"fp": fp,
	"fn": fn,
	"tp": tp
    }


def choose_threshold(y_true, p, thresholds, min_recall):
    """The largest threshold from `thresholds` whose recall is at least min_recall, or None if there is none."""
    # TODO (Z2)
    chosen_threshold = -1
    for threshold in thresholds:
        metrics_dict = binary_metrics(y_true, p, threshold)
        if metrics_dict["recall"] >= min_recall and threshold > chosen_threshold:
            chosen_threshold = threshold
        if chosen_threshold == -1:
            return None
        else:
            return chosen_threshold


def multiclass_metrics(y_true, y_pred):
    """Metrics for K classes.

    Returns a dict: "accuracy", "f1_macro", "f1_micro", "f1_weighted" (floats) and "recall_per_class"
    (a list of floats, one per class, in the order of the class labels).
    """
    accuracy = float(metrics.accuracy_score(y_true, y_pred))
    f1_macro = float(metrics.f1_score(y_true, y_pred, average='macro'))
    f1_micro = float(metrics.f1_score(y_true, y_pred, average='micro'))
    f1_weighted = float(metrics.f1_score(y_true, y_pred, average='weighted'))

    # Get unique classes in sorted order
    classes = sorted(set(y_true) | set(y_pred))
    recall_per_class = metrics.recall_score(y_true, y_pred, average=None, labels=classes).tolist()

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "recall_per_class": recall_per_class
    }

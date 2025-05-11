

import numpy as np


metric2name = {
    "accuracy": "Average of correct guessings",
    "nll": "Listener Cross-entropy"
}


def compute_metric(probs, target_cat, metric):
    if metric == "accuracy":
        return (probs.argmax() == target_cat).astype(float)
    elif metric == "nll":
        return -np.log(probs[target_cat])
    else:
        raise ValueError(f"Metric {metric} not supported")
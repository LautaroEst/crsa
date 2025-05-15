import numpy as np
import matplotlib.pyplot as plt


def compute_metric(probs, target, metric, prior=None, prior_target=None):
    if metric == "accuracy":
        return (probs.argmax(axis=1) == target).mean()
    elif metric == "cross_entropy":
        return -np.log(probs[np.arange(len(target)),target]).mean()
    elif metric == "igain":
        ce_prior = compute_metric(prior, prior_target, "cross_entropy")
        ce_model = compute_metric(probs, target, "cross_entropy")
        return ce_prior - ce_model
    else:
        raise ValueError(f"Metric {metric} not supported")



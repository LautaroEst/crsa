import numpy as np
import matplotlib.pyplot as plt


def compute_metric(probs, target, metric, prior=None, prior_target=None):
    if metric == "accuracy":
        r = (probs.argmax(axis=1) == target)
        return r.mean(), r.std()
    elif metric == "cross_entropy":
        r = -np.log(probs[np.arange(len(target)),target])
        return r.mean(), r.std()
    elif metric == "igain":
        ce_prior = -np.log(prior[np.arange(len(target)),prior_target])
        ce_model = -np.log(probs[np.arange(len(target)),target])
        r = ce_prior - ce_model
        return r.mean(), r.std()
    else:
        raise ValueError(f"Metric {metric} not supported")



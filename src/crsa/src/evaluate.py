import torch


def compute_metric(logprobs, target, metric, prior_logprobs=None, prior_target=None):
    if metric == "accuracy":
        r = (logprobs.argmax(axis=1) == target).float()
    elif metric == "cross_entropy":
        r = -logprobs[torch.arange(len(target)),target]
    elif metric == "igain":
        ce_prior = -prior_logprobs[torch.arange(len(prior_target)),prior_target]
        ce_model = -logprobs[torch.arange(len(target)),target]
        r = ce_prior - ce_model
    else:
        raise ValueError(f"Metric {metric} not supported")
    
    return r.mean().item(), r.std().item()




import re
import torch

ZERO = 1e-10
INF = 1e10

def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch.compiler.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)

def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Example:
    # sorted_probs=[0.1, 0.15, 0.2, 0.25, 0.3] -> sorted_cumprobs=[0.1, 0.25, 0.45, 0.7, 1.0]
    # sorted_indices_to_remove = [1, 1, 0, 0, 0] if top_p=0.7
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least 1 token always to prevent the case where no token is selected
    # In this case the most probable one is always kept
    sorted_indices_to_remove[-1:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample_utterance(logits: torch.Tensor, sampling_strategy: str) -> torch.Tensor:
    top_k, top_p = parse_sampling_strategy(sampling_strategy)
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    # optionally crop the logits to only the top k options
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.size(-1)))
        # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    # optionally scale the logits and sample from a probability distribution
    if top_p > 0.0:
        # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True).item()


def parse_sampling_strategy(sampling_strategy: str) -> tuple:
    """
    Parse the sampling strategy string into a tuple of (top_k, top_p).
    """
    if sampling_strategy == "greedy":
        top_k = None
        top_p = 0.0
    elif re.match(r"top_k_\d+", sampling_strategy):
        top_k = int(sampling_strategy.split("_")[2])
    elif re.match(r"top_p_\d+(\.\d+)?", sampling_strategy):
        top_k = None
        top_p = float(sampling_strategy.split("_")[2])
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    return top_k, top_p
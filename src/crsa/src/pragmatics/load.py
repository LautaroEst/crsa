
import torch
from .crsa import CRSA
from .yrsa import YRSA
from .prior import Prior

def init_model(model_name: str, logprior: torch.Tensor, max_depth: int = float('inf'), tolerance: float = 1e-3, save_memory: bool = False):
    if model_name == "crsa":
        return CRSA(logprior, max_depth=max_depth, tolerance=tolerance, save_memory=save_memory)
    elif model_name == "crsa-literal":
        return CRSA(logprior, max_depth=0, tolerance=float('inf'), save_memory=save_memory)
    elif model_name == "yrsa":
        return YRSA(logprior, max_depth=max_depth, tolerance=tolerance, save_memory=save_memory)
    elif model_name == "yrsa-literal":
        return YRSA(logprior, max_depth=0, tolerance=float('inf'), save_memory=save_memory)
    elif model_name == "prior":
        return Prior(logprior)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

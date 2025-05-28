
import torch
from .crsa import CRSA
from .yrsa import YRSA
from .literal import Literal
from .prior import Prior

def init_model(model_name: str, logprior: torch.Tensor, **kwargs):
    if model_name == "crsa":
        return CRSA(logprior)
    elif model_name == "rsa":
        return YRSA(logprior)
    elif model_name == "literal":
        return Literal(logprior)
    elif model_name == "prior":
        return Prior(logprior)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

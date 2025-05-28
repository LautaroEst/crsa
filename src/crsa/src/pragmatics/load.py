
import torch
from .crsa import CRSA
# from .crsa2 import CRSA as CRSA2
# from .naive_crsa import NaiveCRSA
from .yrsa import YRSA
from .literal import Literal
from .prior import Prior

def init_model(model_name: str, logprior: torch.Tensor, **kwargs):
    if model_name == "crsa":
        return CRSA(logprior)
    elif model_name == "crsa2":
        return CRSA2(torch.exp(logprior))
    elif model_name == "naive_crsa":
        return NaiveCRSA(
            kwargs["meanings_A"], 
            kwargs["meanings_B"], 
            kwargs["categories"], 
            kwargs["utterances"], 
            kwargs["lexicon_A"], 
            kwargs["lexicon_B"], 
            torch.exp(logprior).numpy())
    elif model_name == "rsa":
        return YRSA(logprior)
    elif model_name == "literal":
        return Literal(logprior)
    elif model_name == "prior":
        return Prior(logprior)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

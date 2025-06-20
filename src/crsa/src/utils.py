
import torch

def cosine_similarity(X, v):
    """
    Compute the cosine similarity between each row in X and vector v.
    
    Args:
        X (torch.Tensor): A 2D tensor where each row is a vector.
        v (torch.Tensor): A 1D tensor representing the vector to compare against.
        
    Returns:
        torch.Tensor: A 1D tensor containing the cosine similarities.
    """
    v_norm = (v ** 2).sum().sqrt().item()
    X_norm = (X ** 2).sum(dim=1).sqrt()
    dot_product = torch.mm(X, v.unsqueeze(1)).squeeze(1)
    return dot_product / (X_norm * v_norm)

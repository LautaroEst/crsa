

from pathlib import Path
import pickle

import torch


for path in Path("outputs/mddial/llama_alpha\=2.5/processed/").glob("*.pkl"):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    torch.save(data, path.with_suffix('.pt'))
    path.unlink()  # Remove the original .pkl file

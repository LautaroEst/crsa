
from .utils import ZERO

class Prior:

    def __init__(self, meanings_A, meanings_B, categories, prior):
        self.round_meaning_A = None
        self.meanings_A = meanings_A
        self.round_meaning_B = None
        self.meanings_B = meanings_B
        self.categories = categories
        self.prior = prior
        self.turns_history = []

    def get_category_distribution(self):
        if not self.turns_history:
            raise ValueError("No turns have been run yet.")
        if self.turns_history[-1]["speaker"] == "A":
            prior = self.prior.sum(axis=0) / self.prior.sum(axis=0).sum(axis=1, keepdims=True) + ZERO
        elif self.turns_history[-1]["speaker"] == "B":
            prior = self.prior.sum(axis=1) / self.prior.sum(axis=1).sum(axis=1, keepdims=True) + ZERO
        meaning_L = self.meanings_B.index(self.round_meaning_B) if self.turns_history[-1]["speaker"] == "A" else self.meanings_A.index(self.round_meaning_A)
        return prior[meaning_L]
    
    def reset(self, meaning_A, meaning_B):
        self.round_meaning_A = meaning_A
        self.round_meaning_B = meaning_B
        self.turns_history = []

    def run_turn(self, speaker):
        self.turns_history.append({"speaker": speaker})
        

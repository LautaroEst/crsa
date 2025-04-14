
from .lexicons import get_lexicon_cls


class ToyCRSA:

    def __init__(self, meanings_A, meanings_B, categories, utterances_A, utterances_B, cost_A, cost_B, lexicon, alpha=1.0, max_depth=None, tolerance=1e-6):
        # World parameters        
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.categories = categories
        self.utterances_A = utterances_A
        self.utterances_B = utterances_B
        self.cost_A = cost_A
        self.cost_B = cost_B
        self.alpha = alpha
        self.lxn_cls = get_lexicon_cls(lexicon)
        
        # Iteration parameters
        self.max_depth = max_depth
        self.tolerance = tolerance

    def run(self, output_dir, verbose=False):
        # Run CRSA
        pass

    def save(self, output_dir):
        # Save results
        pass
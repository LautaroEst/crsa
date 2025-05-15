

class LLMCRSA:

    def __init__(self, meanings_A, meanings_B, categories, llm, prior, alpha=1.0, max_depth=None, tolerance=None):
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.categories = categories
        self.llm = llm
        self.prior = prior
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance
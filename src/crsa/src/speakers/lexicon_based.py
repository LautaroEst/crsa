

import torch


class StaticLexicon:

    def __init__(self, spk_A, spk_B):
        self.spk_A = spk_A
        self.spk_B = spk_B

    def __call__(self, past_utterances, spk_name):
        costs = torch.zeros(self.spk_A.shape[1])
        if spk_name == "A":
            return torch.log(self.spk_A), costs
        else:
            return torch.log(self.spk_B), costs
    
    @classmethod
    def from_lexicon(cls, lexicon_A, lexicon_B):
        spk_A = lexicon_A / lexicon_A.sum(dim=0, keepdims=True)
        spk_B = lexicon_B / lexicon_B.sum(dim=0, keepdims=True)
        return cls(spk_A.T, spk_B.T)
    

class DynamicLexicon:

    def __init__(self, initial_spk_A, initial_spk_B):
        self.initial_spk_A = initial_spk_A
        self.initial_spk_B = initial_spk_B

    def __call__(self, past_utterances, spk_name):
        if spk_name == "A":
            spk = self.initial_spk_A.clone()
        else:
            spk = self.initial_spk_B.clone()
            
        for u in range(spk.shape[1]):
            if u in past_utterances and u != past_utterances[-1]["utterance"]:
                spk[:, u] = 0
        spk = spk / spk.sum(dim=0, keepdims=True)
        logspk = torch.log(spk)

        costs = torch.zeros(spk.shape[1])
        return logspk, costs
    
    @classmethod
    def from_lexicon(cls, lexicon_A, lexicon_B):
        spk_A = lexicon_A / lexicon_A.sum(dim=1, keepdims=True)
        spk_B = lexicon_B / lexicon_B.sum(dim=1, keepdims=True)
        return cls(spk_A.T, spk_B.T)
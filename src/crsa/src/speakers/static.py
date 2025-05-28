

import torch


class StaticSpeaker:

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


# class StaticSpeaker:

#     def __init__(self, spk_A, spk_B):
#         self.spk_A = spk_A
#         self.spk_B = spk_B

#     def __call__(self, past_utterances, spk_name):
#         costs = torch.zeros(self.spk_A.shape[1])
#         if spk_name == "A":
#             return torch.log(self.spk_A), costs
#         else:
#             return torch.log(self.spk_B), costs
    
#     @classmethod
#     def from_lexicon(cls, lexicon_A, lexicon_B):
#         return cls(lexicon_A.T, lexicon_B.T)

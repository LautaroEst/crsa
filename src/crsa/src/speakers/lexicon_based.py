

import torch


class StaticLexicon:

    def __init__(self, lexicon_A, lexicon_B): # L(u,m), S(u|m) = L(u,m) / L(:,m)
        self.logspk_A = torch.log(lexicon_A / lexicon_A.sum(dim=0, keepdims=True)).T
        self.logspk_B = torch.log(lexicon_B / lexicon_B.sum(dim=0, keepdims=True)).T
        self.costs = torch.zeros(self.logspk_A.shape[1])

    def __call__(self, past_utterances, spk_name):
        if spk_name == "A":
            logspk = self.logspk_A.clone()
        else:
            logspk = self.logspk_B.clone()

        return logspk, self.costs
        

class DynamicLexicon:

    def __init__(self, lexicon_A, lexicon_B):
        self.initial_logspk_A = torch.log(lexicon_A / lexicon_A.sum(dim=0, keepdims=True)).T
        self.initial_logspk_B = torch.log(lexicon_B / lexicon_B.sum(dim=0, keepdims=True)).T
        self.costs = torch.zeros(self.initial_logspk_A.shape[1])

    def __call__(self, past_utterances, spk_name):
        if spk_name == "A":
            logspk = self.initial_logspk_A.clone()
        else:
            logspk = self.initial_logspk_B.clone()

        for u in range(logspk.shape[1]):
            if u in [utt["utterance"] for utt in past_utterances] and u != past_utterances[-1]["utterance"]:
                logspk[:, u] = -torch.inf
        logspk[logspk.logsumexp(dim=1) == -torch.inf, :] = 0
        logspk = logspk - logspk.logsumexp(dim=1, keepdims=True)
        return logspk, self.costs
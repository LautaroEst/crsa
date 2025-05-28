
import torch

from .utils import sample_utterance


class Listener:

    def __init__(self, logprior):
        self.logprior = logprior
        self.history = []

    def init(self, lit_logspk):
        pre_softmax = torch.logsumexp(self.logprior.unsqueeze(0) + lit_logspk.T.unsqueeze(2).unsqueeze(2), dim=1)
        lit_loglst = torch.log_softmax(pre_softmax, dim=2)
        self.history = [lit_loglst]

    @property
    def literal_as_tensor(self):
        return self.history[0]
    
    @property
    def as_tensor(self):
        return self.history[-1]
    

class Speaker:

    def __init__(self, logprior, costs=None, alpha=1.0):
        self.logprior = logprior
        self.costs = costs
        self.alpha = alpha
        self.history = []

    def init(self, lit_logspk):
        self.history = [lit_logspk.clone()]

    @property
    def as_tensor(self):
        if self.history:
            return self.history[-1]
    

class LiteralTurn:
    
    def __init__(
        self,
        spk_name,
        logprior,
        costs,
        alpha,
    ):
        self.spk_name = spk_name
        self.costs = costs
        self.logprior = logprior
        self.alpha = alpha

        self.listener = Listener(logprior)
        self.speaker = Speaker(logprior, self.costs, self.alpha)

    def run(self, lit_logspk):

        # Init agents
        self.listener.init(lit_logspk)
        self.speaker.init(lit_logspk)


class Literal:

    def __init__(self, logprior):
        self.logprior = logprior

    def reset(self):
        self.turns = []

    def sample_utterance(self, meaning_S, sampling_strategy):
        # Get pragmatic speaker of the last turn
        prag_logspk = self.turns[-1].speaker.as_tensor

        # Sample an utterance from the pragmatic speaker
        utt_idx = sample_utterance(prag_logspk, meaning_S, sampling_strategy)

        return utt_idx
    
    def run_turn(self, lit_logspk, spk_name, costs, alpha=1.0, max_depth=float('inf'), tolerance=1e-3):

        logprior = self.logprior.clone() if spk_name == "A" else self.logprior.clone().transpose(0, 1)

        model = LiteralTurn(
            spk_name=spk_name,
            logprior=logprior,
            costs=costs,
            alpha=alpha,
        )
        model.run(lit_logspk)
        self.turns.append(model)

        return model.speaker.as_tensor, model.listener.as_tensor

        

import torch

from .utils import sample_utterance


class Listener:

    def __init__(self, logprior):
        self.logprior = logprior
        self.history = []

    def init(self, utt_dim):
        logprior_ly = torch.logsumexp(self.logprior, dim=0)
        logprior_y_give_l = logprior_ly - torch.logsumexp(logprior_ly, dim=1, keepdim=True)
        logprior_y_give_l[logprior_y_give_l.isnan()] = -torch.inf
        lit_loglst = logprior_y_give_l.unsqueeze(0).repeat(utt_dim, 1, 1) 
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

    def init(self):
        prior_spk = torch.log_softmax(-self.alpha * self.costs, dim=0)
        prior_spk = prior_spk.view(1,-1).repeat(self.logprior.shape[0], 1)
        self.history = [prior_spk]

    @property
    def as_tensor(self):
        if self.history:
            return self.history[-1]
    

class PriorTurn:
    
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
        self.listener.init(lit_logspk.shape[1])
        self.speaker.init()


class Prior:

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

        model = PriorTurn(
            spk_name=spk_name,
            logprior=logprior,
            costs=costs,
            alpha=alpha,
        )
        model.run(lit_logspk)
        self.turns.append(model)

        return model.speaker.as_tensor, model.listener.as_tensor

        
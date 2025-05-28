
import torch

from .utils import sample_utterance


class Listener:

    def __init__(self, logprior):
        self.logprior = logprior
        self.history = []

    def init(self, lit_logspk):
        lit_loglst = self._update(lit_logspk)
        self.history = [lit_loglst]

    def update(self, speaker):

        if not self.history:
            raise ValueError("Listener has not been initialized. Call init() first.")

        logspk = speaker.as_tensor.clone()
        prag_lst = self._update(logspk)
        self.history.append(prag_lst)

    def _update(self, logspk):
        pre_softmax = torch.logsumexp(self.logprior.unsqueeze(0) + logspk.T.unsqueeze(2).unsqueeze(2), dim=1)
        prag_loglst = torch.log_softmax(pre_softmax, dim=2)
        return prag_loglst

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

        # Compute conditional priors
        logprior_sl = torch.logsumexp(self.logprior, dim=2, keepdim=True) # logP(s,l,:)
        logprior_s = torch.logsumexp(logprior_sl, dim=1, keepdim=True) # P(s)
        logprior_l_given_s = logprior_sl - logprior_s # P(l|s)
        logprior_l_given_s[logprior_l_given_s.isnan()] = -torch.inf
        logprior_yl_given_s = self.logprior - logprior_s # P(y,l|s)
        logprior_yl_given_s[logprior_yl_given_s.isnan()] = -torch.inf
        logprior_y_given_sl = self.logprior - logprior_sl # P(y|s,l)
        logprior_y_given_sl[logprior_y_given_sl.isnan()] = -torch.inf

        self.logprior_l_given_s = logprior_l_given_s
        self.logprior_yl_given_s = logprior_yl_given_s
        self.logprior_y_given_sl = logprior_y_given_sl

    def update(self, listener):

        if self.history is None:
            raise ValueError("Speaker has not been initialized. Call init() before update().")

        # Compute the listener value: V_L = log(L(u,l,y)) - C(u)
        v_l = listener.as_tensor - self.costs.view(-1, 1, 1)

        pre_softmax = torch.exp(self.logprior_yl_given_s).unsqueeze(0) * v_l.unsqueeze(1)  # P(:,s,l,y) * V_L(u,:,l,y)
        pre_softmax[pre_softmax.isnan()] = 0.0  # approx x * log(x) = 0
        pre_softmax = pre_softmax.sum(dim=2).sum(dim=2).T
        prag_logspk = torch.log_softmax(self.alpha * pre_softmax, dim=1)
        prag_logspk[prag_logspk.isnan()] = -torch.inf 
        self.history.append(prag_logspk)
        
    @property
    def as_tensor(self):
        if self.history:
            return self.history[-1]
    


class YRSAGain:

    def __init__(self, logprior, costs, alpha):
        self.logprior = logprior
        self.costs = costs
        self.alpha = alpha

    def init(self):
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []

    def _compute_cond_entropy(self, speaker):
        prag_logspk = speaker.as_tensor
        logprior_s = torch.logsumexp(torch.logsumexp(self.logprior, dim=1), dim=1).unsqueeze(0)
        logspk = prag_logspk.T
        logps = logprior_s + logspk
        ps = torch.exp(logps)
        
        cond_entropy_us = ps * logspk
        cond_entropy_us[cond_entropy_us.isnan()] = 0.0
        cond_entropy = -torch.sum(cond_entropy_us)
        
        return cond_entropy
    
    def _listener_value(self, listener, speaker):
        prag_logspk = speaker.as_tensor
        logprior = self.logprior.unsqueeze(0)
        logspk = prag_logspk.T.unsqueeze(2).unsqueeze(2)
        logps = logprior + logspk
        ps = torch.exp(logps)

        v_l = listener.as_tensor - self.costs.view(-1, 1, 1) # V_L(u,l,y)
        v_l = v_l.unsqueeze(1)

        v_l_usly = ps * v_l
        v_l_usly[v_l_usly.isnan()] = 0.0
        expected_v_l = torch.sum(v_l_usly)
        return expected_v_l

    def compute_gain(self, listener, speaker):
        if self.cond_entropy_history is None:
            raise ValueError("Gain has not been initialized. Call init() before compute_gain().")

        # H_S(U|Ms,W)
        cond_entropy = self._compute_cond_entropy(speaker)
        self.cond_entropy_history.append(cond_entropy)

        # E[V_L]
        expected_v_l = self._listener_value(listener, speaker)
        self.listener_value_history.append(expected_v_l)

        # Gain = H_S(U|Ms,W) + alpha * E[V_L]
        gain = cond_entropy + self.alpha * expected_v_l
        self.gain_history.append(gain)
        return gain
    
    def get_diff(self):
        if len(self.gain_history) < 2:
            return float("inf")
        return self.gain_history[-1] - self.gain_history[-2] / abs(self.gain_history[-2])



class YRSATurn:
    
    def __init__(
        self,
        spk_name,
        logprior,
        costs,
        alpha,
        max_depth,
        tolerance,
    ):
        self.spk_name = spk_name
        self.costs = costs
        self.logprior = logprior
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.listener = Listener(logprior)
        self.speaker = Speaker(logprior, self.costs, self.alpha)
        self.gain = YRSAGain(logprior, self.costs, self.alpha)

    def run(self, lit_logspk):

        # Init agents
        self.listener.init(lit_logspk)
        self.speaker.init(lit_logspk)
        self.gain.init()

        # Run the model for the given number of iterations
        i = 0
        while i < self.max_depth:

            # First update the speaker then the listener
            self.speaker.update(self.listener)
            self.listener.update(self.speaker)

            # Check for convergence
            gain = self.gain.compute_gain(self.listener, self.speaker)
            if self.gain.get_diff() < self.tolerance:
                break
            i += 1



class YRSA:

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

        model = YRSATurn(
            spk_name=spk_name,
            logprior=logprior,
            costs=costs,
            alpha=alpha,
            max_depth=max_depth,
            tolerance=tolerance,
        )
        model.run(lit_logspk)
        self.turns.append(model)

        return model.speaker.as_tensor, model.listener.as_tensor

        
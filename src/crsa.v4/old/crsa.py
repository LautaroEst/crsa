
import torch

from .utils import sample_utterance


class Listener:

    def __init__(self, lit_logspk, logbelief_L, logprior):
        self.lit_logspk = lit_logspk
        self.logbelief_L = logbelief_L
        self.logprior = logprior

    def init(self):
        lit_loglst = self._update(self.lit_logspk)
        self.history = [lit_loglst]

    def _update(self, logspk):
        if self.logbelief_L is None:
            logbelief_L = torch.zeros(logspk.shape[0])
        else:
            logbelief_L = self.logbelief_L
        pre_softmax = torch.logsumexp(logbelief_L.view(1,-1,1,1) + self.logprior.unsqueeze(0) + logspk.T.unsqueeze(2).unsqueeze(2), dim=1)
        prag_loglst = torch.log_softmax(pre_softmax, dim=2)
        return prag_loglst

    def update(self, speaker):
        logspk = speaker.as_tensor(log=True)
        prag_loglst = self._update(logspk)
        self.history.append(prag_loglst)

    def as_tensor(self, log=True):
        if self.history and log:
            return self.history[-1]
        elif self.history and not log:
            return torch.exp(self.history[-1])
        else:
            raise ValueError("Listener history is empty. Call init() before accessing as_tensor().")
        
    def literal_as_tensor(self, log=True):
        if self.history and log:
            return self.history[0]
        elif self.history and not log:
            return torch.exp(self.history[0])
        else:
            raise ValueError("Listener history is empty. Call init() before accessing as_tensor().")
    

class Speaker:

    def __init__(self, lit_logspk, logbelief_S, logprior, costs, alpha):
        self.lit_logspk = lit_logspk
        self.logbelief_S = logbelief_S
        self.logprior = logprior
        self.costs = costs
        self.alpha = alpha
        self.history = None

    def init(self):
        self.history = [self.lit_logspk.clone()]

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
        v_l = listener.as_tensor(log=True) - self.costs.view(-1, 1, 1)

        if self.logbelief_S is None:
            logbelief_prime_t = self.logprior_l_given_s + self.logprior_y_given_sl
        else:
            logbelief_prime_t = torch.log_softmax(self.logbelief_S.view(-1,1,1) + self.logprior_l_given_s, dim=1) + self.logprior_y_given_sl

        pre_softmax = torch.exp(logbelief_prime_t).unsqueeze(0) * v_l.unsqueeze(1)  # P(:,s,l,y) * V_L(u,:,l,y)
        pre_softmax[pre_softmax.isnan()] = 0.0  # approx x * log(x) = 0
        pre_softmax = pre_softmax.sum(dim=2).sum(dim=2).T
        prag_logspk = torch.log_softmax(self.alpha * pre_softmax, dim=1)
        prag_logspk[prag_logspk.isnan()] = -torch.inf 
        self.history.append(prag_logspk)
        
    def as_tensor(self, log=True):
        if self.history and log:
            return self.history[-1]
        elif self.history and not log:
            return torch.exp(self.history[-1])
        else:
            raise ValueError("Speaker history is empty. Call init() before accessing as_tensor().")
        
    def literal_as_tensor(self, log=True):
        if self.history and log:
            return self.history[0]
        elif self.history and not log:
            return torch.exp(self.history[0])
        else:
            raise ValueError("Speaker history is empty. Call init() before accessing as_tensor().")
        

class CRSAGain:

    def __init__(self, logprior, logbelief_S, logbelief_L, costs, alpha):
        self.logprior = logprior
        self.logbelief_S = logbelief_S
        self.logbelief_L = logbelief_L
        self.costs = costs
        self.alpha = alpha

        self.cond_entropy_history = None
        self.listener_value_history = None
        self.gain_history = None

    def init(self):
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        
    def compute_gain(self, listener, speaker):

        if self.cond_entropy_history is None:
            raise ValueError("Gain has not been initialized. Call init() before compute_gain().")
        
        # P_S(u,ms,ml,y)
        prag_logspk = speaker.as_tensor(log=True)
        logps = self.logprior.unsqueeze(0) + prag_logspk.T.unsqueeze(2).unsqueeze(2)
        if self.logbelief_S is not None:
            logps += self.logbelief_S.view(1, -1, 1, 1)
            logps += self.logbelief_L.view(1, 1, -1, 1)
        ps = torch.exp(logps)
        
        # V_L(u,ml,y)
        v_l = listener.as_tensor(log=True) - self.costs.view(-1, 1, 1)

        # H_S(U|Ms,W)
        cond_entropy = ps * logps
        cond_entropy[cond_entropy.isnan()] = 0.0
        cond_entropy = -torch.sum(cond_entropy)
        self.cond_entropy_history.append(cond_entropy)

        # E[V_L]
        expected_v_l = ps * v_l.unsqueeze(1)
        expected_v_l[expected_v_l.isnan()] = 0.0
        expected_v_l = torch.sum(expected_v_l)
        self.listener_value_history.append(expected_v_l)

        # Gain = H_S(U|Ms,W) + alpha * E[V_L]
        gain = cond_entropy + self.alpha * expected_v_l
        self.gain_history.append(gain)
        
        return gain
    
    def get_diff(self):
        if len(self.gain_history) < 2:
            return torch.inf
        return torch.abs((self.gain_history[-1] - self.gain_history[-2]) / self.gain_history[-2])



class CRSATurn:
    
    def __init__(
        self,
        spk_name,
        lit_logspk,
        logprior,
        logbelief_S,
        logbelief_L,
        costs,
        alpha,
        max_depth,
        tolerance,
    ):
        self.spk_name = spk_name
        self.costs = costs
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.speaker = Speaker(lit_logspk, logbelief_S, logprior, costs, alpha)
        self.listener = Listener(lit_logspk, logbelief_L, logprior)
        self.gain = CRSAGain(logprior, logbelief_S, logbelief_L, costs, alpha)

    def run(self):

        # Init agents
        self.listener.init()
        self.speaker.init()
        self.gain.init()

        # Run the model for the given number of iterations
        i = 0
        while i < self.max_depth:

            # First update the speaker and then the listener
            self.speaker.update(self.listener)
            self.listener.update(self.speaker)

            # Check for convergence
            self.gain.compute_gain(self.listener, self.speaker)
            if self.gain.get_diff() < self.tolerance:
                break
            i += 1



class CRSA:

    def __init__(self, logprior: torch.Tensor):
        self.logprior = logprior
        self.logbeliefs = []
        self.turns = []

    def reset(self):
        self.logbeliefs = []
        self.turns = []

    def sample_utterance(self, meaning_S, sampling_strategy):
        # Get pragmatic speaker of the last turn
        prag_logspk = self.turns[-1].speaker.as_tensor(log=True)

        # Sample an utterance from the pragmatic speaker
        new_utt = sample_utterance(prag_logspk, meaning_S, sampling_strategy)

        # Update the belief history
        spk_name = self.turns[-1].spk_name
        if spk_name == "A" and self.logbeliefs:
            self.logbeliefs.append({
                "A": self.logbeliefs[-1]["A"] + prag_logspk[:, new_utt],
                "B": self.logbeliefs[-1]["B"].clone()
            })
        elif spk_name == "A" and not self.logbeliefs:
            self.logbeliefs.append({
                "A": prag_logspk[:, new_utt].clone(),
                "B": torch.zeros(self.logprior.shape[1], dtype=prag_logspk.dtype, device=prag_logspk.device)
            })
        elif spk_name == "B" and self.logbeliefs:
            self.logbeliefs.append({
                "A": self.logbeliefs[-1]["A"].clone(),
                "B": self.logbeliefs[-1]["B"] + prag_logspk[:, new_utt]
            })
        else:
            self.logbeliefs.append({
                "A": torch.zeros(self.logprior.shape[0], dtype=prag_logspk.dtype, device=prag_logspk.device),
                "B": prag_logspk[:, new_utt].clone()
            })

        return new_utt

    def run_turn(self, lit_logspk, spk_name, costs, alpha=1.0, max_depth=float('inf'), tolerance=1e-3):

        # Listener name
        lst_name = "B" if spk_name == "A" else "A"
        
        # Prior and beliefs of the turn
        logprior = self.logprior.clone() if spk_name == "A" else self.logprior.clone().transpose(0, 1)
        logbelief_S = self.logbeliefs[-1][spk_name] if self.logbeliefs else None
        logbelief_L = self.logbeliefs[-1][lst_name] if self.logbeliefs else None

        # Init model
        model = CRSATurn(
            spk_name=spk_name,
            lit_logspk=lit_logspk,
            logprior=logprior,
            logbelief_S=logbelief_S,
            logbelief_L=logbelief_L,
            costs=costs,
            alpha=alpha,
            max_depth=max_depth,
            tolerance=tolerance,
        )

        # Run the model and save it
        model.run()
        self.turns.append(model)
        
        prag_logspk = model.speaker.as_tensor(log=True)
        prag_list = model.listener.as_tensor(log=True)
        return prag_logspk, prag_list

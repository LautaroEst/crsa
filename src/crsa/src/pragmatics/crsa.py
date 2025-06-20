
from pathlib import Path
import torch

from .utils import sample_utterance


class Listener:

    def __init__(self, logprior, logbelief_L, save_memory=False):
        self.logprior = logprior
        self.logbelief_L = logbelief_L
        self.history = []
        self.save_memory = save_memory

    def init(self, lit_logspk):
        lit_loglst = self._update(lit_logspk)
        self.history = [lit_loglst]

    def update(self, speaker):

        if not self.history:
            raise ValueError("Listener has not been initialized. Call init() first.")

        logspk = speaker.as_tensor.clone()
        prag_lst = self._update(logspk)

        if self.save_memory and len(self.history) > 0:
            self.history[-1] = prag_lst
        else:
            self.history.append(prag_lst)

    def _update(self, logspk):
        if self.logbelief_L is None:
            logbelief_L = torch.zeros(logspk.shape[0])
        else:
            logbelief_L = self.logbelief_L
        pre_softmax = torch.logsumexp(logbelief_L.view(1,-1,1,1) + self.logprior.unsqueeze(0) + logspk.T.unsqueeze(2).unsqueeze(2), dim=1)
        prag_loglst = torch.log_softmax(pre_softmax, dim=2)
        return prag_loglst

    @property
    def literal_as_tensor(self):
        if self.save_memory:
            raise ValueError("Literal tensor is not available when save_memory is True.")
        return self.history[0]
    
    @property
    def as_tensor(self):
        return self.history[-1]
    

class Speaker:

    def __init__(self, logprior, logbelief_S=None, costs=None, alpha=1.0, save_memory=False):
        self.logprior = logprior
        self.logbelief_S = logbelief_S
        self.costs = costs
        self.alpha = alpha
        self.history = []
        self.save_memory = save_memory

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

        if self.logbelief_S is None:
            logbelief_prime_t = self.logprior_l_given_s + self.logprior_y_given_sl
        else:
            logbelief_prime_t = torch.log_softmax(self.logbelief_S.view(1,-1,1) + self.logprior_l_given_s, dim=1) + self.logprior_y_given_sl

        pre_softmax = torch.exp(logbelief_prime_t).unsqueeze(0) * v_l.unsqueeze(1)  # P(:,s,l,y) * V_L(u,:,l,y)
        pre_softmax[pre_softmax.isnan()] = 0.0  # approx x * log(x) = 0
        pre_softmax = pre_softmax.sum(dim=2).sum(dim=2).T
        prag_logspk = torch.log_softmax(self.alpha * pre_softmax, dim=1)
        prag_logspk[prag_logspk.isnan()] = -torch.inf 

        if self.save_memory and len(self.history) > 0:
            self.history[-1] = prag_logspk
        else:
            self.history.append(prag_logspk)
        
    @property
    def literal_as_tensor(self):
        if self.save_memory:
            raise ValueError("Literal tensor is not available when save_memory is True.")
        return self.history[0]
    
    @property
    def as_tensor(self):
        return self.history[-1]
    


class CRSAGain:

    def __init__(self, logprior, logbelief_L, logbelief_S, costs, alpha):
        self.logprior = logprior
        self.logbelief_L = logbelief_L if logbelief_L is not None else torch.ones(logprior.shape[1], dtype=logprior.dtype, device=logprior.device)
        self.logbelief_S = logbelief_S if logbelief_S is not None else torch.ones(logprior.shape[0], dtype=logprior.dtype, device=logprior.device)
        self.costs = costs
        self.alpha = alpha

    def init(self, listener, speaker):
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        self.compute_gain(listener, speaker)

    def _compute_cond_entropy(self, speaker):
        prag_logspk = speaker.as_tensor
        logprior_s = torch.logsumexp(torch.logsumexp(self.logprior, dim=1), dim=1).unsqueeze(0)
        logspk = prag_logspk.T
        logbelief_L = self.logbelief_L.view(1,-1) if self.logbelief_L is not None else torch.zeros(1, prag_logspk.shape[0], dtype=self.logprior.dtype, device=self.logprior.device)
        logps = logbelief_L + logprior_s + logspk
        logps = logps - torch.logsumexp(logps, dim=(0,1))
        
        cond_entropy_us = torch.exp(logps) * logspk
        cond_entropy_us[cond_entropy_us.isnan()] = 0.0
        cond_entropy = -torch.sum(cond_entropy_us)
        
        return cond_entropy
    
    def _listener_value(self, listener, speaker):
        prag_logspk = speaker.as_tensor
        logprior = self.logprior.unsqueeze(0)
        logspk = prag_logspk.T.unsqueeze(2).unsqueeze(2)
        logbelief_L = self.logbelief_L.view(1, -1, 1, 1) if self.logbelief_L is not None else torch.zeros(1, logprior.shape[1], 1, 1, dtype=self.logprior.dtype, device=self.logprior.device)
        logbelief_S = self.logbelief_S.view(1, 1, -1, 1) if self.logbelief_S is not None else torch.zeros(1, 1, logprior.shape[2], 1, dtype=self.logprior.dtype, device=self.logprior.device)
        logps = logbelief_L + logbelief_S + logprior + logspk # Ps(u,s,l,y)

        x = (logbelief_L + logprior).squeeze()
        x = x - torch.logsumexp(x, dim=(0,1))
        x = torch.exp(x)

        logps = logps - torch.logsumexp(logps, dim=(0, 1, 2, 3))

        v_l = listener.as_tensor - self.costs.view(-1, 1, 1) # V_L(u,l,y)
        v_l = v_l.unsqueeze(1) # V_L(u,:,l,y)
        
        v_l_usly = torch.exp(logps) * v_l
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
            return torch.inf
        elif self.gain_history[-1] - self.gain_history[-2] == 0: 
            return torch.tensor(0.)
        else:
            return self.gain_history[-1] - self.gain_history[-2] / abs(self.gain_history[-2])



class CRSATurn:
    
    def __init__(
        self,
        spk_name,
        logprior,
        logbelief_S,
        logbelief_L,
        costs,
        alpha,
        max_depth,
        tolerance,
        save_memory=False
    ):
        self.spk_name = spk_name
        self.costs = costs
        self.logprior = logprior
        self.logbelief_S = logbelief_S
        self.logbelief_L = logbelief_L
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.listener = Listener(logprior, self.logbelief_L, save_memory=save_memory)
        self.speaker = Speaker(logprior, self.logbelief_S, self.costs, self.alpha, save_memory=save_memory)
        self.gain = CRSAGain(logprior, self.logbelief_L, self.logbelief_S, self.costs, self.alpha)

    def run(self, lit_logspk):

        # Init agents
        self.listener.init(lit_logspk)
        self.speaker.init(lit_logspk)
        self.gain.init(self.listener, self.speaker)

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

    @property
    def iter_num(self):
        return len(self.gain.gain_history)
    

    def save(self, path: Path):
        torch.save({
            'spk_name': self.spk_name,
            'logprior': self.logprior,
            'logbelief_S': self.logbelief_S,
            'logbelief_L': self.logbelief_L,
            'costs': self.costs,
            'alpha': self.alpha,
            'max_depth': self.max_depth,
            'tolerance': self.tolerance,
            'iter_num': self.iter_num,
            'listeners': self.listener.history,
            'speakers': self.speaker.history,
            'gain_history': self.gain.gain_history,
            'cond_entropy_history': self.gain.cond_entropy_history,
            'listener_value_history': self.gain.listener_value_history,
        }, path)

    @classmethod
    def load(cls, path: Path):
        data = torch.load(path, weights_only=False)
        model = cls(
            spk_name=data['spk_name'],
            logprior=data['logprior'],
            logbelief_S=data['logbelief_S'],
            logbelief_L=data['logbelief_L'],
            costs=data['costs'],
            alpha=data['alpha'],
            max_depth=data['max_depth'],
            tolerance=data['tolerance']
        )
        model.listener.history = data['listeners']
        model.speaker.history = data['speakers']
        model.gain.gain_history = data['gain_history']
        model.gain.cond_entropy_history = data['cond_entropy_history']
        model.gain.listener_value_history = data['listener_value_history']
        return model





class CRSA:

    def __init__(self, logprior, max_depth=float('inf'), tolerance=1e-3, save_memory=False):
        self.logprior = logprior
        self.max_depth = max_depth
        self.tolerance = tolerance
        self.save_memory = save_memory

    def reset(self):
        self.logbeliefs = []
        self.turns = []

    def sample_utterance(self, meaning_S, sampling_strategy, output_dist=False):
        # Get pragmatic speaker of the last turn
        prag_logspk = self.turns[-1].speaker.as_tensor

        # Sample an utterance from the pragmatic speaker
        logits = prag_logspk[meaning_S, :]
        utt_idx = sample_utterance(logits, sampling_strategy)

        if output_dist:
            dist = torch.softmax(logits, dim=0)
            return utt_idx, dist
        else:
            return utt_idx

    def update_belief_(self, utt_idx):

        # Update the belief history
        model = self.turns[-1]
        spk_name = self.turns[-1].spk_name
        if spk_name == "A":
            if not self.logbeliefs:
                self.logbeliefs.append({
                    "B": model.speaker.as_tensor[:, utt_idx].clone(),
                    "A": torch.zeros(self.logprior.shape[1], dtype=self.logprior.dtype, device=self.logprior.device),
                })
            else:
                self.logbeliefs.append({
                    "A": self.logbeliefs[-1]["A"],
                    "B": self.logbeliefs[-1]["B"] + model.speaker.as_tensor[:, utt_idx]
                })
        else:
            if not self.logbeliefs:
                self.logbeliefs.append({
                    "A": model.speaker.as_tensor[:, utt_idx].clone(),
                    "B": torch.zeros(self.logprior.shape[0], dtype=self.logprior.dtype, device=self.logprior.device),
                })
            else:
                self.logbeliefs.append({
                    "A": self.logbeliefs[-1]["A"] + model.speaker.as_tensor[:, utt_idx],
                    "B": self.logbeliefs[-1]["B"],
                })

    
    def run_turn(self, lit_logspk, spk_name, costs, alpha=1.0):

        logprior = self.logprior.clone() if spk_name == "A" else self.logprior.clone().transpose(0, 1)

        lst_name = "B" if spk_name == "A" else "A"
        logbelief_S = self.logbeliefs[-1][spk_name] if self.logbeliefs else None
        logbelief_L = self.logbeliefs[-1][lst_name] if self.logbeliefs else None

        model = CRSATurn(
            spk_name=spk_name,
            logprior=logprior,
            logbelief_S=logbelief_S,
            logbelief_L=logbelief_L,
            costs=costs,
            alpha=alpha,
            max_depth=self.max_depth,
            tolerance=self.tolerance,
            save_memory=self.save_memory
        )
        model.run(lit_logspk)
        if self.save_memory and len(self.turns) > 0:
            self.turns[-1] = model
        else:
            self.turns.append(model)

        return model.speaker.as_tensor, model.listener.as_tensor

    def save(self, path: Path):
        torch.save({
            'logprior': self.logprior,
            'max_depth': self.max_depth,
            'tolerance': self.tolerance,
            'save_memory': self.save_memory,
            'logbeliefs': self.logbeliefs,
            'turns': len(self.turns),
        }, path / 'model.pt')
        for t, turn in enumerate(self.turns, start=1):
            turn.save(path / f'turn_{t}.pt')
    
    @classmethod
    def load(cls, path: Path):
        data = torch.load(path / "model.pt", weights_only=False)
        model = cls(
            logprior=data['logprior'],
            max_depth=data['max_depth'],
            tolerance=data['tolerance'],
            save_memory=data['save_memory']
        )
        model.logbeliefs = data['logbeliefs']
        model.turns = [CRSATurn.load(path / f'turn_{t}.pt') for t in range(1, data['turns'] + 1)]
        return model


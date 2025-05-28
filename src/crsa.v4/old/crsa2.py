
import torch

from .utils import sample_utterance


ZERO = 1e-10
INF = 1e10


class Listener:

    def __init__(self, lit_spk, belief_L, prior):
        self.lit_spk = lit_spk
        self.dm = belief_L
        self.prior = prior

    def init(self):
        lit_lst = self._update(self.lit_spk.clone())
        self.history = [lit_lst]

    def _update(self, spk):
        spk[spk <= ZERO] = ZERO

        if self.dm is None:
            # lexicon(u,a), prior(a,b,y)
            pragmatic_listener = torch.einsum('au,aby->uby', spk, self.prior)
        else:
            # prior(a,b,y), dm(a,b), speaker(a,u)
            pragmatic_listener = torch.einsum('a,aby,au->uby', self.dm, self.prior, spk)

        pragmatic_listener[pragmatic_listener <= ZERO] = ZERO
        norm_term = pragmatic_listener.sum(dim=-1, keepdim=True)
        norm_term[norm_term <= ZERO] = ZERO
        pragmatic_listener = pragmatic_listener / norm_term
        return pragmatic_listener

    def update(self, speaker):
        spk = speaker.as_tensor(log=False).clone()
        prag_lst = self._update(spk)
        self.history.append(prag_lst)

    def as_tensor(self, log=True):
        if self.history and log:
            return torch.log(self.history[-1])
        elif self.history and not log:
            return self.history[-1]
        else:
            raise ValueError("Listener history is empty. Call init() before accessing as_tensor().")
        
    def literal_as_tensor(self, log=True):
        if self.history and log:
            return torch.log(self.history[0])
        elif self.history and not log:
            return self.history[0]
        else:
            raise ValueError("Listener history is empty. Call init() before accessing as_tensor().")
    

class Speaker:

    def __init__(self, lit_spk, belief_S, prior, costs, alpha):
        self.lit_spk = lit_spk
        self.prior = prior
        self.dm = belief_S
        self.cost = costs
        self.alpha = alpha
        self.history = []

    def init(self):
        self.history = [self.lit_spk.clone()]

    def update(self, listener):

        # Compute the conditional priors 
        prior = self.prior.clone()
        prior[prior <= ZERO] = ZERO
        prior_ab = prior.sum(dim=2) # P(a,b)
        prior_a = prior_ab.sum(dim=1, keepdim=True) # P(a)
        prior_ab[prior_ab <= ZERO] = ZERO
        prior_a[prior_a <= ZERO] = ZERO
        prior_b_given_a = prior_ab / prior_a # P(b|a)
        prior_by_given_a = prior / prior_a # P(b,y|a)
        prior_y_given_ab = prior / prior_ab.unsqueeze(2) # P(y|a,b)

        # Compute the log_listener
        listener_arr = listener.as_tensor(log=False).clone()
        mask = listener_arr > 0
        log_listener = torch.zeros_like(listener_arr, dtype=torch.float)
        log_listener[mask] = torch.log(listener_arr[mask])
        log_listener[~mask] = -INF

        if self.dm is None:
            # Compute the pragmatic speaker
            log_pragmatic_speaker = self.alpha * torch.einsum('aby,uby->au', prior_by_given_a, log_listener) - self.cost.reshape(1,-1)
        else:
            # Compute the quotient of dm
            dm_num = torch.einsum('b,ab->ab', self.dm, prior_b_given_a)
            dm_num[dm_num <= ZERO] = ZERO
            dm_frac = dm_num / dm_num.sum(dim=1, keepdims=True)
            dm_frac[dm_frac <= ZERO] = ZERO

            # Compute the pragmatic speaker
            log_pragmatic_speaker = self.alpha * torch.einsum('ab,aby,uby->au', dm_frac, prior_y_given_ab, log_listener) - self.cost.reshape(1,-1)
        
        pragmatic_speaker = torch.softmax(log_pragmatic_speaker, dim=1)
        pragmatic_speaker[pragmatic_speaker <= ZERO] = ZERO
        self.history.append(pragmatic_speaker)
        
    def as_tensor(self, log=True):
        if self.history and log:
            return torch.log(self.history[-1])
        elif self.history and not log:
            return self.history[-1]
        else:
            raise ValueError("Speaker history is empty. Call init() before accessing as_tensor().")
        
    def literal_as_tensor(self, log=True):
        if self.history and log:
            return torch.log(self.history[0])
        elif self.history and not log:
            return self.history[0]
        else:
            raise ValueError("Speaker history is empty. Call init() before accessing as_tensor().")
        

class CRSAGain:

    def __init__(self, prior, belief_S, belief_L, costs, alpha):
        self.prior = prior
        self.dm_s = belief_S if belief_S is not None else torch.ones(prior.shape[0])
        self.dm_l = belief_L if belief_L is not None else torch.ones(prior.shape[1])
        self.cost = costs
        self.alpha = alpha
        self.cond_entropy_history = None
        self.listener_value_history = None
        self.gain_history = None

    def init(self):
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        
    def _compute_cond_entropy(self, speaker):
        dm_s = self.dm_s.clone()
        dm_s[dm_s <= ZERO] = ZERO
        prior = self.prior.clone().sum(dim=1).sum(dim=1)
        prior[prior <= ZERO] = ZERO
        speaker_arr = speaker.as_tensor(log=False).clone()
        log_speaker_times_speaker = torch.zeros_like(speaker_arr, dtype=torch.float)
        mask = speaker_arr > 0
        log_speaker_times_speaker[mask] = speaker_arr[mask] * torch.log(speaker_arr[mask])
        log_speaker_times_speaker[~mask] = ZERO # approx 0 * log(0) = 0
        return -torch.einsum('a,a,au->', dm_s, prior, log_speaker_times_speaker)
    
    def _compute_listener_value(self, speaker, listener):
        dm = torch.outer(self.dm_s, self.dm_l)
        dm[dm <= ZERO] = ZERO
        prior = self.prior.clone()
        prior[prior <= ZERO] = ZERO
        speaker_arr = speaker.as_tensor(log=False).clone()
        listener_arr = listener.as_tensor(log=False).clone()
        log_listener = torch.zeros_like(listener_arr, dtype=torch.float)
        mask = listener_arr > 0
        log_listener[mask] = torch.log(listener_arr[mask])
        log_listener[~mask] = -INF
        return torch.einsum('ab,aby,au,uby->', dm, prior, speaker_arr, log_listener)

    def compute_gain(self, listener, speaker):
        cond_ent = self._compute_cond_entropy(speaker)
        self.cond_entropy_history.append(cond_ent)
        listener_value = self._compute_listener_value(speaker, listener)
        self.listener_value_history.append(listener_value)
        gain = cond_ent + self.alpha * listener_value
        self.gain_history.append(gain)
        return gain
    
    def get_diff(self):
        if len(self.gain_history) < 2:
            return float("inf")
        return abs(self.gain_history[-1] - self.gain_history[-2]) / abs(self.gain_history[-2])



class CRSATurn:
    
    def __init__(
        self,
        spk_name,
        lit_spk,
        prior,
        belief_S,
        belief_L,
        costs,
        alpha,
        max_depth,
        tolerance,
    ):
        self.spk_name = spk_name
        self.costs = costs
        self.prior = prior
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.speaker = Speaker(lit_spk, belief_L, prior, costs, alpha)
        self.listener = Listener(lit_spk, belief_S, prior)
        self.gain = CRSAGain(prior, belief_S, belief_L, costs, alpha)

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

    def __init__(self, prior: torch.Tensor):
        self.prior = prior
        self.beliefs = []
        self.turns = []

    def reset(self):
        self.beliefs = []
        self.turns = []

    def sample_utterance(self, meaning_S, sampling_strategy):
        # Get pragmatic speaker of the last turn
        prag_logspk = self.turns[-1].speaker.as_tensor(log=True)

        # Sample an utterance from the pragmatic speaker
        new_utt = sample_utterance(prag_logspk, meaning_S, sampling_strategy)
        prag_spk = torch.exp(prag_logspk)

        # Update the belief history
        spk_name = self.turns[-1].spk_name
        if spk_name == "A" and self.beliefs:
            self.beliefs.append({
                "A": self.beliefs[-1]["A"] * prag_spk[:, new_utt],
                "B": self.beliefs[-1]["B"].clone()
            })
        elif spk_name == "A" and not self.beliefs:
            self.beliefs.append({
                "A": prag_spk[:, new_utt].clone(),
                "B": torch.ones(self.prior.shape[1], dtype=prag_spk.dtype, device=prag_spk.device)
            })
        elif spk_name == "B" and self.beliefs:
            self.beliefs.append({
                "A": self.beliefs[-1]["A"].clone(),
                "B": self.beliefs[-1]["B"] * prag_spk[:, new_utt]
            })
        else:
            self.beliefs.append({
                "A": torch.ones(self.prior.shape[0], dtype=prag_spk.dtype, device=prag_spk.device),
                "B": prag_spk[:, new_utt].clone()
            })

        return new_utt

    def run_turn(self, lit_spk, spk_name, costs, alpha=1.0, max_depth=float('inf'), tolerance=1e-3):

        # Listener name
        lst_name = "B" if spk_name == "A" else "A"
        
        # Prior and beliefs of the turn
        prior = self.prior.clone() if spk_name == "A" else self.prior.clone().transpose(0, 1)
        belief_S = self.beliefs[-1][spk_name] if self.beliefs else None
        belief_L = self.beliefs[-1][lst_name] if self.beliefs else None

        # Init model
        model = CRSATurn(
            spk_name=spk_name,
            lit_spk=lit_spk,
            prior=prior,
            belief_S=belief_S,
            belief_L=belief_L,
            costs=costs,
            alpha=alpha,
            max_depth=max_depth,
            tolerance=tolerance,
        )

        # Run the model and save it
        model.run()
        self.turns.append(model)
        
        prag_spk = model.speaker.as_tensor(log=False)
        prag_list = model.listener.as_tensor(log=False)
        return prag_spk, prag_list

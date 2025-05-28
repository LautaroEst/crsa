
import torch

from .utils import INF, ZERO
from .utils import sample_utterance


class Listener:

    def __init__(self, prior, belief_L=None):
        self.prior = prior
        self.belief_L = belief_L

    def init(self, lit_spk):
        lit_lst = self._update(lit_spk)
        self.history = [lit_lst]

    def update(self, speaker):
        speaker_arr = speaker.as_array.clone()
        speaker_arr[speaker_arr <= ZERO] = ZERO
        prag_lst = self._update(speaker_arr)
        self.history.append(prag_lst)

    def _update(self, spk):

        if self.belief_L is None:
            # lexicon(u,a), prior(a,b,y)
            pragmatic_listener = torch.einsum('au,aby->uby', spk, self.prior)
        else:
            # prior(a,b,y), dm(a,b), speaker(a,u)
            pragmatic_listener = torch.einsum('a,aby,au->uby', self.belief_L, self.prior, spk)

        pragmatic_listener[pragmatic_listener <= ZERO] = ZERO
        norm_term = pragmatic_listener.sum(dim=-1, keepdim=True)
        norm_term[norm_term <= ZERO] = ZERO
        pragmatic_listener = pragmatic_listener / norm_term
        return pragmatic_listener

    @property
    def literal_as_array(self):
        return self.history[0]
    
    @property
    def as_array(self):
        return self.history[-1]
    

class Speaker:

    def __init__(self, prior, belief_S=None, costs=None, alpha=1.0):
        self.prior = prior
        self.belief_S = belief_S
        self.costs = costs
        self.alpha = alpha

    def init(self, lit_spk):
        self.history = [lit_spk]

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
        listener_arr = listener.as_array.clone()
        mask = listener_arr > 0
        log_listener = torch.zeros_like(listener_arr, dtype=listener_arr.dtype, device=listener_arr.device)
        log_listener[mask] = torch.log(listener_arr[mask])
        log_listener[~mask] = -INF

        if self.belief_S is None:
            # Compute the pragmatic speaker
            log_pragmatic_speaker = self.alpha * torch.einsum('aby,uby->au', prior_by_given_a, log_listener) - self.costs.reshape(1,-1)
        else:
            # Compute the quotient of dm
            dm_num = torch.einsum('b,ab->ab', self.belief_S, prior_b_given_a)
            dm_num[dm_num <= ZERO] = ZERO
            dm_frac = dm_num / dm_num.sum(dim=1, keepdim=True)
            dm_frac[dm_frac <= ZERO] = ZERO

            # Compute the pragmatic speaker
            log_pragmatic_speaker = self.alpha * torch.einsum('ab,aby,uby->au', dm_frac, prior_y_given_ab, log_listener) - self.costs.reshape(1,-1)
        
        pragmatic_speaker = torch.softmax(log_pragmatic_speaker, dim=1)
        pragmatic_speaker[pragmatic_speaker <= ZERO] = ZERO
        self.history.append(pragmatic_speaker)
        
    @property
    def as_array(self):
        if self.history:
            return self.history[-1]
    


class CRSAGain:

    def __init__(self, prior, belief_L, belief_S, costs, alpha):
        self.prior = prior
        self.belief_L = belief_L if belief_L is not None else torch.ones(prior.shape[1], dtype=prior.dtype, device=prior.device)
        self.belief_S = belief_S if belief_S is not None else torch.ones(prior.shape[0], dtype=prior.dtype, device=prior.device)
        self.costs = costs
        self.alpha = alpha

    def init(self):
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        
    def _compute_cond_entropy(self, speaker):
        belief_L = self.belief_L.clone()
        belief_L[belief_L <= ZERO] = ZERO
        prior = self.prior.clone().sum(dim=1).sum(dim=1)
        prior[prior <= ZERO] = ZERO
        speaker_arr = speaker.as_array.clone()
        log_speaker_times_speaker = torch.zeros_like(speaker_arr, dtype=speaker_arr.dtype, device=speaker_arr.device)
        mask = speaker_arr > 0
        log_speaker_times_speaker[mask] = speaker_arr[mask] * torch.log(speaker_arr[mask])
        log_speaker_times_speaker[~mask] = ZERO # approx 0 * log(0) = 0
        return -torch.einsum('a,a,au->', belief_L, prior, log_speaker_times_speaker)
    
    def _compute_listener_value(self, speaker, listener):
        dm = torch.outer(self.belief_L, self.belief_S)
        dm[dm <= ZERO] = ZERO
        prior = self.prior.clone()
        prior[prior <= ZERO] = ZERO
        speaker_arr = speaker.as_array.clone()
        listener_arr = listener.as_array.clone()
        log_listener = torch.zeros_like(listener_arr, dtype=listener_arr.dtype, device=listener_arr.device)
        mask = listener_arr > 0
        log_listener[mask] = torch.log(listener_arr[mask])
        log_listener[~mask] = -INF
        return torch.einsum('ab,aby,au,uby->', dm, prior, speaker_arr, log_listener)

    def compute_gain(self, listener, speaker):
        if speaker.as_array is None:
            return torch.nan
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
        self.belief_S = belief_S
        self.belief_L = belief_L
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.listener = Listener(self.prior, self.belief_L)
        self.speaker = Speaker(self.prior, self.belief_S, self.costs, self.alpha)
        self.gain = CRSAGain(self.prior, self.belief_L, self.belief_S, self.costs, self.alpha)

    def run(self, lit_spk):

        # Init agents
        self.listener.init(lit_spk)
        self.speaker.init(lit_spk)
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



class CRSA:

    def __init__(self, logprior):
        self.prior = torch.exp(logprior)

    def reset(self):
        self.beliefs = []
        self.turns = []

    def sample_utterance(self, meaning_S, sampling_strategy):
        # Get pragmatic speaker of the last turn
        prag_logspk = self.turns[-1].speaker.as_array

        # Sample an utterance from the pragmatic speaker
        utt_idx = sample_utterance(prag_logspk, meaning_S, sampling_strategy)

        # Update the belief history
        model = self.turns[-1]
        spk_name = self.turns[-1].spk_name
        if spk_name == "A":
            if not self.beliefs:
                self.beliefs.append({
                    "B": model.speaker.as_array[:, utt_idx].clone(),
                    "A": torch.ones(self.prior.shape[1], dtype=self.prior.dtype, device=self.prior.device),
                })
            else:
                self.beliefs.append({
                    "A": self.beliefs[-1]["A"],
                    "B": self.beliefs[-1]["B"] * model.speaker.as_array[:, utt_idx]
                })
        else:
            if not self.beliefs:
                self.beliefs.append({
                    "A": model.speaker.as_array[:, utt_idx].clone(),
                    "B": torch.ones(self.prior.shape[0], dtype=self.prior.dtype, device=self.prior.device),
                })
            else:
                self.beliefs.append({
                    "A": self.beliefs[-1]["A"] * model.speaker.as_array[:, utt_idx],
                    "B": self.beliefs[-1]["B"],
                })

        return utt_idx
    
    def run_turn(self, lit_logspk, spk_name, costs, alpha=1.0, max_depth=float('inf'), tolerance=1e-3):

        lit_spk = torch.exp(lit_logspk)
        prior = self.prior.clone() if spk_name == "A" else self.prior.clone().transpose(0, 1)

        lst_name = "B" if spk_name == "A" else "A"
        belief_S = self.beliefs[-1][spk_name] if self.beliefs else None
        belief_L = self.beliefs[-1][lst_name] if self.beliefs else None

        model = CRSATurn(
            spk_name=spk_name,
            prior=prior,
            belief_S=belief_S,
            belief_L=belief_L,
            costs=costs,
            alpha=alpha,
            max_depth=max_depth,
            tolerance=tolerance,
        )
        model.run(lit_spk)
        self.turns.append(model)

        return model.speaker.as_array, model.listener.as_array

        
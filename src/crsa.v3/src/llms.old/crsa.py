





from typing import Optional
import numpy as np
import torch
from scipy.special import log_softmax, logsumexp

from ..utils import ZERO, INF, sample_top_p, multinomial_num_samples_1




class Listener:

    def __init__(self, logds, logprior, loglexicon):
        self.logds = logds
        self.logprior = logprior
        self.loglexicon = loglexicon
        
        log_listener = logds[:,np.newaxis,np.newaxis,np.newaxis] + logprior[:,np.newaxis,:,:] + loglexicon.T[:,:,np.newaxis,np.newaxis]
        log_listener = logsumexp(log_listener, axis=0)
        log_listener = log_softmax(log_listener, axis=-1)
        self.log_listener = log_listener

    def update(self, log_speaker):
        log_listener = self.logds[:,np.newaxis,np.newaxis,np.newaxis] + self.logprior[:,np.newaxis,:,:] + log_speaker[:,:,np.newaxis,np.newaxis]
        log_listener = logsumexp(log_listener, axis=0)
        log_listener = log_softmax(log_listener, axis=-1)
        self.log_listener = log_listener


class Speaker:

    def __init__(self, logdsl, logprior, loglexicon, alpha, costs):
        self.logdsl = logdsl
        self.logprior = logprior
        self.loglexicon = loglexicon
        self.alpha = alpha
        self.costs = costs

    def update(self, log_listener):
        logprior_y_given_sl = logprior_y_given_sl - logsumexp(self.logprior, axis=2)[:,:,np.newaxis]
        # listener(u,l,y)
        sumterm = self.logdsl[:,:,np.newaxis,np.newaxis] + logprior_y_given_sl[:,:,:,np.newaxis]
        sumterm = np.exp(sumterm)
        expterm = self.alpha * np.einsum("slyu,uly->su", sumterm, log_listener - self.costs[:,np.newaxis,np.newaxis])
        log_speaker = log_softmax(expterm, axis=1)
        self.log_speaker = log_speaker


class CRSAGain:

    def __init__(self, logprior, logds, logdsl, costs, alpha):
        self.prior = np.exp(logprior)
        self.dm_s = np.exp(logds)
        self.dm = np.exp(logdsl)
        self.costs = costs
        self.alpha = alpha
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        
    def _compute_cond_entropy(self, log_speaker):
        dm_s = self.dm_s.copy()
        dm_s[dm_s <= ZERO] = ZERO
        prior = self.prior.copy().sum(axis=1).sum(axis=1)
        prior[prior <= ZERO] = ZERO
        log_speaker_times_speaker = log_speaker * np.exp(log_speaker)
        return -np.einsum('a,a,au->', dm_s, prior, log_speaker_times_speaker)
    
    def _compute_listener_value(self, log_speaker, log_listener):
        dm = self.dm.copy()
        dm[dm <= ZERO] = ZERO
        prior = self.prior.copy()
        prior[prior <= ZERO] = ZERO
        speaker_arr = np.exp(log_speaker)
        return np.einsum('ab,aby,au,uby->', dm, prior, speaker_arr, log_listener)

    def compute_gain(self, log_listener, log_speaker):
        if log_speaker.as_array is None:
            return np.nan
        cond_ent = self._compute_cond_entropy(log_speaker)
        self.cond_entropy_history.append(cond_ent)
        listener_value = self._compute_listener_value(log_speaker, log_listener)
        self.listener_value_history.append(listener_value)
        gain = cond_ent + self.alpha * listener_value
        self.gain_history.append(gain)
        return gain
    
    def get_diff(self):
        if len(self.gain_history) < 2:
            return float("inf")
        return abs(self.gain_history[-1] - self.gain_history[-2]) / abs(self.gain_history[-2])



class CRSATimeStep:
    
    def __init__(
        self,
        meanings_S,
        meanings_L,
        categories,
        logprior,
        loglexicon,
        logds,
        logdsl,
        alpha,
        costs,
        max_depth,
        tolerance,
    ):
        self.meanings_S = meanings_S
        self.meanings_L = meanings_L
        self.categories = categories
        self.cost = costs
        self.loglexicon = loglexicon
        self.logprior = logprior
        self.logds = logds
        self.logdsl = logdsl
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.speaker = None
        self.listener = None
        self.gain = None

    def run(self):

        # Initialize the speaker and listener
        self.speaker = Speaker(self.logdsl, self.logprior, self.loglexicon, self.alpha, self.cost)
        self.listener = Listener(self.logds, self.logprior, self.loglexicon)
        self.gain = CRSAGain(self.logprior, self.logds, self.logdsl, self.cost, self.alpha)        
        
        # Run the model for the given number of iterations
        i = 0
        while i < self.max_depth:

            # First update the speaker then the listener
            self.speaker.update(self.listener.log_listener)
            self.listener.update(self.speaker.log_speaker)

            # Check for convergence
            gain = self.gain.compute_gain(self.listener, self.speaker)
            if self.gain.get_diff() < self.tolerance:
                break
            i += 1


class CRSA:

    def __init__(self, round_meaning_A, meanings_A, round_meaning_B, meanings_B, categories, prior, llm, alpha=1.0, costs=None, max_depth=np.inf('inf'), tolerance=1e-3):
        self.round_meaning_A = round_meaning_A
        self.meanings_A = meanings_A
        self.round_meaning_B = round_meaning_B
        self.meanings_B = meanings_B
        self.categories = categories
        self.prior = prior
        self.llm = llm

        self.alpha = alpha
        self.costs = costs
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.past_tokens_ids = []
        self.runned_turns = 0
        self.log_da = np.zeros(len(self.meanings_A))
        self.log_db = np.zeros(len(self.meanings_B))

    def run_turn(self, speaker="A", max_tokens=10, top_k: Optional[int] = None, top_p: float = 1.0):
        meaning_S = self.round_meaning_A if speaker == "A" else self.round_meaning_B
        meanings_S = self.meanings_A if speaker == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker == "A" else self.meanings_A
        prior = self.prior.copy() if speaker == "A" else self.prior.copy().transpose(1, 0, 2)
        logprior = np.log(prior + ZERO)

        new_token_id = -1
        tokens_counter = 0
        should_stop = False
        user_token_ids = self.llm.USER_TOKEN_IDS
        len_user_token_ids = len(user_token_ids)
        while tokens_counter < max_tokens and not should_stop:
            tokens_dist = self._get_token_dist_per_meaning(meanings_S)
            logds, logdsl = self._retrieve_dialog_history(speaker, logprior)
            timestep_model = CRSATimeStep(
                meanings_S=meanings_S,
                meanings_L=meanings_L,
                categories=self.categories,
                logprior=logprior,
                loglexicon=tokens_dist,
                logds=logds,
                logdsl=logdsl,
                alpha=self.alpha,
                costs=self.costs,
                max_depth=self.max_depth,
                tolerance=self.tolerance,
            )
            timestep_model.run()
            log_speaker = timestep_model.speaker.log_speaker
            new_token_id = self._next_token(torch.from_numpy(log_speaker[meaning_S,:]), top_k=top_k, top_p=top_p)
            tokens_counter += 1
            
            self._update_dialog_history(log_speaker[:,new_token_id], speaker)
            self.past_tokens_ids["token_id"].append(new_token_id)
            self.past_tokens_ids["speaker"].append(speaker)
            self.past_tokens_ids["turn"].append(self.runned_turns + 1)

            if self.past_tokens_ids["token_id"][-len_user_token_ids:] == user_token_ids:
                should_stop = True
                self.past_tokens_ids["token_id"] = self.past_tokens_ids["token_id"][:-len_user_token_ids]
                self.past_tokens_ids["speaker"] = self.past_tokens_ids["speaker"][:-len_user_token_ids]
                self.past_tokens_ids["turn"] = self.past_tokens_ids["turn"][:-len_user_token_ids]

        self.runned_turns += 1

    def _next_token(self, logits, top_k: Optional[int] = None, top_p: float = 1.0):
        if top_p < 0.0 or top_p > 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, i = torch.topk(logits, min(top_k, logits.size(-1)))
            # do not use `torch.where` as in nanogpt because it will repeat top-k collisions
            logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
        # optionally scale the logits and sample from a probability distribution
        if top_p > 0.0:
            # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
            if top_p < 1.0:
                logits = sample_top_p(logits, top_p)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return multinomial_num_samples_1(probs)
        return torch.argmax(logits, dim=-1, keepdim=True).to(device="cpu", dtype=torch.long).item()

    def _get_token_dist_per_meaning(self, meanings_S):
        prompts = self.llm.generate_prompt(self.past_tokens_ids, meanings_S)
        tokens_dists = []
        for prompt in prompts:
            token_dist = self.llm(prompt).to(device="cpu", dtype=torch.float32).numpy()
            tokens_dists.append(token_dist)
        tokens_dists = np.vstack(tokens_dists)
        return tokens_dists
    
    def _retrieve_dialog_history(self, speaker, logprior):
        log_ds = self.log_da if speaker == "A" else self.log_db
        log_dl = self.log_db if speaker == "A" else self.log_da
        
        logprior_sl = logsumexp(logprior, axis=2)
        logprior_s = logsumexp(logprior_sl, axis=1, keepdims=True)
        logprior_l_given_s = logprior_sl - logprior_s
        log_dsl = log_softmax(log_dl.reshape(1,-1) + logprior_l_given_s, dim=1)

        return log_ds, log_dsl
    
    def _update_dialog_history(self, log_speaker, speaker):
        if speaker == "A":
            self.log_da += log_speaker
        else:
            self.log_db += log_speaker


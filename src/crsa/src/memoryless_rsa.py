
import pandas as pd
import numpy as np
from scipy.special import softmax

from ..src.utils import (
    ZERO, INF
)


class Listener:

    def __init__(self, meanings, utterances, categories, prior, lexicon):
        self.meanings = meanings
        self.utterances = utterances
        self.categories = categories
        self.prior = prior
        self.lexicon = lexicon
        self.history = []

    def update(self, speaker):
        speaker_arr = speaker.as_array.copy()
        speaker_arr[speaker_arr <= ZERO] = ZERO

        # lexicon(u,a), prior(a,b,y)
        pragmatic_listener = np.einsum('au,aby->uby', speaker_arr, self.prior)

        pragmatic_listener[pragmatic_listener <= ZERO] = ZERO
        norm_term = pragmatic_listener.sum(axis=-1, keepdims=True)
        norm_term[norm_term <= ZERO] = ZERO
        pragmatic_listener = pragmatic_listener / norm_term
        self.history.append(pragmatic_listener)

    def compute_literal(self):
        literal_listener = np.einsum('ua,aby->uby', self.lexicon, self.prior)
        literal_listener[literal_listener <= ZERO] = ZERO
        norm_term = literal_listener.sum(axis=-1, keepdims=True)
        norm_term[norm_term <= ZERO] = ZERO
        literal_listener = literal_listener / norm_term
        self.history.append(literal_listener)

    @property
    def as_array(self):
        return self.history[-1]
    
    @property
    def as_df(self):
        return pd.DataFrame(
            self.history[-1].reshape(-1, len(self.categories)),
            index=pd.MultiIndex.from_product([self.utterances, self.meanings], names=["utterance", "meaning"]), 
            columns=self.categories
        )
    
class Speaker:

    def __init__(self, meanings, utterances, prior, lexicon, alpha, costs):
        self.meanings = meanings
        self.utterances = utterances
        self.prior = prior
        self.lexicon = lexicon
        self.alpha = alpha
        self.costs = costs
        self.history = []

    def update(self, listener):
        # Compute the conditional priors 
        prior = self.prior.copy()
        prior[prior <= ZERO] = ZERO
        prior_ab = prior.sum(axis=2) # P(a,b)
        prior_a = prior_ab.sum(axis=1, keepdims=True) # P(a)
        prior_ab[prior_ab <= ZERO] = ZERO
        prior_a[prior_a <= ZERO] = ZERO
        prior_by_given_a = prior / prior_a # P(b,y|a)

        # Compute the log_listener
        listener_arr = listener.as_array.copy()
        mask = listener_arr > 0
        log_listener = np.zeros_like(listener_arr, dtype=float)
        log_listener[mask] = np.log(listener_arr[mask])
        log_listener[~mask] = -INF

        # Compute the pragmatic speaker
        log_pragmatic_speaker = self.alpha * np.einsum('aby,uby->au', prior_by_given_a, log_listener) - self.costs.reshape(1,-1)
        pragmatic_speaker = softmax(log_pragmatic_speaker, axis=1)
        pragmatic_speaker[pragmatic_speaker <= ZERO] = ZERO
        self.history.append(pragmatic_speaker)


    def compute_literal(self):
        # Compute the conditional priors 
        prior = self.prior.copy()
        prior[prior <= ZERO] = ZERO
        prior_ab = prior.sum(axis=2) # P(a,b)
        prior_a = prior_ab.sum(axis=1, keepdims=True) # P(a)
        prior_ab[prior_ab <= ZERO] = ZERO
        prior_a[prior_a <= ZERO] = ZERO
        prior_b_given_a = prior_ab / prior_a # P(b|a)

        mask = self.lexicon > 0
        log_lexicon = np.zeros_like(self.lexicon, dtype=float)
        log_lexicon[mask] = np.log(self.lexicon[mask])
        log_lexicon[~mask] = -INF
        literal_speaker = softmax(self.alpha * np.einsum("ub,ab->au",log_lexicon - self.costs.reshape(-1,1), prior_b_given_a), axis=1)
        self.history.append(literal_speaker)        
        
    @property
    def as_array(self):
        return self.history[-1]
    
    @property
    def as_df(self):
        return pd.DataFrame(
            self.history[-1], 
            index=pd.Index(self.meanings, name="meaning"), 
            columns=self.utterances
        )
    

class RSAGain:

    def __init__(self, prior, cost, alpha):
        self.prior = prior
        self.cost = cost
        self.alpha = alpha
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        
    def _compute_cond_entropy(self, speaker):
        prior = self.prior.copy().sum(axis=1).sum(axis=1)
        prior[prior <= ZERO] = ZERO
        speaker_arr = speaker.as_array.copy()
        log_speaker_times_speaker = np.zeros_like(speaker_arr, dtype=float)
        mask = speaker_arr > 0
        log_speaker_times_speaker[mask] = speaker_arr[mask] * np.log(speaker_arr[mask])
        log_speaker_times_speaker[~mask] = ZERO # approx 0 * log(0) = 0
        return -np.einsum('a,au->', prior, log_speaker_times_speaker)
    
    def _compute_listener_value(self, speaker, listener):
        prior = self.prior.copy()
        prior[prior <= ZERO] = ZERO
        speaker_arr = speaker.as_array.copy()
        listener_arr = listener.as_array.copy()
        log_listener = np.zeros_like(listener_arr, dtype=float)
        mask = listener_arr > 0
        log_listener[mask] = np.log(listener_arr[mask])
        log_listener[~mask] = -INF
        return np.einsum('aby,au,uby->', prior, speaker_arr, log_listener)

    def compute_gain(self, listener, speaker):
        if speaker.as_array is None:
            return np.nan
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


class MemorylessRSATurn:

    def __init__(self, meanings_S, meanings_L, categories, utterances, lexicon_S, lexicon_L, prior, alpha=1.0, costs=None, pov="listener", max_depth=100, tolerance=1e-5):
        self.meanings_S = meanings_S
        self.meanings_L = meanings_L
        self.categories = categories
        self.utterances = utterances
        self.lexicon_S = lexicon_S
        self.lexicon_L = lexicon_L
        self.prior = prior
        self.alpha = alpha
        self.costs = costs if costs is not None else np.zeros(len(utterances))
        self.pov = pov
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.speaker = Speaker(meanings_S, utterances, prior, lexicon_L, alpha, costs)
        self.speaker.compute_literal()
        self.listener = Listener(meanings_L, utterances, categories, prior, lexicon_S)
        self.listener.compute_literal()
        self.gain = RSAGain(prior, costs, alpha)

    def run(self):
        # Run the model for the given number of iterations
        i = 0
        while i < self.max_depth:
            if self.pov == "listener":
                # First update the speaker then the listener
                self.speaker.update(self.listener)
                self.listener.update(self.speaker)
            elif self.pov == "speaker":
                # First update the listener then the speaker
                self.listener.update(self.speaker)
                self.speaker.update(self.listener)
            else:
                raise ValueError("pov must be either 'listener' or 'speaker'")

            # Check for convergence
            gain = self.gain.compute_gain(self.listener, self.speaker)
            if self.gain.get_diff() < self.tolerance:
                break
            i += 1



class MemorylessRSA:

    def __init__(self, meanings_A, meanings_B, categories, utterances, lexicon_A, lexicon_B, prior, alpha=1.0, costs=None, pov="listener", max_depth=100, tolerance=1e-5):
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.categories = categories
        self.utterances = utterances
        self.lexicon_A = lexicon_A
        self.lexicon_B = lexicon_B
        self.prior = prior # P(a,b,y)
        self.alpha = alpha
        self.costs = costs if costs is not None else np.zeros(len(utterances))
        self.pov = pov
        self.max_depth = max_depth
        self.tolerance = tolerance
        self.past_utterances = []
        self.speaker_now = None
        self.turns_history = []

    def sample_new_utterance_from_last_speaker(self, meaning_S):
        speaker = self.turns_history[-1].speaker.as_df
        utt_dist = speaker.loc[meaning_S,:].squeeze()
        return utt_dist[utt_dist == utt_dist.max()].sample(n=1).index[0]

    def get_category_dist_from_last_listener(self, new_utt, meaning_L):
        return self.turns_history[-1].listener.as_df.loc[(new_utt, meaning_L),:].values.reshape(-1)

    def run(self, utterances, speaker_now):

        turns_runned = len(self.turns_history)
        self.past_utterances.extend(utterances)
        self.speaker_now = speaker_now

        turns = len(self.past_utterances) + 1
        for turn in range(turns_runned + 1, turns + 1):

            # Determine the speaker and listener
            speaking_agent = self.past_utterances[turn-1]["speaker"] if turn <= len(self.past_utterances) else self.speaker_now
            past_utterances = self.past_utterances[:turn-1]

            # Run for the turn
            model = self._run_turn(past_utterances, speaking_agent)
            self.turns_history.append(model)

    def _run_turn(self, past_utterances, speaker_now):
        prior = self.prior.copy() if speaker_now == "A" else self.prior.copy().transpose(1, 0, 2)
        meanings_S = self.meanings_A if speaker_now == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker_now == "A" else self.meanings_A
        lexicon_S = self.lexicon_A if speaker_now == "A" else self.lexicon_B
        lexicon_L = self.lexicon_B if speaker_now == "A" else self.lexicon_A
        model = MemorylessRSATurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=self.utterances,
            prior=prior,
            lexicon_S=lexicon_S,
            lexicon_L=lexicon_L,
            alpha=self.alpha,
            costs=self.costs,
            pov=self.pov,
            max_depth=self.max_depth,
            tolerance=self.tolerance
        )
        model.run()
        return model

        


    
        


import pandas as pd
import numpy as np
from scipy.special import softmax

from .utils import (
    INF, ZERO,
)


class Listener:

    def __init__(self, categories, meanings, utterances, prior, dm=None):
        self.categories = categories
        self.meanings = meanings
        self.utterances = utterances
        self.prior = prior
        self.dm = dm
        self.history = []

    def compute_literal_listener(self, lexicon):

        if self.dm is None:
            # lexicon(u,a), prior(a,b,y)
            literal_listener = np.einsum('ua,aby->uby', lexicon, self.prior)
        else:
            # lexicon(u,a), prior(a,b,y), dm(a,b)
            literal_listener = np.einsum('a,aby,ua->uby', self.dm, self.prior, lexicon)
        
        literal_listener[literal_listener <= ZERO] = ZERO
        norm_term = literal_listener.sum(axis=-1, keepdims=True)
        norm_term[norm_term <= ZERO] = ZERO
        literal_listener = literal_listener / norm_term
        self.history.append(literal_listener)
    
    def update(self, speaker):
        speaker_arr = speaker.as_array.copy()
        speaker_arr[speaker_arr <= ZERO] = ZERO

        if self.dm is None:
            # lexicon(u,a), prior(a,b,y)
            pragmatic_listener = np.einsum('au,aby->uby', speaker_arr, self.prior)
        else:
            # prior(a,b,y), dm(a,b), speaker(a,u)
            pragmatic_listener = np.einsum('a,aby,au->uby', self.dm, self.prior, speaker_arr)

        pragmatic_listener[pragmatic_listener <= ZERO] = ZERO
        norm_term = pragmatic_listener.sum(axis=-1, keepdims=True)
        norm_term[norm_term <= ZERO] = ZERO
        pragmatic_listener = pragmatic_listener / norm_term
        self.history.append(pragmatic_listener)

    @property
    def literal_listener_as_array(self):
        return self.history[0]
    
    @property
    def literal_listener_as_df(self):
        return pd.DataFrame(self.history[0].reshape(-1,len(self.categories)), index=pd.MultiIndex.from_product([self.utterances, self.meanings], names=["utterance", "meaning"]), columns=self.categories)
        
    @property
    def as_array(self):
        return self.history[-1]
    
    @property
    def as_df(self):
        return pd.DataFrame(self.history[-1].reshape(-1,len(self.categories)), index=pd.MultiIndex.from_product([self.utterances, self.meanings], names=["utterance", "meaning"]), columns=self.categories)


class Speaker:

    def __init__(self, meanings, utterances, prior, dm=None, cost=None, alpha=1.0):
        self.meanings = meanings
        self.utterances = utterances
        self.prior = prior
        self.dm = dm
        self.cost = cost if cost is not None else np.zeros(len(utterances), dtype=float)
        self.alpha = alpha
        self.history = []

    def update(self, listener):

        # Compute the conditional priors 
        prior = self.prior.copy()
        prior[prior <= ZERO] = ZERO
        prior_ab = prior.sum(axis=2) # P(a,b)
        prior_a = prior_ab.sum(axis=1, keepdims=True) # P(a)
        prior_ab[prior_ab <= ZERO] = ZERO
        prior_a[prior_a <= ZERO] = ZERO
        prior_b_given_a = prior_ab / prior_a # P(b|a)
        prior_by_given_a = prior / prior_a # P(b,y|a)
        prior_y_given_ab = prior / prior_ab[:,:,np.newaxis] # P(y|a,b)

        # Compute the log_listener
        listener_arr = listener.as_array.copy()
        mask = listener_arr > 0
        log_listener = np.zeros_like(listener_arr, dtype=float)
        log_listener[mask] = np.log(listener_arr[mask])
        log_listener[~mask] = -INF

        if self.dm is None:
            # Compute the pragmatic speaker
            log_pragmatic_speaker = self.alpha * np.einsum('aby,uby->au', prior_by_given_a, log_listener) - self.cost.reshape(1,-1)
        else:
            # Compute the quotient of dm
            dm_num = np.einsum('b,ab->ab', self.dm, prior_b_given_a)
            dm_num[dm_num <= ZERO] = ZERO
            dm_frac = dm_num / dm_num.sum(axis=1, keepdims=True)
            dm_frac[dm_frac <= ZERO] = ZERO

            # Compute the pragmatic speaker
            log_pragmatic_speaker = self.alpha * np.einsum('ab,aby,uby->au', dm_frac, prior_y_given_ab, log_listener) - self.cost.reshape(1,-1)
        
        pragmatic_speaker = softmax(log_pragmatic_speaker, axis=1)
        pragmatic_speaker[pragmatic_speaker <= ZERO] = ZERO
        self.history.append(pragmatic_speaker)

    @property
    def as_array(self):
        if self.history:
            return self.history[-1]
    
    @property
    def as_df(self):
        return pd.DataFrame(self.history[-1], index=pd.MultiIndex.from_product([self.meanings], names=["meaning"]), columns=self.utterances)


class CRSAGain:

    def __init__(self, prior, dm_s, dm_l, cost, alpha):
        self.prior = prior
        self.dm_s = dm_s if dm_s is not None else np.ones(prior.shape[0])
        self.dm_l = dm_l if dm_l is not None else np.ones(prior.shape[1])
        self.cost = cost
        self.alpha = alpha
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        
    def _compute_cond_entropy(self, speaker):
        dm_s = self.dm_s.copy()
        dm_s[dm_s <= ZERO] = ZERO
        prior = self.prior.copy().sum(axis=1).sum(axis=1)
        prior[prior <= ZERO] = ZERO
        speaker_arr = speaker.as_array.copy()
        log_speaker_times_speaker = np.zeros_like(speaker_arr, dtype=float)
        mask = speaker_arr > 0
        log_speaker_times_speaker[mask] = speaker_arr[mask] * np.log(speaker_arr[mask])
        log_speaker_times_speaker[~mask] = ZERO # approx 0 * log(0) = 0
        return -np.einsum('a,a,au->', dm_s, prior, log_speaker_times_speaker)
    
    def _compute_listener_value(self, speaker, listener):
        dm = np.outer(self.dm_s, self.dm_l)
        dm[dm <= ZERO] = ZERO
        prior = self.prior.copy()
        prior[prior <= ZERO] = ZERO
        speaker_arr = speaker.as_array.copy()
        listener_arr = listener.as_array.copy()
        log_listener = np.zeros_like(listener_arr, dtype=float)
        mask = listener_arr > 0
        log_listener[mask] = np.log(listener_arr[mask])
        log_listener[~mask] = -INF
        return np.einsum('ab,aby,au,uby->', dm, prior, speaker_arr, log_listener)

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



class CRSATurn:
    
    def __init__(
        self,
        meanings_S,
        meanings_L,
        categories,
        utterances,
        prior,
        lexicon,
        ds,
        dl,
        cost,
        alpha,
        max_depth,
        tolerance,
        pov="listener",
    ):
        self.meanings_S = meanings_S
        self.meanings_L = meanings_L
        self.categories = categories
        self.utterances = utterances
        self.cost = cost
        self.lexicon = lexicon
        self.prior = prior
        self.ds = ds
        self.dl = dl
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance
        self.pov = pov

        self.speaker = None
        self.listener = None
        self.gain = None

    def run(self):

        # Init agents
        self.listener = Listener(self.categories, self.meanings_L, self.utterances, self.prior, self.ds)
        self.speaker = Speaker(self.meanings_S, self.utterances, self.prior, self.dl, self.cost, self.alpha)
        self.listener.compute_literal_listener(self.lexicon)
        self.gain = CRSAGain(self.prior, self.ds, self.dl, self.cost, self.alpha)

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



class CRSA:

    def __init__(self, meanings_A, meanings_B, categories, utterances, lexicon_A, lexicon_B, prior, costs=None, alpha=1.0,  pov="listener", max_depth=None, tolerance=1e-6):
        
        # World parameters
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.categories = categories
        self.utterances = utterances
        self.lexicon_A = np.asarray(lexicon_A, dtype=float)
        self.lexicon_B = np.asarray(lexicon_B, dtype=float)
        self.prior = np.asarray(prior, dtype=float)
        self.prior = self.prior / np.sum(self.prior)

        # Pragmatic parameters
        self.costs = np.asarray(costs, dtype=float) if costs is not None else np.zeros(len(utterances))
        self.alpha = alpha
        
        # Iteration parameters
        self.max_depth = max_depth
        self.tolerance = tolerance
        self.pov = pov

        # History
        self.past_utterances = []
        self.speaker_now = None
        self.turns_history = []

    def sample_new_utterance_from_last_speaker(self, meaning_S):
        speaker = self.turns_history[-1].speaker.as_df
        utt_dist = speaker.loc[meaning_S,:].squeeze()
        return utt_dist[utt_dist == utt_dist.max()].sample(n=1).index[0]

    def get_category_dist_from_last_listener(self, new_utt, meaning_L):
        return self.turns_history[-1].listener.as_df.loc[(new_utt, meaning_L),:].values.reshape(-1)

    def run(self, utterances, speaker_now="A"):

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
            
    def _run_turn(self, past_utterances, speaker="A"):
        meanings_S = self.meanings_A if speaker == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker == "A" else self.meanings_A
        lexicon = self.lexicon_A if speaker == "A" else self.lexicon_B
        prior = self.prior.copy() if speaker == "A" else self.prior.copy().transpose(1, 0, 2)
        ds, dl = self._compute_dialog_model(past_utterances, speaker)

        model = CRSATurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=self.utterances,
            prior=prior,
            lexicon=lexicon,
            ds=ds,
            dl=dl,
            cost=self.costs,
            alpha=self.alpha,
            max_depth=self.max_depth,
            tolerance=self.tolerance,
            pov=self.pov,
        )
        model.run()
        return model
    
    def _compute_dialog_model(self, past_utterances, speaker):
        if not past_utterances:
            return None, None
        
        ds, dl = [], []
        for data, model in zip(past_utterances, self.turns_history):
            utt_idx = self.utterances.index(data["utterance"])
            if data["speaker"] == speaker:
                ds.append(model.speaker.as_array[:, utt_idx])
            else:
                dl.append(model.speaker.as_array[:, utt_idx])
        
        ds_arr = np.ones(len(self.meanings_A if speaker == "A" else self.meanings_B))
        for d in ds:
            ds_arr *= d

        dl_arr = np.ones(len(self.meanings_B if speaker == "A" else self.meanings_A))
        for d in dl:
            dl_arr *= d
        
        return ds_arr, dl_arr

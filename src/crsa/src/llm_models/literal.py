
import pandas as pd
import numpy as np
from scipy.special import softmax

from ..naive_models.utils import ZERO, INF


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
        # prior = self.prior.copy()
        # prior[prior <= ZERO] = ZERO
        # prior_ab = prior.sum(axis=2) # P(a,b)
        # prior_a = prior_ab.sum(axis=1, keepdims=True) # P(a)
        # prior_ab[prior_ab <= ZERO] = ZERO
        # prior_a[prior_a <= ZERO] = ZERO
        # prior_b_given_a = prior_ab / prior_a # P(b|a)

        mask = self.lexicon > 0
        log_lexicon = np.zeros_like(self.lexicon, dtype=float)
        log_lexicon[mask] = np.log(self.lexicon[mask])
        log_lexicon[~mask] = -INF
        literal_speaker = softmax(log_lexicon.T, axis=1)
        # literal_speaker = softmax(self.alpha * np.einsum("ub,ab->au",log_lexicon - self.costs.reshape(-1,1), prior_b_given_a), axis=1)
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
    
class LiteralTurn:

    def __init__(self, meanings_S, meanings_L, categories, utterances, lexicon_S, lexicon_L, prior, alpha=1.0, costs=None):
        self.meanings_S = meanings_S
        self.meanings_L = meanings_L
        self.categories = categories
        self.utterances = utterances
        self.lexicon_S = lexicon_S
        self.lexicon_L = lexicon_L
        self.prior = prior
        self.alpha = alpha
        self.costs = costs if costs is not None else np.zeros(len(utterances))

        self.speaker = Speaker(meanings_S, utterances, prior, lexicon_S, alpha, costs)
        self.listener = Listener(meanings_L, utterances, categories, prior, lexicon_L)

    def run(self):
        # Run the model for the given number of iterations
        self.listener.compute_literal()
        self.speaker.compute_literal()



class LLMLiteral:

    def __init__(self, meanings_A, meanings_B, categories, prior, alpha=1.0):
        self.round_meaning_A = None
        self.meanings_A = meanings_A
        self.round_meaning_B = None
        self.meanings_B = meanings_B
        self.categories = categories
        self.prior = prior # P(a,b,y)
        self.alpha = alpha
        self.past_utterances = []
        self.speaker_now = None
        self.turns_history = []
        self.past_speaker_dist = []
        self.past_lexicon_dist = []

    def sample_utterance(self, speaker):
        meaning_S = self.round_meaning_A if speaker == "A" else self.round_meaning_B
        speaker = self.turns_history[-1].speaker.as_df
        utt_dist = speaker.loc[meaning_S,:].squeeze()
        new_utt = utt_dist[utt_dist == utt_dist.max()].index[0]
        return new_utt

    def get_category_distribution(self):
        return np.ones(len(self.categories), dtype=float) / len(self.categories)

    def reset(self, meaning_A, meaning_B):
        self.round_meaning_A = meaning_A
        self.round_meaning_B = meaning_B
        self.past_utterances = []
        self.turns_history = []
        self.past_speaker_dist = []
        self.past_lexicon_dist = []

    def run_turn(self, ground_truth_utt, log_lexicon, utterances, costs=None, speaker="A"):
        if self.round_meaning_A is None or self.round_meaning_B is None:
            raise ValueError("Please set the round meanings before running the model by calling the reset method.")
        meanings_S = self.meanings_A if speaker == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker == "A" else self.meanings_A
        prior = self.prior.copy() if speaker == "A" else self.prior.copy().transpose(1, 0, 2)
        lexicon = np.exp(log_lexicon) / np.sum(np.exp(log_lexicon))

        model = LiteralTurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=utterances,
            prior=prior,
            lexicon_S=lexicon,
            lexicon_L=lexicon,
            alpha=self.alpha,
            costs=costs if costs is not None else np.zeros(len(utterances), dtype=float),
        )
        model.run()
        self.turns_history.append(model)

        # get lexicon distribution
        meaning_S = self.round_meaning_A if speaker == "A" else self.round_meaning_B
        meaning_S_idx = meanings_S.index(meaning_S)
        self.past_lexicon_dist.append(lexicon[:, meaning_S_idx].copy())

        # get speaker distribution
        self.past_utterances.append({"utterance": ground_truth_utt, "speaker": speaker})
        # self.past_speaker_dist.append(self.turns_history[-1].speaker.as_df.loc[meaning_S,:].values.reshape(-1))
        # self.past_speaker_dist.append(np.ones(len(utterances), dtype=float) / len(utterances))
        self.past_speaker_dist.append(lexicon[:, meaning_S_idx].copy())
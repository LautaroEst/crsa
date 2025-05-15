
import pandas as pd
import numpy as np
from scipy.special import softmax

from .utils import ZERO, INF


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



class NaiveLiteral:

    def __init__(self, meanings_A, meanings_B, categories, utterances, lexicon_A, lexicon_B, prior, alpha=1.0, costs=None, update_lexicon=False):
        self.round_meaning_A = None
        self.meanings_A = meanings_A
        self.round_meaning_B = None
        self.meanings_B = meanings_B
        self.categories = categories
        self.utterances = utterances
        self.lexicon_A = lexicon_A
        self.lexicon_B = lexicon_B
        self.prior = prior # P(a,b,y)
        self.alpha = alpha
        self.costs = costs if costs is not None else np.zeros(len(utterances))
        self.past_utterances = []
        self.speaker_now = None
        self.turns_history = []
        self.update_lexicon = update_lexicon

    def reset(self, meaning_A, meaning_B):
        self.round_meaning_A = meaning_A
        self.round_meaning_B = meaning_B
        self.past_utterances = []
        self.turns_history = []

    def _sample_utterance(self, meaning_S, speaker_now):
        speaker = self.turns_history[-1].speaker.as_df
        utt_dist = speaker.loc[meaning_S,:].squeeze()
        new_utt = utt_dist[utt_dist == utt_dist.max()].index[0]
        self.past_utterances.append({"utterance": new_utt, "speaker": speaker_now})

    def get_category_distribution(self):
        if not self.turns_history:
            raise ValueError("No turns have been run yet.")
        listener = self.turns_history[-1].listener.as_df
        meaning_L = self.round_meaning_B if self.past_utterances[-1]["speaker"] == "A" else self.round_meaning_A
        return listener.loc[(self.past_utterances[-1]["utterance"], meaning_L),:].values.reshape(-1)

    def reset(self, meaning_A, meaning_B):
        self.round_meaning_A = meaning_A
        self.round_meaning_B = meaning_B
        self.past_utterances = []
        self.turns_history = []

    def run_turn(self, speaker="A"):
        if self.round_meaning_A is None or self.round_meaning_B is None:
            raise ValueError("Please set the round meanings before running the model by calling the reset method.")
        meanings_S = self.meanings_A if speaker == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker == "A" else self.meanings_A
        lexicon_S = self.lexicon_A if speaker == "A" else self.lexicon_B
        lexicon_L = self.lexicon_B if speaker == "A" else self.lexicon_A
        prior = self.prior.copy() if speaker == "A" else self.prior.copy().transpose(1, 0, 2)
        
        for lexicon in [lexicon_S, lexicon_L]:
            if self.update_lexicon:
                past_utt_idx = [self.utterances.index(utt["utterance"]) for utt in self.past_utterances[:-1]]
                for utt in self.utterances:
                    utt_idx = self.utterances.index(utt)
                    if utt_idx in past_utt_idx and utt != self.past_utterances[-1]["utterance"]:
                        lexicon[utt_idx,:] = 0

        model = LiteralTurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=self.utterances,
            prior=prior,
            lexicon_S=lexicon_S,
            lexicon_L=lexicon_L,
            alpha=self.alpha,
            costs=self.costs,
        )
        model.run()
        self.turns_history.append(model)

        # Sample the utterance
        meaning_S = self.round_meaning_A if speaker == "A" else self.round_meaning_B
        self._sample_utterance(meaning_S, speaker)
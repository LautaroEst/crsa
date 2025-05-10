
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

    def run(self):
        literal_listener = np.einsum('ua,aby->uby', self.lexicon, self.prior)
        literal_listener[literal_listener <= ZERO] = ZERO
        norm_term = literal_listener.sum(axis=-1, keepdims=True)
        norm_term[norm_term <= ZERO] = ZERO
        self._listener = literal_listener / norm_term

    @property
    def as_array(self):
        return self._listener
    
    @property
    def as_df(self):
        return pd.DataFrame(
            self._listener.reshape(-1, len(self.categories)),
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

    def run(self):

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
        self._speaker = softmax(self.alpha * np.einsum("ub,ab->au",log_lexicon - self.costs.reshape(-1,1), prior_b_given_a), axis=1)
        
    @property
    def as_array(self):
        return self._speaker
    
    @property
    def as_df(self):
        return pd.DataFrame(
            self._speaker, 
            index=pd.Index(self.meanings, name="meaning"), 
            columns=self.utterances
        )
    

class MemorylessLiteralTurn:

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

        self.speaker = Speaker(meanings_S, utterances, prior, lexicon_L, alpha, costs)
        self.listener = Listener(meanings_L, utterances, categories, prior, lexicon_S)

    def run(self):
        self.speaker.run()
        self.listener.run()



class MemorylessLiteral:

    def __init__(self, meanings_A, meanings_B, categories, utterances, lexicon_A, lexicon_B, prior, alpha=1.0, costs=None):
        self.meanings_A = meanings_A
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
        model = MemorylessLiteralTurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=self.utterances,
            prior=prior,
            lexicon_S=lexicon_S,
            lexicon_L=lexicon_L,
            alpha=self.alpha,
            costs=self.costs
        )
        model.run()
        return model

        


    
        

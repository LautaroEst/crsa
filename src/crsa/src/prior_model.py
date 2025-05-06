
import pandas as pd
import numpy as np


class Listener:

    def __init__(self, meanings, utterances, categories, prior):
        self.meanings = meanings
        self.utterances = utterances
        self.categories = categories
        self.prior = prior

    def run(self):
        self._listener = self.prior.sum(axis=0)
        self._listener = np.tile(self._listener, (len(self.utterances), 1, 1))

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

    def __init__(self, meanings, utterances, costs):
        self.meanings = meanings
        self.utterances = utterances
        self.costs = costs

    def run(self):
        self._speaker = np.exp(-self.costs)
        self._speaker = self._speaker / np.sum(self._speaker)
        self._speaker = np.tile(self._speaker, (len(self.meanings), 1))


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
    

class PriorModelTurn:

    def __init__(self, meanings_L, meanings_S, categories, utterances, prior, costs=None):
        self.meanings_L = meanings_L
        self.meanings_S = meanings_S
        self.categories = categories
        self.utterances = utterances
        self.prior = prior

        self.speaker = Speaker(meanings_S, utterances, costs)
        self.listener = Listener(meanings_L, utterances, categories, prior)

    def run(self):
        self.speaker.run()
        self.listener.run()



class PriorModel:

    def __init__(self, meanings_A, meanings_B, categories, utterances, prior, costs=None):
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.categories = categories
        self.utterances = utterances
        self.prior = prior # P(a,b,y)
        self.costs = costs if costs is not None else np.zeros(len(utterances))
        self.past_utterances = []
        self.speaker_now = None
        self.turns_history = []

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
        model = PriorModelTurn(
            meanings_L=meanings_L,
            meanings_S=meanings_S,
            categories=self.categories,
            utterances=self.utterances,
            prior=prior,
            costs=self.costs
        )
        model.run()
        return model

        


    
        

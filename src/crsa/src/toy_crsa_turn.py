

from pathlib import Path
import pickle
import numpy as np
import yaml

from .utils import save_yaml


class Listener:
    def __init__(self, categories, meanings, utterances, past_utterances, prior):
        self.categories = categories
        self.meanings = meanings
        self.utterances = utterances
        self.past_utterances = past_utterances
        self.prior = prior
        self.history = []

class Speaker:
    def __init__(self, meanings, utterances, past_utterances, cost):
        self.meanings = meanings
        self.utterances = utterances
        self.past_utterances = past_utterances
        self.cost = cost
        self.history = []

class CRSAGain:
    def __init__(self):
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        self.coop_index_history = []



class ToyCRSATurn:
    
    def __init__(
        self,
        meanings_S,
        meanings_L,
        categories,
        utterances,
        prior,
        lexicon,
        cost,
        alpha,
        max_depth,
        tolerance,
    ):
        self.meanings_S = meanings_S
        self.meanings_L = meanings_L
        self.categories = categories
        self.utterances = utterances
        self.cost = cost
        self.lexicon = lexicon
        self.prior = prior
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.speaker = None
        self.listener = None
        self.gain = None

    def run(self, dialog_model, output_dir, verbose=False, prefix=""):
        self.speaker = Speaker(self.meanings_S, self.utterances, dialog_model.past_utterances, self.cost)
        self.listener = Listener(self.categories, self.meanings_L, self.utterances, dialog_model.past_utterances, self.prior)
        self.gain = CRSAGain()


    @property
    def history(self):
        return {
            "listener": self.listener.history,
            "speaker": self.speaker.history,
            "cond_entropy": self.gain.cond_entropy_history,
            "listener_value": self.gain.listener_value_history,
            "gain": self.gain.gain_history,
        }
    
    @history.setter
    def history(self, value):
        raise AttributeError("history is a read-only property")
    
    def save(self, output_dir: Path, prefix: str = ""):
        args = {
            "meanings_S": self.meanings_S,
            "meanings_L": self.meanings_L,
            "categories": self.categories,
            "utterances": self.utterances,
            "prior": self.prior.tolist(),
            "lexicon": self.lexicon.tolist(),
            "cost": self.cost.tolist(),
            "alpha": self.alpha,
            "max_depth": self.max_depth,
            "tolerance": self.tolerance
        }
        save_yaml(args, output_dir / f"{prefix}args.yaml")

        with open(output_dir / f"{prefix}history.pkl", "wb") as f:
            pickle.dump({
                "listeners": self.listener.history,
                "speakers": self.speaker.history,
                "cond_entropy": np.asarray(self.gain.cond_entropy_history),
                "listener_value": np.asarray(self.gain.listener_value_history),
                "gain": np.asarray(self.gain.gain_history),
            }, f)

    @classmethod
    def load(cls, output_dir: Path, prefix: str = ""):
        with open(output_dir / "args.yaml", "r") as f:
            args = yaml.safe_load(f)
        with open(output_dir / f"{prefix}history.pkl", "rb") as f:
            history = pickle.load(f)
        model = cls(**args)
        model.listener = Listener(model.categories, model.meanings, model.utterances, model.lexicon.past_utterances, model.prior, model.lexicon)
        model.listener.history = history["listeners"]
        model.speaker = Speaker(model.meanings, model.utterances, model.lexicon.past_utterances, model.prior, model.cost, model.alpha)
        model.speaker.history = history["speakers"]
        model.gain = CRSAGain()
        model.gain.cond_entropy_history = history["cond_entropy"].tolist()
        model.gain.listener_value_history = history["listener_value"].tolist()
        model.gain.gain_history = history["gain"].tolist()
        return model


import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax

from .utils import (
    save_yaml,
    INF, ZERO,
)


class Listener:

    def __init__(self, categories, meanings, utterances, prior):
        self.categories = categories
        self.meanings = meanings
        self.utterances = utterances
        self.prior = prior
        self.history = []

    def compute_literal_listener(self, lexicon):

        # lexicon(u,a), prior(a,b,y)
        literal_listener = np.einsum('ua,aby->uby', lexicon, self.prior)
        norm_term = literal_listener.sum(axis=-1, keepdims=True)
        norm_term[norm_term <= ZERO] = ZERO
        literal_listener = literal_listener / norm_term
        self.history.append(literal_listener)
    
    @property
    def as_array(self):
        return self.history[-1]
    
    @property
    def as_df(self):
        return pd.DataFrame(self.history[-1].reshape(-1,len(self.categories)), index=pd.MultiIndex.from_product([self.utterances, self.meanings], names=["utterance", "meaning"]), columns=self.categories)


class Speaker:

    def __init__(self, meanings, utterances, prior, cost=None, alpha=1.0):
        self.meanings = meanings
        self.utterances = utterances
        self.prior = prior
        self.cost = cost if cost is not None else np.zeros(len(utterances), dtype=float)
        self.alpha = alpha
        self.history = []

    def compute_literal_speaker(self, lexicon):

        # Compute the log_lexicon
        mask = lexicon > 0
        log_lexicon = np.zeros_like(lexicon, dtype=float)
        log_lexicon[mask] = np.log(lexicon[mask])
        log_lexicon[~mask] = -INF

        # Compute the literal speaker
        literal_speaker = softmax(self.alpha * log_lexicon.T - self.cost.reshape(1,-1), axis=1)
        self.history.append(literal_speaker)

    @property
    def as_array(self):
        return self.history[-1]
    
    @property
    def as_df(self):
        return pd.DataFrame(self.history[-1], index=pd.MultiIndex.from_product([self.meanings], names=["meaning"]), columns=self.utterances)



class LiteralTurn:
    
    def __init__(
        self,
        meanings_S,
        meanings_L,
        categories,
        utterances,
        prior,
        lexicon,
        alpha=1.0,
        cost=None,
    ):
        self.meanings_S = meanings_S
        self.meanings_L = meanings_L
        self.categories = categories
        self.utterances = utterances
        self.lexicon = lexicon
        self.prior = prior
        self.alpha = alpha
        self.cost = cost if cost is not None else np.zeros(len(utterances), dtype=float)

        self.speaker = None
        self.listener = None
        self.gain = None

    def run(self, output_dir=None, verbose=False, prefix=""):

        # Configure logging
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter("%(message)s")
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        if output_dir is not None:
            file_handler = logging.FileHandler(output_dir / f"{prefix}history.log", mode="w", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Log configuration
        logger.info(f"Running one turn of the Literal model.")
        logger.info("-" * 40)
        logger.info(
            f"\nLexicon:\n\n{self.lexicon}\n\n"
            f"Cost:\n\n{pd.Series(self.cost, index=self.utterances).to_string()}\n\n"
            f"Alpha: {self.alpha}\n"
        )
        logger.info("-" * 40 + "\n" + "-" * 40 + "\n")
        
        # Init agents and compute literals
        self.listener = Listener(self.categories, self.meanings_L, self.utterances, self.prior)
        self.listener.compute_literal_listener(self.lexicon)
        logger.info(f"Literal listener:\n\n{self.listener.as_df}\n\n")

        self.speaker = Speaker(self.meanings_S, self.utterances, self.prior, self.cost, self.alpha)
        self.speaker.compute_literal_speaker(self.lexicon)
        logger.info(f"Literal speaker:\n\n{self.speaker.as_df}\n\n")

        logger.info("-" * 40 + "\n" + "-" * 40 + "\n")

        # Close logging
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)


    def save(self, output_dir: Path, prefix: str = ""):
        args = {
            "meanings_S": self.meanings_S,
            "meanings_L": self.meanings_L,
            "categories": self.categories,
            "utterances": self.utterances,
            "prior": self.prior.tolist(),
            "lexicon": self.lexicon.tolist(),
            "past_utterances": self.past_utterances,
        }
        save_yaml(args, output_dir / f"{prefix}args.yaml")

        with open(output_dir / f"{prefix}history.pkl", "wb") as f:
            pickle.dump({
                "listener": self.listener.history,
                "speaker": self.speaker.history,
                "gain": self.gain.gain_history,
                "cond_entropy": self.gain.cond_entropy_history,
                "listener_value": self.gain.listener_value_history,
            }, f)

    @classmethod
    def load(cls, output_dir: Path, prefix: str = ""):
        with open(output_dir / f"{prefix}args.yaml", "r") as f:
            args = yaml.safe_load(f)
        with open(output_dir / f"{prefix}history.pkl", "rb") as f:
            history = pickle.load(f)
        model = cls(**args)
        model.listener = Listener(model.categories, model.meanings_L, model.utterances, model.prior)
        model.listener.history = history["listener"]
        model.speaker = Speaker(model.meanings_S, model.utterances, model.prior, model.cost, model.alpha)
        model.speaker.history = history["speaker"]
        return model
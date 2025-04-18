

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
        self.past_utterances = None

    def compute_literal_listener(self, dialog_model, lexicon):

        if dialog_model.past_utterances:
            # dialog_model(a,b,w) = Ps(w|a,b), Lex(u,a) = Lex(u,a), prior(a,b,y) = P(a,b,y)
            literal_listener = np.einsum("abw,ua,aby->uwby", dialog_model.as_array, lexicon, self.prior)
        else:
            # dialog_model = None, Lex(u,a) = Lex(u,a), prior(a,b,y) = P(a,b,y)
            literal_listener = np.einsum("ua,aby->uby", lexicon, self.prior)
            literal_listener = literal_listener[:,np.newaxis,...]
        literal_listener /= np.sum(literal_listener, axis=-1, keepdims=True)
        self.history.append(literal_listener)
        self.past_utterances = dialog_model.past_utterances
    
    def update(self, speaker, dialog_model):
        if dialog_model.past_utterances:
            # dialog_model(a,b,w) = Ps(w|a,b), prior(a,b,y) = P(a,b,y), speaker(a,w,u) = S(u|a,w)
            pragmatic_listener = np.einsum("abw,aby,awu->uwby", dialog_model.as_array, self.prior, speaker)
        else:
            # dialog_model = None, prior(a,b,y) = P(a,b,y), speaker(a,w,u) = S(u|a,w)
            pragmatic_listener = np.einsum("aby,awu->uwby", self.prior, speaker.as_array)
        pragmatic_listener /= np.sum(pragmatic_listener, axis=-1, keepdims=True)
        self.history.append(pragmatic_listener)
        self.past_utterances = dialog_model.past_utterances

    @property
    def as_array(self):
        return self.history[-1]
    
    @property
    def as_df(self):
        if self.past_utterances is None:
            return
        
        if self.past_utterances:
            index = pd.MultiIndex.from_product([self.utterances, self.past_utterances, self.meanings], names=["utterance", "past_utterance", "meaning"])
        else:
            index = pd.MultiIndex.from_product([self.utterances, self.meanings], names=["utterance", "meaning"])
        
        return pd.DataFrame(self.history[-1].reshape(-1,len(self.categories)), index=index, columns=self.categories)


class Speaker:

    def __init__(self, meanings, utterances, cost, alpha):
        self.meanings = meanings
        self.utterances = utterances
        self.cost = cost
        self.alpha = alpha
        self.history = []
        self.past_utterances = None

    def update(self, listener, dialog_model, prior):

        # Log listener with -inf for zero probabilities
        l = listener.as_array
        mask = l > 0
        log_listener = np.zeros_like(l)
        log_listener[mask] = np.log(l[mask])
        log_listener[~mask] = -INF

        prior = prior.copy()
        prior[prior == 0] = ZERO
        cond_prior = prior / prior.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True) # Cond(a,b,y) = P(b,y|a) = P(a,b,y) / P(a)
        cond_prior[cond_prior == 0] = ZERO

        if dialog_model.past_utterances:
            dm = dialog_model.as_array       
            dm_num = dm.copy()
            dm_num[dm_num == 0] = ZERO
            dm_den = np.einsum("abw,aby->aw", dm, cond_prior)[:,np.newaxis,np.newaxis,:]
            dm_den[dm_den == 0] = ZERO
            dm_frac = dm_num / dm_den
            exp_term = np.einsum("abyw,aby,uwby->awu", dm_frac, cond_prior, log_listener)
        else:
            exp_term = np.einsum("aby,uwby->awu", cond_prior, log_listener)
        pragmatic_speaker = softmax(self.alpha * (exp_term - self.cost.reshape(1,1,-1)), axis=-1)
        self.history.append(pragmatic_speaker)
        self.past_utterances = dialog_model.past_utterances

    @property
    def as_array(self):
        return self.history[-1]
    
    @property
    def as_df(self):
        if self.past_utterances is None:
            return
        
        if self.past_utterances:
            index = pd.MultiIndex.from_product([self.meanings, self.past_utterances], names=["meaning", "past_utterance"])
        else:
            index = pd.MultiIndex.from_product([self.meanings], names=["meaning"])
        
        return pd.DataFrame(self.history[-1].reshape(-1,len(self.utterances)), index=index, columns=self.utterances)


class CRSAGain:

    def __init__(self):
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []

    def compute_gain(self, dialog_model, listener, speaker, prior, cost, alpha):
        # Placeholder for gain computation logic
        return 0.0
    
    def get_diff(self):
        if len(self.gain_history) < 2:
            return float("inf")
        return abs(self.gain_history[-1] - self.gain_history[-2]) / abs(self.gain_history[-2])



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
        # Configure logging
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter("%(message)s")
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        file_handler = logging.FileHandler(output_dir / f"{prefix}history.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Log configuration
        logger.info(f"Running one turn of the CRSA model for max depth {self.max_depth} and tolerance {self.tolerance:.2e}")
        logger.info("-" * 40)
        logger.info(
            f"\nLexicon:\n\n{self.lexicon}\n\n"
            f"Cost:\n\n{pd.Series(self.cost, index=self.utterances).to_string()}\n\n"
            f"Alpha: {self.alpha}\n"
        )
        logger.info("-" * 40 + "\n")
        
        self.speaker = Speaker(self.meanings_S, self.utterances, self.cost, self.alpha)
        self.listener = Listener(self.categories, self.meanings_L, self.utterances, self.prior)
        self.listener.compute_literal_listener(dialog_model, self.lexicon)
        self.gain = CRSAGain()
        gain = self.gain.compute_gain(dialog_model, self.listener, self.speaker, self.prior, self.cost, self.alpha)
        logger.info(f"Literal listener:\n\n{self.listener.as_df}\n\n")
        logger.info(f"Initial gain: {gain:.4f}\n")
        logger.info("-" * 40 + "\n")

        # Run the model for the given number of iterations
        for i in range(self.max_depth):
            # Update speaker
            self.speaker.update(self.listener, dialog_model, self.prior)
            logger.info(f"Pragmatic speaker at step {i+1}:\n{self.speaker.as_df}\n")
            
            # Update listener
            self.listener.update(self.speaker, dialog_model)
            logger.info(f"Pragmatic listener at step {i+1}:\n{self.listener.as_df}\n")

            # Check for convergence
            gain = self.gain.compute_gain(dialog_model, self.listener, self.speaker, self.prior, self.cost, self.alpha)
            logger.info(f"Step: {i+1} | Gain: {gain:.4f}")
            logger.info("\n" + "-" * 40 + "\n")
            if self.gain.get_diff() < self.tolerance:
                logger.info(f"Converged after {i+1} iterations")
                break

        # Close logging
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)


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
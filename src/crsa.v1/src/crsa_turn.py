

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
        past_utterances,
        ds,
        dl,
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
        self.past_utterances = past_utterances
        self.ds = ds
        self.dl = dl
        self.alpha = alpha
        self.max_depth = max_depth
        self.tolerance = tolerance

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
        logger.info(f"Running one turn of the CRSA model for max depth {self.max_depth} and tolerance {self.tolerance:.2e}")
        logger.info("-" * 40)
        logger.info(
            f"\nLexicon:\n\n{self.lexicon}\n\n"
            f"Cost:\n\n{pd.Series(self.cost, index=self.utterances).to_string()}\n\n"
            f"Alpha: {self.alpha}\n"
        )
        logger.info("-" * 40 + "\n" + "-" * 40 + "\n")
        
        logger.info(f"Past utterances: {self.past_utterances}\n")
        logger.info(f"Prod Ps(u|w,x_s): {self.ds}\n")
        logger.info(f"Prod Ps(u|w,x_l): {self.dl}\n")
    
        # Init agents
        self.listener = Listener(self.categories, self.meanings_L, self.utterances, self.prior, self.ds)
        self.speaker = Speaker(self.meanings_S, self.utterances, self.prior, self.dl, self.cost, self.alpha)
        self.listener.compute_literal_listener(self.lexicon)
        self.gain = CRSAGain(self.prior, self.ds, self.dl, self.cost, self.alpha)
        logger.info(f"Literal listener:\n\n{self.listener.as_df}\n\n")
        logger.info("-" * 40 + "\n")

        # Run the model for the given number of iterations
        i = 0
        while i < self.max_depth:
            # Update speaker
            self.speaker.update(self.listener)
            logger.info(f"Pragmatic speaker at step {i+1}:\n{self.speaker.as_df}\n")
            
            # Update listener
            self.listener.update(self.speaker)
            logger.info(f"Pragmatic listener at step {i+1}:\n{self.listener.as_df}\n")

            # Check for convergence
            gain = self.gain.compute_gain(self.listener, self.speaker)
            logger.info(f"Step: {i+1} | Gain: {gain:.4f}")
            logger.info("\n" + "-" * 40 + "\n")
            if self.gain.get_diff() < self.tolerance:
                logger.info(f"Converged after {i+1} iterations")
                break
            i += 1

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
            "ds": self.ds.tolist() if self.ds is not None else None,
            "dl": self.dl.tolist() if self.dl is not None else None,
            "cost": self.cost.tolist(),
            "alpha": self.alpha,
            "max_depth": self.max_depth,
            "tolerance": self.tolerance
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
        args["prior"] = np.asarray(args["prior"])
        args["prior"] = args["prior"] / args["prior"].sum()
        args["lexicon"] = np.asarray(args["lexicon"])
        args["ds"] = np.asarray(args["ds"]) if args["ds"] is not None else None
        args["dl"] = np.asarray(args["dl"]) if args["dl"] is not None else None
        args["cost"] = np.asarray(args["cost"])
        model = cls(**args)
        model.listener = Listener(model.categories, model.meanings_L, model.utterances, model.prior, args["ds"])
        model.listener.history = history["listener"]
        model.speaker = Speaker(model.meanings_S, model.utterances, model.prior, args["dl"], model.cost, model.alpha)
        model.speaker.history = history["speaker"]
        model.gain = CRSAGain(model.prior, args["ds"], args["dl"], model.cost, model.alpha)
        model.gain.cond_entropy_history = history["cond_entropy"]
        model.gain.listener_value_history = history["listener_value"]
        model.gain.gain_history = history["gain"]
        return model
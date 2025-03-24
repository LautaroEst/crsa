
from pathlib import Path
import numpy as np
from scipy.special import softmax
import logging
import pandas as pd
import yaml
import pickle

from ..src.utils import (
    is_list_of_strings, 
    is_numeric_ndarray, 
    is_list_of_numbers, 
    is_positive_number, 
    is_positive_integer, 
    is_list_of_list_of_numbers,
    save_yaml
)


class Listener:

    def __init__(self, meanings, utterances, prior, lexicon):
        self.meanings = meanings
        self.utterances = utterances
        self.prior = prior
        self.lexicon = lexicon

        # Initialize with literal listener        
        literal_listener = lexicon * prior
        literal_listener /= literal_listener.sum(axis=1, keepdims=True)
        self.history = [literal_listener]

    def update(self, speaker):
        """
        Update the listener based on the speaker
        """
        pragmatic_listener = speaker * self.prior.reshape(-1, 1)
        pragmatic_listener = pragmatic_listener / pragmatic_listener.sum(axis=0)
        self.history.append(pragmatic_listener.T)

    def get_literal_as_df(self):
        return pd.DataFrame(self.history[0], index=self.utterances, columns=self.meanings)
    
    def get_literal_as_array(self):
        return self.history[0]
      
    @property
    def as_df(self):
        return pd.DataFrame(self.as_array, index=self.utterances, columns=self.meanings)
    
    @property
    def as_array(self):
        return self.history[-1]
    

class Speaker:

    def __init__(self, meanings, utterances, cost, alpha):
        self.meanings = meanings
        self.utterances = utterances
        self.cost = cost
        self.alpha = alpha

        literal_speaker = np.ones((len(meanings), len(utterances))) * np.nan
        self.history = [literal_speaker]

    def update(self, listener):
        """
        Update the speaker based on the listener
        """

        # Log listener with -inf for zero probabilities
        mask = listener > 0
        log_listener = np.zeros_like(listener)
        log_listener[mask] = np.log(listener[mask])
        log_listener[~mask] = -np.inf
        log_listener = log_listener.T

        pragmatic_speaker = softmax(self.alpha * (log_listener - self.cost.reshape(-1,1)), axis=1)
        self.history.append(pragmatic_speaker)

    @property
    def as_df(self):
        return pd.DataFrame(self.as_array, index=self.meanings, columns=self.utterances)
    
    @property
    def as_array(self):
        return self.history[-1]


class RSAGain:

    def __init__(self):
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        self.coop_index_history = []

    def H_S_of_U_given_M(self, prior, speaker):
        """
        Compute the conditional mutual information of the utterances given the meanings.

        Parameters
        ----------
        prior : np.array (M,)
            The prior probability of each meaning.
        speaker : np.array (M,U)
            The speaker probability of each meaning given each utterance.

        """
        mask = speaker > 0
        log_speaker = np.zeros_like(speaker) # approximate x * log(x) to 0
        log_speaker[mask] = np.log(speaker[mask]) 
        cond_entropy = -np.sum(speaker * prior.reshape(-1,1) * log_speaker)
        self.cond_entropy_history.append(cond_entropy)
        return cond_entropy

    def expected_V_L_over_S(self, listener, speaker, prior, cost):
        """
        Compute the expected value of the listener over the speaker.

        Parameters
        ----------
        listener : np.array (U,M)
            The listener probability of each meaning given each utterance.
        speaker : np.array (M,U)
            The speaker probability of each meaning given each utterance.
        prior : np.array (M,)
            The prior probability of each meaning.
        cost : np.array (U,)
            The cost of each utterance.
        """
        joint_speaker = speaker * prior.reshape(-1,1)
        log_listener = np.zeros_like(listener)
        mask = listener > 0
        log_listener[mask] = np.log(listener[mask])
        log_listener[~mask] = -np.inf
        V_L = log_listener - cost.reshape(-1,1)
        pre_expected_V_L = np.zeros_like(joint_speaker.T)
        mask = (joint_speaker.T > 0) & (V_L != -np.inf)
        pre_expected_V_L[mask] = joint_speaker.T[mask] * V_L[mask]
        expected_V_L = np.sum(pre_expected_V_L)
        self.listener_value_history.append(expected_V_L)
        return expected_V_L
    
    def compute_gain(self, listener, speaker, prior, cost, alpha):
        """
        Compute the gain function.

        Parameters
        ----------
        listener : np.array (U,M)
            The listener probability of each meaning given each utterance.
        speaker : np.array (M,U)
            The speaker probability of each meaning given each utterance.
        prior : np.array (M,)
            The prior probability of each meaning.
        cost : np.array (U,)
            The cost of each utterance.
        alpha : float
            The rationality parameter.
        """
        gain = alpha * self.expected_V_L_over_S(listener, speaker, prior, cost) + self.H_S_of_U_given_M(prior, speaker)
        self.gain_history.append(gain)
        return gain
    
    def coop_index(self, listener, speaker):
        """
        Compute the cooperation index.

        Parameters
        ----------
        listener : np.array (U,M)
            The listener probability of each meaning given each utterance.
        speaker : np.array (M,U)
            The speaker probability of each meaning given each utterance.
        """
        coop_index = np.sum(speaker.T * listener) / listener.shape[0]
        self.coop_index_history.append(coop_index)
        return coop_index
    
    def get_diff(self):
        if len(self.gain_history) < 2:
            return float("inf")
        return abs(self.gain_history[-1] - self.gain_history[-2])



class RSA:
    """
    Rational Speech Act (RSA) model from the listener's perspective

    Parameters
    ----------
    meanings : List[str]
        List of possible meanings
    utterances : List[str]
        List of possible utterances
    lexicon : Union[np.ndarray, list[list[int]]]
        Lexicon matrix of shape (len(utterances),  len((meanings))
    prior : Optional[Union[np.ndarray,list[int]]]
        Prior probability of meanings. If None, uniform prior is assumed.
    cost : Optional[Union[np.ndarray,list[float]]]
        Cost of utterances. If None, uniform cost (zero) is assumed.
    alpha : Optional[float]
        Rationality parameter
    max_depth : Optional[int]
        Maximum depth of the model (number of iterations)
    tolerance : Optional[float]
        Tolerance for convergence
    """

    def __init__(self, meanings, utterances, lexicon, prior=None, cost=None, alpha=1., max_depth=None, tolerance=1e-6):

        # Check meanings and utterances
        if not is_list_of_strings(meanings):
            raise ValueError("meanings should be a list of strings")
        self.meanings = meanings
        if not is_list_of_strings(utterances):
            raise ValueError("utterances should be a list of strings")
        self.utterances = utterances

        # Check lexicon
        if is_list_of_list_of_numbers(lexicon) and len(lexicon) == len(utterances) and all(len(row) == len(meanings) for row in lexicon):
            lexicon = np.asarray(lexicon, dtype=float)
        elif not is_numeric_ndarray(lexicon) or lexicon.shape != (len(utterances), len(meanings)):
            raise ValueError("lexicon should be a numpy array of shape (len(utterances), len(meanings))")
        self.lexicon = lexicon

        # Check prior
        if prior is None:
            prior = np.ones(len(meanings), dtype=float) / len(meanings)
        elif is_numeric_ndarray(prior) and prior.shape == (len(meanings),):
            pass
        elif is_list_of_numbers(prior) and len(prior) == len(meanings):
            prior = np.asarray(prior).astype(float)
        else:
            raise ValueError("prior should be a list of floats or a numpy array of shape (len(meanings),)")
        self.prior = prior

        # Check cost
        if cost is None:
            cost = np.zeros(len(utterances), dtype=float)
        elif is_numeric_ndarray(cost) and cost.shape == (len(utterances),):
            pass
        elif is_list_of_numbers(cost) and len(cost) == len(utterances):
            cost = np.asarray(cost).astype(float)
        else:
            raise ValueError("cost should be a list of floats or a numpy array of shape (len(utterances),)")
        self.cost = cost

        # Check alpha
        if not is_positive_number(alpha):
            raise ValueError("alpha should be a positive number")
        self.alpha = float(alpha)

        # Check max_depth and tolerance
        if not is_positive_integer(max_depth) and max_depth is not None:
            raise ValueError("depth should be a positive integer or None")
        if not is_positive_number(tolerance) and tolerance is not None:
            raise ValueError("tolerance should be a positive number or None")
        if max_depth is None and tolerance is None:
            raise ValueError("Either max_depth or tolerance should be provided")
        self.max_depth = max_depth if max_depth is not None else float("inf")
        self.tolerance = tolerance if tolerance is not None else 0

        self.listener = None
        self.speaker = None
        self.gain = None


    def run(self, output_dir: Path, verbose: bool = False):
        """
        Run the RSA model for a given number of iterations

        Parameters
        ----------
        output_dir : Path
            Output directory to save the results
        verbose : bool
            If True, print the results to the console
        """
               
        # Configure logging
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter("%(message)s")
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        file_handler = logging.FileHandler(output_dir / "history.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Log configuration
        logger.info(f"Running RSA model for max depth {self.max_depth} and tolerance {self.tolerance:.2e}")
        logger.info("-" * 40)
        logger.info(
            f"\nLexicon:\n\n{pd.DataFrame(self.lexicon, index=self.utterances, columns=self.meanings)}\n\n"
            f"Prior:\n\n{pd.Series(self.prior, index=self.meanings).to_string()}\n\n"
            f"Cost:\n\n{pd.Series(self.cost, index=self.utterances).to_string()}\n\n"
            f"Alpha: {self.alpha}\n"
        )
        logger.info("-" * 40 + "\n")
        
        # Init listener and speaker
        self.listener = Listener(self.meanings, self.utterances, self.prior, self.lexicon)
        self.speaker = Speaker(self.meanings, self.utterances, self.cost, self.alpha)
        self.gain = RSAGain()
        gain = self.gain.compute_gain(self.listener.as_array, self.speaker.as_array, self.prior, self.cost, self.alpha)
        logger.info(f"Literal listener:\n{self.listener.get_literal_as_df()}\n\n")
        logger.info(f"Initial gain: {gain:.4f}\n")
        logger.info("-" * 40 + "\n")

        # Run the model for the given number of iterations
        i = 0
        while i < self.max_depth:
            # Update speaker
            self.speaker.update(self.listener.as_array)
            logger.info(f"Pragmatic speaker at step {i+1}:\n{self.speaker.as_df}\n")
            
            # Update listener
            self.listener.update(self.speaker.as_array)
            logger.info(f"Pragmatic listener at step {i+1}:\n{self.listener.as_df}\n")

            # Check for convergence
            gain = self.gain.compute_gain(self.listener.as_array, self.speaker.as_array, self.prior, self.cost, self.alpha)
            coop_index = self.gain.coop_index(self.listener.as_array, self.speaker.as_array)
            logger.info(f"Step: {i+1} | Gain: {gain:.4f} | Cooperation index: {coop_index:.4f}")
            logger.info("\n" + "-" * 40 + "\n")
            if self.gain.get_diff() < self.tolerance:
                logger.info(f"Converged after {i+1} iterations")
                break
            i += 1

        # Close logging
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    @property
    def history(self):
        return {
            "listener": [pd.DataFrame(l, index=self.utterances, columns=self.meanings) for l in self.listener.history],
            "speaker": [pd.DataFrame(s, index=self.meanings, columns=self.utterances) for s in self.speaker.history],
            "cond_entropy": self.gain.cond_entropy_history,
            "listener_value": self.gain.listener_value_history,
            "gain": self.gain.gain_history,
            "coop_index": self.gain.coop_index_history,
        }
    
    @history.setter
    def history(self, value):
        raise AttributeError("history is a read-only property")
    
    def save(self, output_dir: Path):
        args = {
            "meanings": self.meanings,
            "utterances": self.utterances,
            "lexicon": self.lexicon.tolist(),
            "prior": self.prior.tolist(),
            "cost": self.cost.tolist(),
            "alpha": self.alpha,
            "max_depth": self.max_depth,
            "tolerance": self.tolerance
        }
        save_yaml(args, output_dir / "args.yaml")

        with open(output_dir / "history.pkl", "wb") as f:
            pickle.dump({
                "listeners": np.asarray(self.listener.history),
                "speakers": np.asarray(self.speaker.history),
                "cond_entropy": np.asarray(self.gain.cond_entropy_history),
                "listener_value": np.asarray(self.gain.listener_value_history),
                "gain": np.asarray(self.gain.gain_history),
                "coop_index": np.asarray(self.gain.coop_index_history),
            }, f)

    @classmethod
    def load(cls, output_dir: Path):
        with open(output_dir / "args.yaml", "r") as f:
            args = yaml.safe_load(f)
        with open(output_dir / "history.pkl", "rb") as f:
            history = pickle.load(f)
        rsa = cls(**args)
        rsa.listener = Listener(rsa.meanings, rsa.utterances, rsa.prior, rsa.lexicon)
        rsa.listener.history = [l for l in history["listeners"]]
        rsa.speaker = Speaker(rsa.meanings, rsa.utterances, rsa.cost, rsa.alpha)
        rsa.speaker.history = [s for s in history["speakers"]]
        rsa.gain = RSAGain()
        rsa.gain.cond_entropy_history = history["cond_entropy"].tolist()
        rsa.gain.listener_value_history = history["listener_value"].tolist()
        rsa.gain.gain_history = history["gain"].tolist()
        rsa.gain.coop_index_history = history["coop_index"].tolist()
        return rsa
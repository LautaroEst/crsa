
from pathlib import Path
import numpy as np
from scipy.special import softmax
import logging
import pandas as pd

from ..src.utils import is_list_of_strings, is_numeric_ndarray, is_list_of_numbers, is_positive_number, is_positive_integer, is_list_of_list_of_numbers


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

    @property
    def literal_listener(self):
        return pd.DataFrame(self.history[0], index=self.utterances, columns=self.meanings)
      
    @property
    def df(self):
        return pd.DataFrame(self.value, index=self.utterances, columns=self.meanings)
    
    @property
    def value(self):
        return self.history[-1]
    

class Speaker:

    def __init__(self, meanings, utterances, cost, alpha):
        self.meanings = meanings
        self.utterances = utterances
        self.cost = cost
        self.alpha = alpha
        self.history = [None]

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
    def df(self):
        return pd.DataFrame(self.value, index=self.meanings, columns=self.utterances)
    
    @property
    def value(self):
        return self.history[-1]


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
    """

    def __init__(self, meanings, utterances, lexicon, prior=None, cost=None, alpha=1., depth=10):

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

        # Check depth
        if not is_positive_integer(depth):
            raise ValueError("depth should be a positive integer")
        self.depth = depth


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
        logger.info(f"Running RSA model for depth {self.depth}\n")
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
        logger.info(f"Literal listener:\n{self.listener.literal_listener}\n")
        logger.info("-" * 40 + "\n")

        # Run the model for the given number of iterations
        for i in range(self.depth):
            # Update speaker
            self.speaker.update(self.listener.value)
            logger.info(f"Pragmatic speaker at step {i+1}:\n{self.speaker.df}\n")
            
            # Update listener
            self.listener.update(self.speaker.value)
            logger.info(f"Pragmatic listener at step {i+1}:\n{self.listener.df}")
            logger.info("\n" + "-" * 40 + "\n")

        # Save history
        np.save(output_dir / "history.npy", self.history)

        # Close logging
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    @property
    def history(self):
        return [{
            "listener": pd.DataFrame(l, index=self.utterances, columns=self.meanings), 
            "speaker": pd.DataFrame(s, index=self.meanings, columns=self.utterances)
        } for l, s in zip(self.listener.history, self.speaker.history)]
    
    @history.setter
    def history(self, value):
        raise AttributeError("history is a read-only property")

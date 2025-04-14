
from collections import OrderedDict
from itertools import product
from pathlib import Path
import numpy as np
from scipy.special import softmax
import logging
import pandas as pd
import yaml
import pickle

from .utils import (
    is_list_of_strings, 
    is_numeric_ndarray, 
    is_list_of_numbers, 
    is_positive_number, 
    is_positive_integer, 
    save_yaml,
)
from .y_rsa import YRSA


class MultiturnYRSA:
    """
    Y-Rational Speech Act (RSA) model from the listener's perspective

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

    def __init__(self, meanings_A, meanings_B, categories, utterances_A, utterances_B, lexicon_A, lexicon_B, prior=None, cost_A=None, cost_B=None, alpha=1., max_depth=None, tolerance=1e-6, turns=1):

        # Check meanings and utterances
        if not is_list_of_strings(meanings_A) or not is_list_of_strings(meanings_B) or not is_list_of_strings(categories):
            raise ValueError("meanings and categories should be a list of strings")
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.categories = categories
        if not is_list_of_strings(utterances_A) or not is_list_of_strings(utterances_B):
            raise ValueError("utterances should be a list of strings")
        self.utterances_A = utterances_A
        self.utterances_B = utterances_B

        # Check lexicon
        ## TODO: Validate lexicon
        for agent, lexicon in [('A', lexicon_A), ('B', lexicon_B)]:
            if not isinstance(lexicon, list):
                raise ValueError("lexicon should be a list of 2D Tuple-like object")
            if not all(isinstance(row, (list, tuple)) for row in lexicon):
                if not all(isinstance(row, dict) and len(row) == 1 for row in lexicon):
                    raise ValueError("lexicon should be a list of 2D Tuple-like object")
                else:
                    lexicon = [next(iter(row.items())) for row in lexicon]
            else:
                lexicon = [tuple(row) for row in lexicon]
            setattr(self, f"lexicon_{agent}", OrderedDict(lexicon))

        # Check prior
        if prior is None:
            prior = np.ones((len(meanings_A), len(meanings_B), len(categories)), dtype=float)
            prior /= prior.sum()
        else:
            try:
                prior = np.asarray(prior).astype(float)
            except:
                raise ValueError("prior should be an array-like object of shape (len(meanings_A), len(meanings_B), len(categories))")
        self.prior = prior

        # Check cost
        for agent, cost, lexicon in [("A", cost_A, self.lexicon_A), ("B", cost_B, self.lexicon_B)]:
            utterances = lexicon.keys()
            if cost is None:
                cost = np.zeros(len(utterances), dtype=float)
            elif is_numeric_ndarray(cost) and cost.shape == (len(utterances),):
                pass
            elif is_list_of_numbers(cost) and len(cost) == len(utterances):
                cost = np.asarray(cost).astype(float)
            else:
                raise ValueError(f"cost_{agent} should be a list of floats or a numpy array of shape (len(utterances_{agent}),)")
            setattr(self, f"cost_{agent}", cost)

        # Check alpha
        if not is_positive_number(alpha):
            raise ValueError("alpha should be a positive number")
        self.alpha = float(alpha)

        # Check max_depth and tolerance
        if not is_positive_integer(max_depth) and max_depth != float("inf"):
            raise ValueError("depth should be a positive integer or inf")
        if not is_positive_number(tolerance) and tolerance != 0:
            raise ValueError("tolerance should be a positive number or None")
        if max_depth == float("inf") and tolerance == 0:
            raise ValueError("Either max_depth or tolerance should be provided")
        self.max_depth = max_depth
        self.tolerance = tolerance

        # Check turns
        if not is_positive_integer(turns):
            raise ValueError("turns should be a positive integer")
        self.turns = turns
        self.turns_history = OrderedDict()

    def _get_turn_world(self, turn, agent=None):
        turn_utterances = []
        for past_turns in range(turn+1):
            if past_turns % 2 == 0:
                turn_utterances.append(self.utterances_A)
            else:
                turn_utterances.append(self.utterances_B)

        lexicon = getattr(self, f"lexicon_{agent}")
        cost = getattr(self, f"cost_{agent}")
        
        turn_utterances_string = []
        turn_lexicon = []
        turn_cost = []
        utt2idx = {u: i for i, u in enumerate(lexicon.keys())}
        for utterance in product(*turn_utterances):
            u = " ".join(utterance)
            turn_utterances_string.append(u)
            turn_lexicon.append(lexicon[u])
            turn_cost.append(cost[utt2idx[u]])
        
        return turn_utterances_string, np.asarray(turn_lexicon, dtype=float), np.asarray(turn_cost, dtype=float)

    def run(self, output_dir: Path, verbose: bool = False):

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
        logger.info(f"Running Multiturn Y-RSA model for {self.turns} turns, max depth {self.max_depth} and tolerance {self.tolerance:.2e}")
        logger.info("-" * 40)
        logger.info(
            f"\nAgent A's Lexicon:\n\n{pd.DataFrame.from_dict(self.lexicon_A, orient='index', columns=self.meanings_A)}\n\n"
            f"Agent B's Lexicon:\n\n{pd.DataFrame.from_dict(self.lexicon_B, orient='index', columns=self.meanings_B)}\n\n"
            f"Prior:\n\n{pd.DataFrame(self.prior.reshape(-1,len(self.categories)), index=pd.MultiIndex.from_product([self.meanings_A, self.meanings_B], names=['meanings_A','meanings_B']), columns=self.categories)}\n\n"
            f"Costs of utterances from agent A: \n\n{pd.Series(self.cost_A, index=self.lexicon_A.keys()).to_string()}\n\n"
            f"Costs of utterances from agent B: \n\n{pd.Series(self.cost_B, index=self.lexicon_B.keys()).to_string()}\n\n"
            f"Alpha: {self.alpha}\n"
        )
        logger.info("-" * 40 + "\n")

        turn = 0
        while turn < self.turns:

            ## Agent A speaks and Agent B listens
            turn_utterances, turn_lexicon, turn_cost = self._get_turn_world(turn, agent="A")
            yrsa_A = YRSA(
                meanings_A=self.meanings_A,
                meanings_B=self.meanings_B,
                categories=self.categories,
                utterances=turn_utterances,
                lexicon=turn_lexicon,
                prior=self.prior,
                cost=turn_cost,
                alpha=self.alpha,
                max_depth=self.max_depth,
                tolerance=self.tolerance,
            )
            yrsa_A.run(output_dir, verbose, prefix=f"agentA_turn{turn:02d}_")
            yrsa_A.save(output_dir, prefix=f"agentA_turn{turn:02d}_")
            self.turns_history[turn] = yrsa_A
            logger.info(f"Agent A's turn {turn} info logged into agentA_turn{turn:02d}_history.log")

            if turn == self.turns - 1:
                break

            ## Agent B speaks and Agent A listens
            turn += 1
            turn_utterances, turn_lexicon, turn_cost = self._get_turn_world(turn, agent="B")
            yrsa_B = YRSA(
                meanings_A=self.meanings_A,
                meanings_B=self.meanings_B,
                categories=self.categories,
                utterances=turn_utterances,
                lexicon=turn_lexicon,
                prior=self.prior,
                cost=turn_cost,
                alpha=self.alpha,
                max_depth=self.max_depth,
                tolerance=self.tolerance,
            )
            yrsa_B.run(output_dir, verbose, prefix=f"agentB_turn{turn:02d}_")
            yrsa_B.save(output_dir, prefix=f"agentB_turn{turn:02d}_")
            self.turns_history[turn] = yrsa_B
            logger.info(f"Agent B's turn {turn} info logged into agentB_turn{turn:02d}_history.log")
            logger.info("-" * 40 + "\n")
            turn += 1
            
        # Log final state
        logger.info("Reached final turn")
        logger.info("-" * 40)

        # Close logging
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    @property
    def history(self):
        return self.turns_history
    
    @history.setter
    def history(self, value):
        raise AttributeError("history is a read-only property")
    
    def save(self, output_dir: Path, prefix: str = ""):
        args = {
            "meanings_A": self.meanings_A,
            "meanings_B": self.meanings_B,
            "categories": self.categories,
            "utterances_A": self.utterances_A,
            "utterances_B": self.utterances_B,
            "lexicon_A": [i for i in self.lexicon_A.items()],
            "lexicon_B": [i for i in self.lexicon_B.items()],
            "prior": self.prior.tolist(),
            "cost_A": self.cost_A.tolist(),
            "cost_B": self.cost_B.tolist(),
            "alpha": self.alpha,
            "max_depth": self.max_depth,
            "tolerance": self.tolerance,
            "turns": self.turns,
        }
        save_yaml(args, output_dir / f"{prefix}args.yaml")

        for turn in range(self.turns):
            self.turns_history[turn].save(output_dir, prefix=f"{prefix}turn{turn:02d}_")

    @classmethod
    def load(cls, output_dir: Path, prefix: str = ""):
        with open(output_dir / f"{prefix}args.yaml", "r") as f:
            args = yaml.safe_load(f)
        model = cls(**args)
        for turn in range(model.turns):
            model.turns_history[turn] = YRSA.load(output_dir, prefix=f"{prefix}turn{turn:02d}_")
        return model
        

    

if __name__ == "__main__":
    with open(Path("configs/worlds/multiround_toy_game.yaml"), "r") as f:
        args = yaml.safe_load(f)
    model = MultiturnYRSA(**args, alpha=1., max_depth=10)
    print(model.get_utterances_for_turn(1))

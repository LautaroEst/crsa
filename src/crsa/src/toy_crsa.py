
from itertools import cycle
import logging
from pathlib import Path
import pandas as pd
import yaml

from .lexicons import get_lexicon_cls
from .toy_crsa_turn import ToyCRSATurn
from .toy_dialog_model import ToyDialogModel
from .utils import save_yaml



class ToyCRSA:

    def __init__(self, meanings_A, meanings_B, categories, utterances_A, utterances_B, cost_A, cost_B, lexicon, prior, alpha=1.0, max_depth=None, tolerance=1e-6, turns=1):
        
        # World parameters        
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.categories = categories
        self.utterances_A = utterances_A
        self.utterances_B = utterances_B
        self.lxn_cls = get_lexicon_cls(lexicon)
        self.prior = prior

        # Pragmatic parameters
        self.cost_A = cost_A
        self.cost_B = cost_B
        self.alpha = alpha
        
        # Iteration parameters
        self.turns = turns
        self.max_depth = max_depth
        self.tolerance = tolerance


    def run(self, output_dir, verbose=False):
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
        logger.info(f"Running CRSA model for {self.turns} turns, max depth {self.max_depth} and tolerance {self.tolerance:.2e}")
        logger.info("-" * 40 + "\n")
        logger.info(
            f"Prior:\n\n{pd.DataFrame(self.prior.reshape(-1,len(self.categories)), index=pd.MultiIndex.from_product([self.meanings_A, self.meanings_B], names=['meanings_A','meanings_B']), columns=self.categories)}\n\n"
            f"Costs of utterances from agent A: \n\n{pd.Series(self.cost_A, index=self.utterances_A).to_string()}\n\n"
            f"Costs of utterances from agent B: \n\n{pd.Series(self.cost_B, index=self.utterances_B).to_string()}\n\n"
            f"Alpha: {self.alpha}\n"
        )
        logger.info("-" * 40 + "\n")

        self.turns_history = []
        self.dialog_model = ToyDialogModel(self.meanings_A, self.meanings_B, self.utterances_A, self.utterances_B)
        for current_turn, speaking_agent in zip(range(1, self.turns+1), cycle("AB")):
            listen_agent = "A" if speaking_agent == "B" else "B"
            logger.info(f"Turn {current_turn}: Agent {speaking_agent} speaks and Agent {listen_agent} listens\n")
            self._run_turn(current_turn, output_dir, verbose=verbose, prefix=f"agent{speaking_agent}_turn{current_turn:02d}_")
            self.dialog_model.update(self.turns_history[-1].speaker)
            
        # Log final state
        logger.info("Reached final turn")
        logger.info("-" * 40)

        # Close logging
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)


    def _run_turn(self, current_turn, output_dir, verbose=False, prefix=""):
        meanings_S = self.meanings_A if current_turn % 2 == 1 else self.meanings_B
        meanings_L = self.meanings_B if current_turn % 2 == 1 else self.meanings_A
        utterances = self.utterances_A if current_turn % 2 == 1 else self.utterances_B
        cost = self.cost_A if current_turn % 2 == 1 else self.cost_B
        lexicon = self.lxn_cls(self.meanings_A, self.meanings_B, self.utterances_A, self.utterances_B, current_turn=current_turn)
        
        model = ToyCRSATurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=utterances,
            prior=self.prior,
            lexicon=lexicon,
            dialog_model=self.dialog_model,
            cost=cost,
            alpha=self.alpha,
            max_depth=self.max_depth,
            tolerance=self.tolerance
        )
        model.run(output_dir, verbose, prefix)
        model.save(output_dir, prefix)
        self.turns_history.append(model)
            

    def save(self, output_dir: Path, prefix: str = ""):
        args = {
            "meanings_A": self.meanings_A,
            "meanings_B": self.meanings_B,
            "categories": self.categories,
            "utterances_A": self.utterances_A,
            "utterances_B": self.utterances_B,
            "lexicon": self.lxn_cls.NAME,
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
        self.dialog_model.save(output_dir, prefix=f"{prefix}dialog_model_")


    @classmethod
    def load(cls, output_dir: Path, prefix: str = ""):
        with open(output_dir / f"{prefix}args.yaml", "r") as f:
            args = yaml.safe_load(f)
        model = cls(**args)
        for turn in range(model.turns):
            model.turns_history[turn] = ToyCRSATurn.load(output_dir, prefix=f"{prefix}turn{turn:02d}_")
        model.dialog_model = ToyDialogModel.load(output_dir, prefix=f"{prefix}dialog_model_")
        return model
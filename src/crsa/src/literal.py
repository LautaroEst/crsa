
import logging
from pathlib import Path
import pandas as pd
import yaml
import numpy as np

from .literal_turn import LiteralTurn
from .utils import save_yaml



class Literal:

    def __init__(self, speaker_now, meanings_A, meanings_B, categories, utterances_A, utterances_B, lexicon_A, lexicon_B, prior, past_utterances, alpha, cost_A, cost_B):
        
        # World parameters
        self.speaker_now = speaker_now        
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.categories = categories
        self.utterances_A = utterances_A
        self.utterances_B = utterances_B
        self.lexicon_A = np.asarray(lexicon_A, dtype=float)
        self.lexicon_B = np.asarray(lexicon_B, dtype=float)
        self.prior = np.asarray(prior, dtype=float)
        self.prior = self.prior / np.sum(self.prior)
        self.past_utterances = past_utterances

        # Pragmatic params
        self.alpha = alpha
        self.cost_A = cost_A
        self.cost_B = cost_B

        # History
        self.turns_history = []


    def run(self, output_dir=None, verbose=False):
        
        # Configure logging
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter("%(message)s")
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        if output_dir is not None:
            file_handler = logging.FileHandler(output_dir / "history.log", mode="w", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Log configuration
        logger.info(f"Running Literal model.")
        logger.info(f"Past utterances: {self.past_utterances}")
        logger.info("-" * 40 + "\n")
        logger.info(
            f"Meanings A: {self.meanings_A}\n"
            f"Meanings B: {self.meanings_B}\n"
            f"Categories: {self.categories}\n\n"
            f"Prior:\n\n{pd.DataFrame(self.prior.reshape(-1,len(self.categories)), index=pd.MultiIndex.from_product([self.meanings_A, self.meanings_B], names=['meanings_A','meanings_B']), columns=self.categories)}\n\n"
            f"Lexicon A:\n\n{pd.DataFrame(self.lexicon_A, index=self.utterances_A, columns=self.meanings_A)}\n\n"
            f"Lexicon B:\n\n{pd.DataFrame(self.lexicon_B, index=self.utterances_B, columns=self.meanings_B)}\n\n"
            f"Costs of utterances from agent A: \n\n{pd.Series(self.cost_A, index=self.utterances_A).to_string()}\n\n"
            f"Costs of utterances from agent B: \n\n{pd.Series(self.cost_B, index=self.utterances_B).to_string()}\n\n"
            f"Alpha: {self.alpha}\n"
        )
        logger.info("-" * 40 + "\n")

        # Run CRSA turns
        turns = len(self.past_utterances) + 1
        for current_turn in range(1, turns + 1):

            if not self.past_utterances:
                speaking_agent = self.speaker_now
                past_utterances = []
            else:
                speaking_agent = self.past_utterances[current_turn-1]["speaker"] if current_turn <= len(self.past_utterances) else self.speaker_now
                past_utterances = self.past_utterances[:current_turn-1]
            listen_agent = "A" if speaking_agent == "B" else "B"

            # Log turn information
            logger.info(f"Turn {current_turn}: Agent {speaking_agent} speaks and Agent {listen_agent} listens\n")
            
            # Run CRSA for the turn
            model = self._run_turn(past_utterances, speaking_agent, output_dir, verbose=verbose, prefix=f"turn{current_turn:02d}_")
            self.turns_history.append(model)
            
        # Log final state
        logger.info("Reached final turn")
        logger.info("-" * 40)

        # Close logging
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    def _run_turn(self, past_utterances, speaker="A", output_dir=None, verbose=False, prefix=""):
        # past_utterances not used. Shown only for compatibility with the CRSA model
        
        meanings_S = self.meanings_A if speaker == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker == "A" else self.meanings_A
        utterances = self.utterances_A if speaker == "A" else self.utterances_B
        lexicon = self.lexicon_A if speaker == "A" else self.lexicon_B
        cost = self.cost_A if speaker == "A" else self.cost_B
        prior = self.prior.copy() if speaker == "A" else self.prior.copy().transpose(1, 0, 2)

        model = LiteralTurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=utterances,
            prior=prior,
            lexicon=lexicon,
            cost=cost,
            alpha=self.alpha,
        )
        model.run(output_dir, verbose, prefix)
        return model

    def save(self, output_dir: Path, prefix: str = ""):
        args = {
            "speaker_now": self.speaker_now,
            "meanings_A": self.meanings_A,
            "meanings_B": self.meanings_B,
            "categories": self.categories,
            "utterances_A": self.utterances_A,
            "utterances_B": self.utterances_B,
            "lexicon_A": self.lexicon_A.tolist(),
            "lexicon_B": self.lexicon_B.tolist(),
            "prior": self.prior.tolist(),
            "past_utterances": self.past_utterances,
            "alpha": self.alpha,
            "cost_A": self.cost_A,
            "cost_B": self.cost_B,
        }
        save_yaml(args, output_dir / f"{prefix}args.yaml")

        for turn, turn_data in enumerate(self.turns_history, start=1):
            turn_data.save(output_dir, prefix=f"{prefix}turn{turn:02d}_")


    @classmethod
    def load(cls, output_dir: Path, prefix: str = ""):
        with open(output_dir / f"{prefix}args.yaml", "r") as f:
            args = yaml.safe_load(f)
        model = cls(**args)
        for turn in range(len(args["past_utterances"]) + 1):
            model.turns_history.append(LiteralTurn.load(output_dir, prefix=f"{prefix}turn{turn+1:02d}_"))

        return model

from itertools import cycle
import logging
from pathlib import Path
import pandas as pd
import yaml
import numpy as np

from .crsa_turn import CRSATurn
from .utils import save_yaml



class CRSA:

    def __init__(self, speaker_now, meanings_A, meanings_B, categories, utterances_A, utterances_B, lexicon_A, lexicon_B, prior, past_utterances, cost_A=None, cost_B=None, alpha=1.0, max_depth=None, tolerance=1e-6):
        
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

        # Pragmatic parameters
        self.cost_A = np.asarray(cost_A, dtype=float) if cost_A is not None else np.zeros(len(utterances_A))
        self.cost_B = np.asarray(cost_B, dtype=float) if cost_B is not None else np.zeros(len(utterances_B))
        self.alpha = alpha
        
        # Iteration parameters
        self.max_depth = max_depth
        self.tolerance = tolerance

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
        logger.info(f"Running CRSA model for alpha={self.alpha}, max_depth={self.max_depth} and tolerance={self.tolerance}.")
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
        meanings_S = self.meanings_A if speaker == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker == "A" else self.meanings_A
        utterances = self.utterances_A if speaker == "A" else self.utterances_B
        lexicon = self.lexicon_A if speaker == "A" else self.lexicon_B
        cost = self.cost_A if speaker == "A" else self.cost_B
        prior = self.prior.copy() if speaker == "A" else self.prior.copy().transpose(1, 0, 2)
        ds, dl = self._compute_dialog_model(past_utterances, speaker)

        model = CRSATurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=utterances,
            prior=prior,
            lexicon=lexicon,
            past_utterances=past_utterances,
            ds=ds,
            dl=dl,
            cost=cost,
            alpha=self.alpha,
            max_depth=self.max_depth,
            tolerance=self.tolerance
        )
        model.run(output_dir, verbose, prefix)
        return model
    
    def _compute_dialog_model(self, past_utterances, speaker):
        if not past_utterances:
            return None, None
        
        ds, dl = [], []
        for data, model in zip(past_utterances, self.turns_history):
            if data["speaker"] == speaker:
                utterances_S = self.utterances_A if speaker == "A" else self.utterances_B
                utt_idx = utterances_S.index(data["utterance"])
                ds.append(model.speaker.as_array[:, utt_idx])
            else:
                utterances_L = self.utterances_B if speaker == "A" else self.utterances_A
                utt_idx = utterances_L.index(data["utterance"])
                dl.append(model.speaker.as_array[:, utt_idx])
        
        ds_arr = np.ones(len(self.meanings_A if speaker == "A" else self.meanings_B))
        for d in ds:
            ds_arr *= d

        dl_arr = np.ones(len(self.meanings_B if speaker == "A" else self.meanings_A))
        for d in dl:
            dl_arr *= d
        
        return ds_arr, dl_arr
    
    def continue_from_last_turn(self, new_past_utterances, speaker_now, output_dir=None, verbose=False):

        turns_runned = len(self.past_utterances) + 1
        if len(self.turns_history) != turns_runned:
            raise ValueError(f"CRSA model has not been run for {turns_runned} turns. Please run the model first.")
        self.past_utterances.extend(new_past_utterances)
        self.speaker_now = speaker_now

        # Configure logging
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter("%(message)s")
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        if output_dir is not None:
            file_handler = logging.FileHandler(output_dir / "history.log", mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        
        # Log configuration
        logger.info(f"Continuing CRSA model for alpha={self.alpha}, max_depth={self.max_depth} and tolerance={self.tolerance}.")
        logger.info(f"Past utterances: {self.past_utterances}")
        logger.info("-" * 40 + "\n")

        # Run CRSA turns
        turns = len(self.past_utterances) + 1
        for turn in range(turns_runned + 1, turns + 1):

            # Determine the speaker and listener
            speaking_agent = self.past_utterances[turn-1]["speaker"] if turn <= len(self.past_utterances) else self.speaker_now
            past_utterances = self.past_utterances[:turn-1]
            listen_agent = "A" if speaking_agent == "B" else "B"

            # Log turn information
            logger.info(f"Turn {turn}: Agent {speaking_agent} speaks and Agent {listen_agent} listens\n")
            
            # Run CRSA for the turn
            model = self._run_turn(past_utterances, speaking_agent, output_dir, verbose=verbose, prefix=f"turn{turn:02d}_")
            self.turns_history.append(model)


            

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
            "cost_A": self.cost_A.tolist(),
            "cost_B": self.cost_B.tolist(),
            "alpha": self.alpha,
            "max_depth": self.max_depth,
            "tolerance": self.tolerance,
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
            model.turns_history.append(CRSATurn.load(output_dir, prefix=f"{prefix}turn{turn+1:02d}_"))

        return model
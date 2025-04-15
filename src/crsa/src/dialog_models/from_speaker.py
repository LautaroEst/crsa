

from pathlib import Path
import pickle

import yaml

from .base import BaseDialogModel
from ..utils import save_yaml


class DialogModelFromSpeaker(BaseDialogModel):

    def __init__(
        self,
        meanings_A,
        meanings_B,
        utterances_A,
        utterances_B,
    ):
        self.meanings = {
            "A": meanings_A,
            "B": meanings_B,
        }
        self.utterances = {
            "A": utterances_A,
            "B": utterances_B,
        }
        self.past_utterances = []

    def update(self, speaker, agent="A"):
        if not self.past_utterances:
            self.past_utterances.extend(self.utterances[agent])
        else:
            new_utterances = []
            for u in self.past_utterances:
                for u_ in self.utterances[agent]:
                    new_utterances.append(u + " " + u_)
            self.past_utterances = new_utterances
                
            

    def save(self, output_dir: Path, prefix: str = ""):
        save_yaml({
            "meanings_A": self.meanings["A"],
            "meanings_B": self.meanings["B"],
            "utterances_A": self.utterances["A"],
            "utterances_B": self.utterances["B"],
        }, output_dir / f"{prefix}args.yaml")

        # TODO: save the dialog model state
        with open(output_dir / f"{prefix}history.pkl", "wb") as f:
            pickle.dump({
            }, f)


    @classmethod
    def load(cls, output_dir: Path, prefix: str = ""):
        with open(output_dir / f"{prefix}args.yaml", "r") as f:
            args = yaml.safe_load(f)
        model = cls(**args)

        # TODO: load the dialog model state
        with open(output_dir / f"{prefix}history.pkl", "rb") as f:
            data = pickle.load(f)

        return model


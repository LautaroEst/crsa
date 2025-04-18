

from pathlib import Path
import pickle

import numpy as np
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
        self.history = []

    def update(self, speaker, agent="A"):
        
        if not self.past_utterances:
            self.past_utterances.extend(self.utterances[agent])
            spk_arr = speaker.as_array
            spk_arr = np.repeat(np.expand_dims(spk_arr, axis=1), len(self.meanings["B"]), axis=1)
            self.history.append(spk_arr.reshape(spk_arr.shape[0], spk_arr.shape[1],-1))
        else:
            new_utterances = []
            for u in self.past_utterances:
                for u_ in self.utterances[agent]:
                    new_utterances.append(u + " " + u_)
            self.past_utterances = new_utterances
            
            agent_lower = agent.lower()
            # old_dm(a,b,w) = Ps(w|a,b)
            # speaker({agent_lower},w,u) = S(u|{agent_lower},w)
            # new_dm(a,b,x) = S(u|{agent_lower},w)Ps(w|a,b) # x = concat[u,w]
            new_dm = np.einsum(f"{agent_lower}wu,abw->abuw", speaker.as_array, self.history[-1])
            new_dm = new_dm.reshape(new_dm.shape[0], new_dm.shape[1], -1)
            self.history.append(new_dm)
        
        
    @property
    def as_array(self):
        return self.history[-1]
            

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




from itertools import product
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
        self.speakers = {
            "A": [],
            "B": [],
        }


    def update(self, speaker):
        if len(self.speakers["A"]) == len(self.speakers["B"]):
            # speaker(a,u) = S(u|a)
            self.speakers["A"].append(speaker)
        else:
            # speaker(b,u) = S(u|b)
            self.speakers["B"].append(speaker)


    def _past_utterances(self):
        if len(self.speakers["A"]) == len(self.speakers["B"]) == 0:
            return [None]

        utts = []
        for t in range(len(self.speakers["A"]) + len(self.speakers["B"])):
            if t % 2 == 0:
                utts.append(self.utterances["A"])
            else:
                utts.append(self.utterances["B"])
        
        return product(*utts)


    def utterances_iterator(self):
        
        for utts in self._past_utterances():
            
            if utts is None:
                yield None, None, None, None

            else:
                dm_a = []
                dm_b = []
                for i, utt in enumerate(utts):
                    if i % 2 == 0:
                        idx_a = self.utterances["A"].index(utt)
                        dm_a.append(self.speakers["A"][i // 2][:, idx_a])
                    else:
                        idx_b = self.utterances["B"].index(utt)
                        dm_b.append(self.speakers["B"][i // 2][:, idx_b])
                if len(dm_b) == 0:
                    dm_a = dm_a[0].copy()
                    dm_b = np.ones(len(self.meanings["B"]))
                else:
                    dm_a = np.prod(dm_a, axis=0)
                    dm_b = np.prod(dm_b, axis=0)
                dm_prod = np.outer(dm_a, dm_b)

                if len(utts) % 2 == 0:
                    dm_s = dm_a
                    dm_l = dm_b
                else:
                    dm_s = dm_b
                    dm_l = dm_a
                    dm_prod = dm_prod.T
                    
                yield utts, dm_s, dm_l, dm_prod
        
        
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
    

if __name__ == "__main__":
    model = DialogModelFromSpeaker(
        meanings_A=["AA", "AB", "BA", "BB"],
        meanings_B=["11", "12", "21", "22"],
        utterances_A=["1st", "2nd"],
        utterances_B=["1", "2"],
    )
    for utts, dm_a, dm_b, dm_prod in model.utterances_iterator():
        print("Utterances:", utts)
        print("DM_A:\n", dm_a)
        print("DM_B:\n", dm_b)
        print("DM_Prod:\n", dm_prod)
    print("===")
    # S(u1|a)
    speaker = np.array([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
    ])
    model.update(speaker)
    for utts, dm_a, dm_b, dm_prod in model.utterances_iterator():
        print("Utterances:", utts)
        print("DM_A:\n", dm_a)
        print("DM_B:\n", dm_b)
        print("DM_Prod:\n", dm_prod)
    print("===")
    # S(u2|u1,b)
    speaker = np.array([
        [0.5, 0.5],
        [0.6, 0.4],
        [0.7, 0.3],
        [0.8, 0.2],
    ])
    model.update(speaker)
    for utts, dm_a, dm_b, dm_prod in model.utterances_iterator():
        print("Utterances:", utts)
        print("DM_A:\n", dm_a)
        print("DM_B:\n", dm_b)
        print("DM_Prod:\n", dm_prod)
    print("===")
    # S(u3|u1,u2,a)
    speaker = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
    ])
    model.update(speaker)
    for utts, dm_a, dm_b, dm_prod in model.utterances_iterator():
        print("Utterances:", utts)
        print("DM_A:\n", dm_a)
        print("DM_B:\n", dm_b)
        print("DM_Prod:\n", dm_prod)




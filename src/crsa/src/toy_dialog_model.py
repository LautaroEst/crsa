

from pathlib import Path


class ToyDialogModel:

    def __init__(
        self,
        meanings_A,
        meanings_B,
        utterances_A,
        utterances_B,
    ):
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.utterances_A = utterances_A
        self.utterances_B = utterances_B

    def update(self, speaker):
        pass

    def save(self, output_dir: Path, prefix: str = ""):
        pass

    @classmethod
    def load(cls, output_dir: Path, prefix: str = ""):
        pass

    def to_dict(self):
        return {
            "meanings_A": self.meanings_A,
            "meanings_B": self.meanings_B,
            "utterances_A": self.utterances_A,
            "utterances_B": self.utterances_B,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            meanings_A=data["meanings_A"],
            meanings_B=data["meanings_B"],
            utterances_A=data["utterances_A"],
            utterances_B=data["utterances_B"],
        )
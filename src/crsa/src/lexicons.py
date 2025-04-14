
def get_lexicon_cls(lexicon_name):
    if lexicon_name == FindA1Lexicon.NAME:
        return FindA1Lexicon
    else:
        raise ValueError(f"Lexicon {lexicon_name} not available.")

class BaseLexicon:
    NAME = None

    def compute_value(self, meaning, utterance, history, current_turn):
        raise NotImplementedError("Subclasses must implement this method.")

class FindA1Lexicon(BaseLexicon):
    NAME = "findA1"

    def __init__(self, meanings_A, meanings_B, utterances_A, utterances_B, current_turn=1):
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.utterances_A = utterances_A
        self.utterances_B = utterances_B
        self.current_turn = current_turn
        self.past_utterances = []
    
    def as_array(self):
        pass

    def to_dict(self):
        return {
            "meanings_A": self.meanings_A,
            "meanings_B": self.meanings_B,
            "utterances_A": self.utterances_A,
            "utterances_B": self.utterances_B,
            "current_turn": self.current_turn,
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            meanings_A=data["meanings_A"],
            meanings_B=data["meanings_B"],
            utterances_A=data["utterances_A"],
            utterances_B=data["utterances_B"],
            current_turn=data["current_turn"],
        )

def get_lexicon_cls(lexicon_name):
    if lexicon_name == "findA1":
        return FindA1Lexicon
    else:
        raise ValueError(f"Lexicon {lexicon_name} not available.")

class BaseLexicon:
    NAME = None

class FindA1Lexicon(BaseLexicon):
    NAME = "findA1"

    def __init__(self, meanings, utterances_A, utterances_B, turn=1):
        self.meanings = meanings
        self.utterances_A = utterances_A
        self.utterances_B = utterances_B
        self.turn = turn
    
    def as_array(self):
        pass
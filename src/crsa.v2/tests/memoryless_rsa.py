
from ..src.memoryless_rsa import MemorylessRSA
import numpy as np

prior = np.array([
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
    [[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
    [[0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
    [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
])
prior = prior / np.sum(prior)

costs = np.array([0, 0, 0])

lexicon_A = np.array([
    [1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1],
])
lexicon_B = np.array([
    [1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1],
])



def main():
    model = MemorylessRSA(
        meanings_A=["AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"],
        meanings_B=["111", "112", "121", "122", "211", "212", "221", "222"],
        categories=["None", "1st", "2nd", "3rd"],
        utterances=["1st", "2nd", "3rd"],
        lexicon_A=lexicon_A,
        lexicon_B=lexicon_B,
        prior=prior,
        alpha=2.0,
        costs=costs,
        max_depth=np.inf,
        tolerance=1e-3
    )
    conversation = [
        {"utterance": "1st", "speaker": "A"},
        {"utterance": "2nd", "speaker": "B"},
        {"utterance": "2nd", "speaker": "A"},
    ]
    model.run(conversation, speaker_now="B")

    for turn in model.turns_history:
        print(turn.speaker.as_df)
        print(turn.listener.as_df)
        print()


if __name__ == "__main__":
    main()
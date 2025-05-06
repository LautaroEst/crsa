
from ..src.prior_model import PriorModel
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



def main():
    model = PriorModel(
        meanings_A=["AAA", "AAB", "ABA", "ABB", "BAA", "BAB", "BBA", "BBB"],
        meanings_B=["111", "112", "121", "122", "211", "212", "221", "222"],
        categories=["None", "1st", "2nd", "3rd"],
        utterances=["1st", "2nd", "3rd"],
        prior=prior,
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
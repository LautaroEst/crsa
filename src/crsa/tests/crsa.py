

from itertools import cycle
from lightning import seed_everything
import torch

from ..src.pragmatics import init_model
from ..src.datasets import FindA1Dataset
from ..src.speakers import StaticSpeaker


def main():

    seed_everything(1234, verbose=False)

    game_size = 4
    dataset = FindA1Dataset(game_size=game_size, n_rounds=10)
    world = dataset.world
    
    model = init_model("crsa", logprior=world["logprior"])

    speaker = StaticSpeaker.from_lexicon(
        world["lexicon_A"], world["lexicon_B"]
    )

    alpha = 2.5

    for i, sample in enumerate(dataset.iter_samples()):

        # Get the meanings and target
        meaning_A = sample["meaning_A"]
        meaning_B = sample["meaning_B"]
        target = sample["target"]

        # Generate dialog with the model
        past_utterances = []
        model.reset()
        for turn, spk_name in enumerate(cycle("AB"), start=1):

            # Get the literal speaker
            lit_logspk, costs = speaker(past_utterances, spk_name)
            
            # Run the pragmatic model
            prag_logspk, prag_loglist = model.run_turn(lit_logspk, spk_name, costs, alpha)

            # Sample an utterance from the pragmatic speaker
            meaning_S = meaning_A if spk_name == "A" else meaning_B
            new_utt = model.sample_utterance(meaning_S, "greedy")
            past_utterances.append({"spk_name": spk_name, "utterance": new_utt})


if __name__ == "__main__":
    main()
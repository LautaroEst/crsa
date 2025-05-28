

from itertools import cycle
from lightning import seed_everything
import torch

from ..src.pragmatics import init_model
from ..src.datasets import FindA1Dataset
from ..src.speakers import StaticSpeaker

import numpy as np


def main():

    seed_everything(1234, verbose=False)

    game_size = 4
    dataset = FindA1Dataset(game_size=game_size, n_rounds=10)
    world = dataset.world
    
    crsa = init_model("crsa2", logprior=world["logprior"])
    naive = init_model("naive_crsa", 
        logprior=world["logprior"], meanings_A=world["meanings_A"],
        meanings_B=world["meanings_B"], categories=world["targets"],
        utterances=world["utterances"], lexicon_A=world["lexicon_A"].numpy(),
        lexicon_B=world["lexicon_B"].numpy()
    )
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
        crsa.reset()
        naive.reset(meaning_A=world["meanings_A"][meaning_A.item()], meaning_B=world["meanings_B"][meaning_B.item()])
        for turn, spk_name in enumerate(cycle("AB"), start=1):

            # Get the literal speaker
            lit_logspk, costs = speaker(past_utterances, spk_name)
            lit_spk = torch.exp(lit_logspk)

            if turn > 1:
                print(crsa.beliefs[-1]["A"])
                print(naive.belief_A)
                print()
                print(crsa.beliefs[-1]["B"])
                print(naive.belief_B)
                print()
            
            # Run the pragmatic model
            prag_spk, prag_list = crsa.run_turn(lit_spk, spk_name, costs, alpha, max_depth=1)
            
            # Sample an utterance from the pragmatic speaker
            meaning_S = meaning_A if spk_name == "A" else meaning_B
            new_utt = crsa.sample_utterance(meaning_S, "greedy")
            past_utterances.append({"spk_name": spk_name, "utterance": new_utt})

            naive.run_turn(speaker=spk_name, costs=costs.numpy(), alpha=alpha, max_depth=1)

            print([{"spk_name": u["spk_name"], "utterance": world["utterances"][u["utterance"]]} for u in past_utterances])
            print(naive.past_utterances)

            pl_crsa = crsa.turns[-1].listener.as_tensor(log=False).numpy()
            pl_naive = naive.turns_history[-1].listener.as_array
            err = np.abs(pl_crsa - pl_naive)
            print(pl_crsa[err > 0.01])
            print(pl_naive[err > 0.01])


if __name__ == "__main__":
    main()
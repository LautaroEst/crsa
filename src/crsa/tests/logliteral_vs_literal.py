
from matplotlib import pyplot as plt
import numpy as np
import torch
from lightning import seed_everything

from ..src.pragmatics.naive_literal import NaiveLiteral
from ..src.pragmatics.naive_rsa import NaiveRSA
from ..src.pragmatics.yrsa import YRSA
from ..src.datasets import FindA1Dataset



def main():

    alpha = 2.5
    seed_everything(42, verbose=False)

    dataset = FindA1Dataset(game_size=6, n_rounds=100)
    world = dataset.world
    costs = torch.zeros(len(world["utterances"]), dtype=torch.float32)

    literal = YRSA(world["logprior"], max_depth=0, tolerance=float('inf'))
    yrsa = YRSA(world["logprior"], max_depth=float('inf'), tolerance=1e-3)

    naive_literal = NaiveLiteral(
        meanings_A=world["meanings_A"], 
        meanings_B=world["meanings_B"], 
        categories=world["targets"], 
        utterances=world["utterances"], 
        lexicon_A=world["lexicon_A"].numpy().astype(float), 
        lexicon_B=world["lexicon_B"].numpy().astype(float), 
        prior=torch.exp(world["logprior"]).numpy().astype(float), 
        costs=costs.numpy().astype(float),
        alpha=alpha,
        update_lexicon=False,
    )

    naive_yrsa = NaiveRSA(
        meanings_A=world["meanings_A"], 
        meanings_B=world["meanings_B"], 
        categories=world["targets"], 
        utterances=world["utterances"], 
        lexicon_A=world["lexicon_A"].numpy().astype(float), 
        lexicon_B=world["lexicon_B"].numpy().astype(float), 
        prior=torch.exp(world["logprior"]).numpy().astype(float), 
        costs=costs.numpy().astype(float),
        alpha=alpha,
        update_lexicon=False,
        max_depth=float('inf'), 
        tolerance=1e-3,
    )


    sample = dataset[0]
    meaning_A = world["meanings_A"][sample["meaning_A"].item()]
    meaning_B = world["meanings_B"][sample["meaning_B"].item()]

    # Set the speaker name
    spk_name = "A"
    model = yrsa
    naive_model = naive_yrsa
    
    # Run model
    lit_logspk = torch.log(world[f"lexicon_{spk_name}"]).T
    # lit_logspk = torch.log_softmax(lit_logspk, dim=1)
    model.reset()
    model.run_turn(lit_logspk, spk_name, costs, alpha)

    # Run naive literal
    naive_model.reset(meaning_A, meaning_B)
    naive_model.run_turn(speaker=spk_name)


    # print("Literal log probabilities:")
    listener = torch.exp(model.turns[-1].listener.as_tensor).numpy().astype(float)
    # print(torch.exp(literal.turns[-1].listener.as_tensor).numpy())
    # print()

    # print("Naive Literal log probabilities:")
    listener_naive = naive_model.turns_history[-1].listener.as_array
    # print(naive_literal.turns_history[-1].listener.as_array)
    # print()

    # plot histogram of abs error
    fig, ax = plt.subplots(1, 1)
    abs_error = np.abs(listener - listener_naive).flatten()
    ax.hist(abs_error, bins=50, density=False)
    plt.savefig("abs_error_histogram.png")
    plt.close(fig)



if __name__ == "__main__":
    main()


from itertools import cycle
import numpy as np

from ..src.naive_models import NaiveCRSA
from ..src.datasets import FindA1Dataset


def main():
    np.random.seed(1234)
    game_size = 4
    dataset = FindA1Dataset(game_size=game_size, n_samples=10)
    world = dataset.world
    
    idx, meaning_A, meaning_B, utterances, cat = next(iter(dataset.iter_samples()))
    
    model = NaiveCRSA(
        meanings_A=world["meanings_A"], 
        meanings_B=world["meanings_B"], 
        categories=world["categories"], 
        utterances=world["utterances"], 
        lexicon_A=world["lexicon_A"], 
        lexicon_B=world["lexicon_B"], 
        prior=world["prior"], 
        costs=world["costs"], 
        alpha=2.5, 
        max_depth=float("inf"),
        tolerance=1e-3,
    )
    model.reset(meaning_A, meaning_B)
    for turn, speaker in zip(range(1, game_size + 1), cycle("AB")):
        model.run_turn(speaker)
        cat_dist = model.get_category_distribution()
        print(f"Turn {turn} - Speaker: {speaker}")
        print(f"Category distribution: {cat_dist}")
        print()


if __name__ == "__main__":
    main()
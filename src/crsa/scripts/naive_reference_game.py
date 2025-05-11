
import argparse
from itertools import cycle
import logging
from pathlib import Path
import time
from typing import List

import numpy as np
import pandas as pd

from ..src import FindA1Dataset
from ..src import compute_metric
from ..src import plot_turns
from ..src import CRSA
from ..src import RSA
from ..src import Literal
from ..src import init_logger



def check_iter_args(alpha, max_depth, tolerance):
    if max_depth is None and tolerance is None:
        raise ValueError("Either max_depth or tolerance must be provided.")
    if max_depth is None and tolerance is not None:
        max_depth = float("inf")
    if max_depth is not None and tolerance is None:
        tolerance = 0.
    return alpha, max_depth, tolerance
    

def init_model(model_name: str, world: dict, alpha: float = 1.0, max_depth: int = None, tolerance: float = None):
    alpha, max_depth, tolerance = check_iter_args(alpha, max_depth, tolerance)
    if model_name == "crsa":    
        model = CRSA(
            meanings_A=world["meanings_A"], 
            meanings_B=world["meanings_B"], 
            categories=world["categories"], 
            utterances=world["utterances"], 
            lexicon_A=world["lexicon_A"], 
            lexicon_B=world["lexicon_B"], 
            prior=world["prior"], 
            costs=world["costs"], 
            alpha=alpha, 
            max_depth=max_depth,
            tolerance=tolerance,
        )
    elif model_name == "rsa":
        model = RSA(
            meanings_A=world["meanings_A"], 
            meanings_B=world["meanings_B"], 
            categories=world["categories"], 
            utterances=world["utterances"], 
            lexicon_A=world["lexicon_A"], 
            lexicon_B=world["lexicon_B"], 
            prior=world["prior"], 
            costs=world["costs"],
            alpha=alpha,
            max_depth=max_depth,
            tolerance=tolerance,
        )
    elif model_name == "literal":
        model = Literal(
            meanings_A=world["meanings_A"], 
            meanings_B=world["meanings_B"], 
            categories=world["categories"], 
            utterances=world["utterances"], 
            lexicon_A=world["lexicon_A"], 
            lexicon_B=world["lexicon_B"], 
            prior=world["prior"], 
            costs=world["costs"],
            alpha=alpha,
        )
    else:
        raise ValueError(f"Model {model_name} not found.")
    return model


def main(
    game_size: int = 3,
    models: List[str] = ["crsa"],
    metrics: List[str] = ["accuracy", "nll"],
    n_turns: int = 10,
    alpha: float = 1.0,
    max_depth: int = None,
    tolerance: float = None,
    n_seeds: int = 1,
    seed: int = 0,
    output_dir: Path = Path("outputs"),
    logger: logging.Logger = None,
):
    
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Init dataset
    dataset = FindA1Dataset(game_size, n_samples=n_seeds)
    
    results = []
    # Run for each model
    for model_name in models:
        # Check if model has already run
        if (output_dir / f"{model_name}_results.csv").exists():
            results.append(pd.read_csv(output_dir / f"{model_name}_results.csv"))
            logger.info(f"Model {model_name} already run. Skipping.")
            continue
        logger.info(f"Running model {model_name}")

        # Initialize the model
        model = init_model(model_name, dataset.world, alpha=alpha, max_depth=max_depth, tolerance=tolerance)

        model_results = []
        # Run the model for each sample
        for i, (idx, meaning_A, meaning_B, utterances, cat) in enumerate(dataset.iter_samples()):
            
            # Log progress
            if i % (len(dataset) // 20) == 0:
                logger.info(f"Running sample {i}/{len(dataset)}")

            cat_idx = dataset.world["categories"].index(cat)
            model.reset(meaning_A, meaning_B)
            # Run the model for each turn
            for turn, speaker in zip(range(1, n_turns + 1), cycle("AB")):
                model.run_turn(speaker)
                cat_dist = model.get_category_distribution()
                model_results.append({
                    "sample_id": idx,
                    "model": model_name,
                    "meaning_A": meaning_A,
                    "meaning_B": meaning_B,
                    **{metric: compute_metric(cat_dist, cat_idx, metric) for metric in metrics},
                    "turn": turn,
                    "speaker": speaker,
                })
        # Convert results to DataFrame
        model_results_df = pd.DataFrame(model_results)
        model_results_df.to_csv(output_dir / f"{model_name}_results.csv", index=False)
        results.append(model_results_df)
    
    # Concatenate all results
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(output_dir / "all_results.csv", index=False)

    # Plot results
    plot_turns(all_results, models, metrics, output_dir)


def parse_args():

    def int_or_inf(value):
        if value.lower() == "inf":
            return float("inf")
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a integer or 'inf'.")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run models for a given configuration file")
    parser.add_argument("--game_size", type=int, help="Number of positions of the Find A1 Game", default=3)
    parser.add_argument("--models", type=str, nargs="+", help="Models to run", default=["crsa_sample"])
    parser.add_argument("--metrics", type=str, nargs="+", help="Metrics to use", default=["accuracy"])
    parser.add_argument("--n_turns", type=int, help="Number of turns", default=5)
    parser.add_argument("--alpha", type=float, help="Alpha to run CRSA with", default=[1.0])
    parser.add_argument("--max_depth", type=int_or_inf, help="Max depth to run CRSA with", default=None)
    parser.add_argument("--tolerance", type=float, help="Tolerance to run CRSA with", default=None)
    parser.add_argument("--seed", type=int, help="Seed to run CRSA with", default=None)
    parser.add_argument("--n_seeds", type=int, help="Number of seeds to run each model", default=1)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / f"game_size={args.game_size}" / f"n_turns={args.n_turns}" / f"alpha={args.alpha}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = init_logger(output_dir)

    # Update configuration
    main_args = {
        "game_size": args.game_size,
        "models": args.models,
        "metrics": args.metrics,
        "n_turns": args.n_turns,
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "output_dir": output_dir,
        "logger": logger,
    }

    return main_args

if __name__ == '__main__':
    args = parse_args()
    logger = args["logger"]

    try:
        main(**args)
        logger.info("Script finished")
    except Exception as e:
        import traceback
        logger.error(f"Error running script:\n\n{traceback.format_exc()}")
        
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

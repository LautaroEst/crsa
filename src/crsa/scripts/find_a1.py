
import argparse
from itertools import cycle
from pathlib import Path
from typing import List, Literal, Union
from lightning import seed_everything
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..src.io import init_logger, read_yaml
from ..src.datasets import FindA1Dataset
from ..src.pragmatics import init_model
from ..src.speakers import StaticSpeaker, DeductiveSpeaker


def check_iter_args(max_depth, tolerance):
    if isinstance(max_depth, (int, str)):
        max_depth = float(max_depth)
    if isinstance(tolerance, (int, str)):
        tolerance = float(tolerance)

    if max_depth is None and tolerance is None:
        raise ValueError("Either max_depth or tolerance must be provided.")
    elif max_depth is None and isinstance(tolerance, (int, float)):
        max_depth = float("inf")
    elif isinstance(max_depth, (int, str)) and tolerance is None:
        tolerance = 0.
    elif isinstance(max_depth, float) and isinstance(tolerance, float):
        if max_depth <= 0:
            raise ValueError("max_depth must be a positive integer or 'inf'.")
        if tolerance < 0:
            raise ValueError("tolerance must be a non-negative number.")
    else:
        raise ValueError("Invalid combination of max_depth and tolerance.")
    return max_depth, tolerance


def plot_turns(df, models, output_dir):
    for model_name in models:
        model_results = df[df['model'] == model_name].copy()
        model_results["nll"] = model_results.apply(lambda x: -np.log(x["prag_list"][x["utterance"], x[f"meaning_{'B' if x['speaker'] == 'A' else 'A'}"],x["target"]]), axis=1)
        model_results["acc"] = model_results.apply(lambda x: float(x["prag_list"][x["utterance"], x[f"meaning_{'B' if x['speaker'] == 'A' else 'A'}"],:].argmax() == x["target"].item()), axis=1)
        print(model_results.groupby("turn").agg({"nll": "mean", "acc": "mean"}).reset_index())
        
def main(
    game_size: int = 3,
    n_rounds: int = 100,
    models: List[str] = ["crsa"],
    alpha: float = 1.0,
    max_depth: Union[int,Literal['inf']] = None,
    tolerance: float = None,
    sampling_strategy: str = "greedy",
    seed: int = 0,
    **kwargs
):
    
    # Validate iteration parameters
    max_depth, tolerance = check_iter_args(max_depth, tolerance)

    # Set up the logger and output directory
    output_dir = kwargs["output_dir"]
    logger = kwargs["logger"]

    # Set random seed for reproducibility
    seed_everything(seed, verbose=False)

    # Model is the combination of literal speaker and pragmatic model
    results = []
    for model_name in models:

        # Check if model has already run
        if (output_dir / f"{model_name}_results.pkl").exists():
            results.append(pd.read_pickle(output_dir / f"{model_name}_results.pkl"))
            logger.info(f"Model {model_name} already run. Skipping.")
            continue
        logger.info(f"Running model {model_name}")

        # Init dataset
        dataset = FindA1Dataset(game_size, n_rounds)

        # Init literal speaker
        if "_wm" in model_name:
            speaker = DeductiveSpeaker.from_lexicon(
                dataset.world["lexicon_A"], dataset.world["lexicon_B"]
            )
        else:
            speaker = StaticSpeaker.from_lexicon(
                dataset.world["lexicon_A"], dataset.world["lexicon_B"]
            )

        # Init pragmatic model
        model = init_model(model_name.split("_")[0], dataset.world["logprior"])       

        # Generate pragmatic dialogs
        model_results = []
        for i, sample in enumerate(dataset.iter_samples()):

            # Get the meanings and target
            meaning_A = sample["meaning_A"]
            meaning_B = sample["meaning_B"]
            target = sample["target"]

            # Generate dialog with the model
            past_utterances = []
            model.reset()
            for turn, spk_name in zip(range(1, game_size + 1), cycle("AB")):

                # Log the current turn
                logger.info(f"Round {i+1}/{len(dataset)}, Turn {turn}, Speaker {spk_name}")

                # Get the literal speaker
                lit_logspk, costs = speaker(past_utterances, spk_name)

                # Run the pragmatic model
                prag_spk, prag_list = model.run_turn(lit_logspk, spk_name, costs, alpha, max_depth, tolerance)

                # Sample an utterance from the pragmatic speaker
                meaning_S = meaning_A if spk_name == "A" else meaning_B
                new_utt = model.sample_utterance(meaning_S, sampling_strategy)
                past_utterances.append({"spk_name": spk_name, "utterance": new_utt})

                # Save results
                model_results.append({
                    "model": model_name,
                    "idx": sample["idx"],
                    "turn": turn,
                    "utterance": new_utt,
                    "speaker": spk_name,
                    "meaning_A": meaning_A,
                    "meaning_B": meaning_B,
                    "target": target,
                    "prag_spk": prag_spk,
                    "prag_list": prag_list,
                })
        # Convert results to DataFrame
        model_results_df = pd.DataFrame(model_results)
        model_results_df.to_pickle(output_dir / f"{model_name}_results.pkl")
        results.append(model_results_df)
    
    # Concatenate all results
    results = pd.concat(results, ignore_index=True)

    # Plot results
    plot_turns(results, models, output_dir)


def setup():

    # Parse arguments and read config path
    parser = argparse.ArgumentParser(description="Run models for a given configuration file")
    parser.add_argument("config", type=str, help="Path to the config file")
    config_path = Path("configs") / Path(__file__).stem / (parser.parse_args().config + ".yaml")

    # Read config file
    config = read_yaml(config_path)

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / config_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = init_logger(__name__, output_dir)

    # Include output_dir and logger in config
    config["output_dir"] = output_dir
    config["logger"] = logger

    # Run main
    try:
        logger.info(f"Starting script {Path(__file__).stem}  with config {config_path.stem}")
        main(**config)
        logger.info(f"Script {Path(__file__).stem}  with config {config_path.stem} finished")
    except Exception:
        import traceback
        logger.error(f"Error running script {Path(__file__).stem}  with config {config_path.stem}:\n\n{traceback.format_exc()}")
        
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    setup()
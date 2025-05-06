import argparse
from pathlib import Path
import pickle
import sys
from typing import List, Optional, Tuple
import shutil
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from ..src.crsa import CRSA
from ..src.utils import read_config_file

def sample_from_prior(prior, meanings_A, meanings_B, y):
    norm_prior = np.array(prior) / np.sum(prior)
    flat_p = norm_prior.flatten()
    sampled_index = np.random.choice(len(flat_p), p=flat_p)
    sampled_indices = np.unravel_index(sampled_index, norm_prior.shape)
    sampled_meaning_A = meanings_A[sampled_indices[0]]
    sampled_meaning_B = meanings_B[sampled_indices[1]]
    sampled_y = y[sampled_indices[2]]
    return sampled_meaning_A, sampled_meaning_B, sampled_y



def main(
    meanings_A: List[str],
    meanings_B: List[str],
    categories: List[str],
    utterances_A: List[str],
    utterances_B: List[str],
    lexicon_A: List[List[int]],
    lexicon_B: List[List[int]],
    prior: List[List[List[float]]],
    cost_A: List[float],
    cost_B: List[float],
    n_conversations: Optional[int] = 10,
    listener_threshold: Optional[float] = 0.5,
    max_turns: Optional[int] = 10,
    alpha: Optional[float] = 1.0,
    max_depth: Optional[int] = None,
    tolerance: Optional[float] = None,
    seed: Optional[int] = None,
    output_dir: Path = Path("outputs"),
    verbose: bool = False
):
    
    np.random.seed(seed)

    # Configure logging
    script_logger = logging.getLogger(__name__)
    console = logging.StreamHandler()
    script_logger.addHandler(console)
    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    file_handler = logging.FileHandler(output_dir / f"{now}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    script_logger.addHandler(file_handler)
    script_logger.setLevel(logging.INFO)

    # Check if RSA is possible
    if max_depth is None and tolerance is None:
        script_logger.error("Either max_depth or tolerance must be provided.")
        sys.exit(1)
    if max_depth is None and tolerance is not None:
        max_depth = float("inf")
    if max_depth is not None and tolerance is None:
        tolerance = 0.

    # Create output directory
    suboutput_dir = output_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}" / f"seed={seed}"
    if suboutput_dir.exists():
        script_logger.warning(f"Experiment already run for alpha={alpha}, max_depth={max_depth}, tolerance={tolerance} and seed={seed}. Skipping.")
        return
    else:
        script_logger.info(f"Running experiment for alpha={alpha}, max_depth={max_depth}, tolerance={tolerance} and seed={seed}.")
    suboutput_dir.mkdir(parents=True, exist_ok=True)


    games = []
    for n in range(n_conversations):

        # Configure logging
        conversation_logger = logging.getLogger(f"conversation_{n:2d}")
        formatter = logging.Formatter("%(message)s")
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            conversation_logger.addHandler(console_handler)
        if output_dir is not None:
            file_handler = logging.FileHandler(suboutput_dir / f"conversation_{n:2d}.log", mode="w", encoding="utf-8")
            file_handler.setFormatter(formatter)
            conversation_logger.addHandler(file_handler)
        conversation_logger.setLevel(logging.INFO)

        meaning_A, meaning_B, y = sample_from_prior(prior, meanings_A, meanings_B, categories)
        
        conversation_logger.info(f"Running conversation {n + 1} of {n_conversations}.")
        conversation_logger.info(f"Sampled meanings: {meaning_A}, {meaning_B}. Sampled category: {y}.")

        past_utterances = []
        speaker_now = "A"
        for turn in range(1,max_turns+1):

            conversation_logger.info(f"Turn {turn} of conversation {n + 1}.")
            
            model = CRSA(
                speaker_now=speaker_now,
                meanings_A=meanings_A,
                meanings_B=meanings_B,
                categories=categories,
                utterances_A=utterances_A,
                utterances_B=utterances_B,
                lexicon_A=lexicon_A,
                lexicon_B=lexicon_B,
                prior=prior,
                past_utterances=past_utterances,
                cost_A=cost_A,
                cost_B=cost_B,
                alpha=alpha,
                max_depth=max_depth,
                tolerance=tolerance,
            )
            model.run()
            last_model_turn = model.turns_history[-1]

            # Sample new utterance
            meaning_S = meaning_A if turn % 2 == 1 else meaning_B
            speaker = last_model_turn.speaker.as_df
            utt_dist = speaker.loc[meaning_S,:].squeeze()
            new_utt = utt_dist.sample(n=1, weights=utt_dist.values).index[0]
            
            # sample listener distribution
            meaning_L = meaning_B if turn % 2 == 1 else meaning_A
            listener = last_model_turn.listener.as_df
            listener_dist = listener.loc[(new_utt,meaning_L),:].squeeze()
            past_utterances.append({"speaker": speaker_now, "utterance": new_utt})
            
            # Log the conversation
            conversation_logger.info(f"Agent {speaker_now} speaks. Distribution: {utt_dist.to_dict()}. Sampled utterance: {new_utt}.")
            conversation_logger.info(f"Agent {'A' if speaker_now == 'B' else 'B'} listens. Distribution: {listener_dist.to_dict()}.")
            
            # Change speaker
            speaker_now = "B" if speaker_now == "A" else "A"

            if listener_dist.max() > listener_threshold:
                break

         # Close logging
        for handler in conversation_logger.handlers:
            handler.close()
            conversation_logger.removeHandler(handler)


        games.append({
            "utterances": past_utterances,
            "meaning_A": meaning_A,
            "meaning_B": meaning_B,
            "true_category": y,
            "final_listener_dist": listener_dist,
        })

        with open(suboutput_dir / f"games.pkl", "wb") as f:
            pickle.dump(games, f)

            
    # Close logging
    for handler in script_logger.handlers:
        handler.close()
        script_logger.removeHandler(handler)


def setup():

    def int_or_inf(value):
        if value.lower() == "inf":
            return float("inf")
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a integer or 'inf'.")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run CRSA for a given configuration file")
    parser.add_argument("--world", type=str, help="Configuration file")
    parser.add_argument("--n_conversations", type=int, help="Number of samples to run", default=10)
    parser.add_argument("--listener_threshold", type=float, help="Listener threshold to run CRSA with", default=0.5)
    parser.add_argument("--max_turns", type=int, help="Max turns to run CRSA with", default=10)
    parser.add_argument("--alpha", type=float, help="Alpha to run CRSA with", default=[1.0])
    parser.add_argument("--max_depth", type=int_or_inf, help="Max depth to run CRSA with", default=None)
    parser.add_argument("--tolerance", type=float, help="Tolerance to run CRSA with", default=None)
    parser.add_argument("--seed", type=int, help="Seed to run CRSA with", default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output", default=False)
    args = parser.parse_args()

    # Read configuration file
    config = read_config_file(f"worlds/{args.world}")

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / args.world
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update configuration
    config["n_conversations"] = args.n_conversations
    config["listener_threshold"] = args.listener_threshold
    config["max_turns"] = args.max_turns
    config["seed"] = args.seed
    config["alpha"] = args.alpha
    config["max_depth"] = args.max_depth
    config["tolerance"] = args.tolerance
    config["output_dir"] = output_dir
    config["verbose"] = args.verbose

    # Save configuration file
    shutil.copy(f"configs/worlds/{args.world}.yaml", output_dir / "config.yaml")

    return config

if __name__ == '__main__':
    config = setup()
    main(**config)


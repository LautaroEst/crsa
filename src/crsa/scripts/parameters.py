
import argparse
from pathlib import Path
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

def plot_history(root_results_dir, past_utterances, alphas, max_depths, tolerances):

    titles2key = [
        ("Conditional entropy", "cond_entropy_history"),
        ("Listener value", "listener_value_history"), 
        ("Gain function", "gain_history"), 
    ]

    turns = len(past_utterances) + 1

    fig, ax = plt.subplots(turns, len(titles2key), figsize=(16, turns*4))
    if turns == 1:
        ax = ax.reshape(1, len(titles2key))

    for i, (alpha, max_depth, tolerance) in enumerate(zip(alphas, max_depths, tolerances)):
        results_dir = root_results_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        model = CRSA.load(results_dir)
        for turn, turn_model in enumerate(model.turns_history):
            for j, (title, key) in enumerate(titles2key):
                values = getattr(turn_model.gain, key)
                ax[turn,j].plot(values, color=f"C{i}")
                ax[turn,j].grid(True)

    for turn in range(turns):
        ax[turn,0].set_ylabel(f"Turn {turn+1}")
    for j, (title, key) in enumerate(titles2key):
        ax[0,j].set_title(title)
        ax[-1,j].set_xlabel("Iteration")

    # Unified legend and title
    fig.suptitle(f"CRSA with past utterances: {', '.join(past_utterances)}", fontsize=16)
    ax[0,-1].legend([f"$\\alpha={alpha},\,D_{{max}}={max_depth},\,tol={tolerance:.2g}$" for (alpha, max_depth, tolerance) in zip(alphas, max_depths, tolerances)], loc="upper right", bbox_to_anchor=(1.8, 1))
    fig.tight_layout(pad=2.3)
    plt.savefig(root_results_dir / "asymptotic_analysis.pdf")


def plot_initial_final(root_results_dir, past_utterances, alphas, max_depths, tolerances):

    turns = len(past_utterances) + 1

    fig, ax = plt.subplots(turns, len(alphas)+1, figsize=(16, turns*5))
    vmin, vmax = 0, 1
    if turns == 1:
        ax = ax.reshape(1, len(alphas)+1)

    for i, (alpha, max_depth, tolerance) in enumerate(zip(alphas, max_depths, tolerances)):
        results_dir = root_results_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        model = CRSA.load(results_dir)
                
        for turn, turn_model in enumerate(model.turns_history):

            if i == 0:
                # Plot initial listener
                literal_listener = turn_model.listener.literal_listener_as_df
                sns.heatmap(literal_listener, ax=ax[turn,0], cmap='viridis', vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cbar=False, yticklabels=literal_listener.index)
            
            # Plot final listener
            final_listener = turn_model.listener.as_df
            sns.heatmap(final_listener, ax=ax[turn,i+1], cmap='viridis', vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cbar=False, yticklabels=final_listener.index)

    for j, name in enumerate(["Literal listener"] + [f"alpha={alpha}" for alpha in alphas]):
        ax[0,j].set_title(name)
        ax[-1,j].set_xlabel("Meanings")

    for turn in range(turns):
        ax[turn,0].set_ylabel(f"Turn {turn+1}")

    # Unified legend and title
    fig.suptitle(f"CRSA with past utterances: {', '.join(past_utterances)}", fontsize=16)
    fig.tight_layout(pad=2.3)
    plt.savefig(root_results_dir / "initial_final.pdf")
        


def main(
    meanings_A: List[str],
    meanings_B: List[str],
    categories: List[str],
    utterances_A: List[str],
    utterances_B: List[str],
    lexicon_A: List[List[int]],
    lexicon_B: List[List[int]],
    prior: List[List[List[float]]],
    pasts: List[str],
    cost_A: List[float],
    cost_B: List[float],
    alphas: List[float] = [1.0],
    max_depths: Optional[List[int]] = None,
    tolerances: Optional[List[float]] = None,
    output_dir: Path = Path("outputs"),
    verbose: bool = False
):
    # Configure logging
    logger = logging.getLogger(__name__)
    console = logging.StreamHandler()
    logger.addHandler(console)
    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    file_handler = logging.FileHandler(output_dir / f"{now}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Check if RSA is possible
    if max_depths is None and tolerances is None:
        logger.error("Either max_depths or tolerances must be provided.")
        sys.exit(1)
    if max_depths is None and tolerances is not None:
        max_depths = [float("inf")] * len(tolerances)
    if max_depths is not None and tolerances is None:
        tolerances = [0.] * len(max_depths)
    if len(alphas) != len(max_depths) or len(alphas) != len(tolerances):
        logger.error("Alphas, max_depths and tolerances must have the same length.")
        sys.exit(1)

    # Run RSA for each alpha and depth
    pasts.insert(0, "")
    for past_utterances in pasts:

        # Split past_utterances
        past_utterances = past_utterances.split(" ") if past_utterances != "" else []

        # Create past utterances string
        past_string = "_".join(past_utterances)

        for alpha, max_depth, tolerance in zip(alphas, max_depths, tolerances):

            # Create output directory
            suboutput_dir = output_dir / f"past_utterances={past_string}" / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
            if suboutput_dir.exists():
                logger.warning(f"Experiment already run for alpha={alpha}, max_depth={max_depth}, tolerance={tolerance} and past={past_string}. Skipping.")
                continue
            else:
                logger.info(f"Running experiment for alpha={alpha}, max_depth={max_depth}, tolerance={tolerance} and past={past_string}.")
            suboutput_dir.mkdir(parents=True, exist_ok=True)

            # Run CRSA
            model = CRSA(
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
            model.run(suboutput_dir, verbose)
            model.save(suboutput_dir)

        # Plot history
        logger.info("Plotting training history for past=%s.", past_string)
        plot_history(output_dir / f"past_utterances={past_string}", past_utterances, alphas, max_depths, tolerances)

        # Plot initial final
        logger.info("Plotting initial and final listeners for past=%s.", past_string)
        plot_initial_final(output_dir / f"past_utterances={past_string}", past_utterances, alphas, max_depths, tolerances)


    # Close logging
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def setup():

    def int_or_inf(value):
        if value.lower() == "inf":
            return float("inf")
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a integer or 'inf'.")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run RSA for a given configuration file")
    parser.add_argument("--world", type=str, help="Configuration file")
    parser.add_argument("--pasts", type=str, nargs="+", help="Past utterances to run RSA with", default=[])
    parser.add_argument("--alphas", type=float, nargs="+", help="Alphas to run RSA with", default=[1.0])
    parser.add_argument("--max_depths", type=int_or_inf, nargs="+", help="Max depths to run RSA with", default=None)
    parser.add_argument("--tolerances", type=float, nargs="+", help="Tolerances to run RSA with", default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output", default=False)
    args = parser.parse_args()

    # Read configuration file
    config = read_config_file(f"worlds/{args.world}")

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / args.world
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update configuration
    config["pasts"] = args.pasts
    config["alphas"] = args.alphas
    config["max_depths"] = args.max_depths
    config["tolerances"] = args.tolerances
    config["output_dir"] = output_dir
    config["verbose"] = args.verbose

    # Save configuration file
    shutil.copy(f"configs/worlds/{args.world}.yaml", output_dir / "config.yaml")

    return config

if __name__ == '__main__':
    config = setup()
    main(**config)


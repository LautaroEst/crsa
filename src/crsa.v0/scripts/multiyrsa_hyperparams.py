
import argparse
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import shutil
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ..src.multi_yrsa import MultiturnYRSA
from ..src.utils import read_config_file

def plot_history(root_results_dir, alphas, max_depths, tolerances):

    titles2key = [
        ("Conditional entropy", "cond_entropy_history"),
        ("Listener value", "listener_value_history"), 
        ("Gain function", "gain_history"), 
        ("Cooperation index", "coop_index_history"),
    ]

    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    for i, (alpha, max_depth, tolerance) in enumerate(zip(alphas, max_depths, tolerances)):
        results_dir = root_results_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        rsa = MultiturnYRSA.load(results_dir)
        for a, (title, key) in zip(ax, titles2key):
            values = getattr(rsa.gain, key)
            a.plot(values, color=f"C{i}")
            a.set_title(title)
            a.set_xlabel("Iteration")
            a.grid()

    # Unified legend
    ax[-1].legend([f"$\\alpha={alpha},\,D_{{max}}={max_depth},\,tol={tolerance:.2g}$" for (alpha, max_depth, tolerance) in zip(alphas, max_depths, tolerances)], loc="upper right", bbox_to_anchor=(1.8, 1))
    fig.tight_layout(pad=2.3)
    plt.savefig(root_results_dir / "asymptotic_analysis.pdf")


def plot_initial_final(root_results_dir, alphas, max_depths, tolerances):
    
    # Read RSA
    results_dir = root_results_dir / f"alpha={alphas[0]}" / f"max_depth={max_depths[0]}_tolerance={tolerances[0]}"
    rsa = MultiturnYRSA.load(results_dir)

    # Init figure
    fig, ax = plt.subplots(1, len(alphas)+1, figsize=(4*(len(alphas)+1), 4))
    vmin, vmax = 0, 1
    
    # Plot initial lexicon
    literal_listener = rsa.listener.get_literal_as_df()
    sns.heatmap(literal_listener, ax=ax[0], cmap='viridis', vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cbar=False, yticklabels=rsa.listener.as_df.index)
    ax[0].set_title(f"Literal listener")

    # Plot final listener for each alpha
    for i, (alpha, max_depth, tolerance) in enumerate(zip(alphas, max_depths, tolerances)):
        results_dir = root_results_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        rsa = MultiturnYRSA.load(results_dir)
        sns.heatmap(rsa.listener.as_df, ax=ax[i+1], cmap='viridis', vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cbar=False, yticklabels=rsa.listener.as_df.index)
        ax[i+1].set_title(f"Final Listener for $\\alpha={alpha}$")
        if i > 0:
            ax[i+1].set_ylabel(None)
            ax[i+1].set_yticklabels([])

    fig.tight_layout()
    plt.savefig(root_results_dir / "initial_final.pdf")
    plt.close(fig)



def main(
    meanings_A: List[str],
    meanings_B: List[str],
    categories: List[str],
    utterances_A: List[str],
    utterances_B: List[str],
    lexicon_A: List[Tuple[str,List[int]]],
    lexicon_B: List[Tuple[str,List[int]]],
    prior: List[List[List[float]]],
    cost_A: List[float],
    cost_B: List[float],
    turns: float,
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
    for alpha, max_depth, tolerance in zip(alphas, max_depths, tolerances):

        # Create output directory
        suboutput_dir = output_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        if suboutput_dir.exists():
            logger.warning(f"Experiment already run for alpha={alpha}, max_depth={max_depth} and tolerance={tolerance}. Skipping.")
            continue
        else:
            logger.info(f"Running experiment for alpha={alpha}, max_depth={max_depth} and tolerance={tolerance}.")
        suboutput_dir.mkdir(parents=True, exist_ok=True)

        # Run Multiturn Y-RSA
        rsa = MultiturnYRSA(meanings_A, meanings_B, categories, utterances_A, utterances_B, lexicon_A, lexicon_B, prior, cost_A, cost_B, alpha, max_depth, tolerance, turns)
        rsa.run(suboutput_dir, verbose)
        rsa.save(suboutput_dir)

    # Plot training history
    # logger.info("Plotting training history.")
    # plot_history(output_dir, alphas, max_depths, tolerances)

    # Plot initial lexicon and final listener for each alpha
    # logger.info("Plotting literal and final listener.")
    # plot_initial_final(output_dir, alphas, max_depths, tolerances)

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

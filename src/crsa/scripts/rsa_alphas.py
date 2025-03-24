
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..src.rsa import RSA

def plot_history(root_results_dir, alphas, max_depths, tolerances):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    for i, (alpha, max_depth, tolerance) in enumerate(zip(alphas, max_depths, tolerances)):
        results_dir = root_results_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        rsa = RSA.load(results_dir)
        ax[0].plot(rsa.gain.cond_entropy_history, color=f"C{i}")
        ax[1].plot(rsa.gain.listener_value_history, color=f"C{i}")
        ax[2].plot(rsa.gain.gain_history, color=f"C{i}")
        ax[3].plot(rsa.gain.coop_index_history, color=f"C{i}")
    ax[0].set_title("Conditional entropy")
    ax[1].set_title("Listener value")
    ax[2].set_title("Gain function")
    ax[3].set_title("Cooperation index")

    for a in ax:
        a.set_xlabel("Iteration")
        a.set_xticks(range(0, len(rsa.gain.cond_entropy_history), len(rsa.gain.cond_entropy_history) // 10))
        a.grid()

    # Unified legend
    ax[-1].legend([f"$\\alpha={alpha}$" for alpha in alphas], loc="upper right", bbox_to_anchor=(1.3, 1))
    fig.tight_layout(pad=1.8)
    plt.savefig(root_results_dir / "asymptotic_analysis.png")


def plot_initial_final(root_results_dir, alphas, max_depths, tolerances):
    
    # Read RSA
    results_dir = root_results_dir / f"alpha={alphas[0]}" / f"max_depth={max_depths[0]}_tolerance={tolerances[0]}"
    rsa = RSA.load(results_dir)

    # Init figure
    fig, ax = plt.subplots(1, len(alphas)+1, figsize=(4*(len(alphas)+1), 4))
    vmin, vmax = 0, 1
    
    # Plot initial lexicon
    lexicon = pd.DataFrame(rsa.lexicon, index=rsa.utterances, columns=rsa.meanings)
    sns.heatmap(lexicon, ax=ax[0], cmap='viridis', vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cbar=False)
    ax[0].set_title(f"Initial Lexicon")

    # Plot final listener for each alpha
    for i, (alpha, max_depth, tolerance) in enumerate(zip(alphas, max_depths, tolerances)):
        results_dir = root_results_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        rsa = RSA.load(results_dir)
        sns.heatmap(rsa.listener.as_df, ax=ax[i+1], cmap='viridis', vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cbar=False)
        ax[i+1].set_title(f"Final Listener for $\\alpha={alpha}$")
        ax[i+1].set_yticklabels([])

    fig.tight_layout()
    plt.savefig(root_results_dir / "initial_final.png")

def main(root_results_dir, alphas, max_depths, tolerances):

    # Check if RSA is possible
    if max_depths is None and tolerances is None:
        raise ValueError("Either max_depths or tolerances must be provided.")
    if max_depths is None and tolerances is not None:
        max_depths = [float("inf")] * len(tolerances)
    if max_depths is not None and tolerances is None:
        tolerances = [0.] * len(max_depths)
    if len(alphas) != len(max_depths) or len(alphas) != len(tolerances):
        raise ValueError("Alphas, max_depths and tolerances must have the same length.")
    
    # Plot training history
    plot_history(root_results_dir, alphas, max_depths, tolerances)

    # Plot initial lexicon and final listener for each alpha
    plot_initial_final(root_results_dir, alphas, max_depths, tolerances)
    



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
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("--alphas", type=float, nargs="+", help="Alphas to run RSA with", default=[1.0])
    parser.add_argument("--max_depths", type=int_or_inf, nargs="+", help="Max depths to run RSA with", default=None)
    parser.add_argument("--tolerances", type=float, nargs="+", help="Tolerances to run RSA with", default=None)
    args = parser.parse_args()

    # Create configuration
    config = {
        "root_results_dir": Path("outputs/rsa") / args.results_dir,
        "alphas": args.alphas,
        "max_depths": args.max_depths,
        "tolerances": args.tolerances,
    }

    return config

if __name__ == '__main__':
    config = setup()
    main(**config)
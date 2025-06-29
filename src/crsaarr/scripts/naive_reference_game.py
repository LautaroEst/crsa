
import argparse
from copy import deepcopy
from itertools import cycle
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..src.naive_models import NaiveCRSA, NaiveRSA, NaiveLiteral, Prior
from ..src.datasets import FindA1Dataset
from ..src.evaluate import compute_metric
from ..src.utils import init_logger


model2config = {
    "crsa_wm": {"label": "CRSA-$W_t$", "color": "tab:blue", "linestyle": ":"},
    "crsa": {"label": "CRSA", "color": "tab:blue", "linestyle": "-"},
    "rsa_wm": {"label": "YRSA-$W_t$", "color": "tab:orange", "linestyle": ":"},
    "rsa": {"label": "YRSA", "color": "tab:orange", "linestyle": "-"},
    "literal_wm": {"label": "Literal-$W_t$", "color": "tab:green", "linestyle": ":"},
    "literal": {"label": "Literal", "color": "tab:green", "linestyle": "-"},
    "prior": {"label": "Prior", "color": "tab:red", "linestyle": "-"},
}


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
    if model_name in ["crsa", "crsa_wm"]:    
        model = NaiveCRSA(
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
            update_lexicon=model_name == "crsa_wm",
        )
    elif model_name in ["rsa", "rsa_wm"]:
        model = NaiveRSA(
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
            update_lexicon=model_name == "rsa_wm"
        )
    elif model_name in ["literal", "literal_wm"]:
        model = NaiveLiteral(
            meanings_A=world["meanings_A"], 
            meanings_B=world["meanings_B"], 
            categories=world["categories"], 
            utterances=world["utterances"], 
            lexicon_A=world["lexicon_A"], 
            lexicon_B=world["lexicon_B"], 
            prior=world["prior"], 
            costs=world["costs"],
            alpha=alpha,
            update_lexicon=model_name == "literal_wm",
        )
    elif model_name == "prior":
        model = Prior(
            meanings_A=world["meanings_A"], 
            meanings_B=world["meanings_B"], 
            categories=world["categories"], 
            prior=world["prior"], 
        )
    else:
        raise ValueError(f"Model {model_name} not found.")
    return model


def plot_turns(df, models, output_dir, plot_error_bars=False):

    metrics = [
        ("accuracy", "Accuracy of the listener"),
        ("igain", "Information Gain: $H_P(Y|M_{L_t})-H_L(Y|U_t,M_{L_t},W_t)$"),
    ]

    turns = df["turn"].unique()

    results = []
    for turn in turns:
        prior_probs = np.vstack(df.loc[(df["turn"] == turn) & (df["model"] == "prior"), "category_distribution"].values)
        prior_targets = df.loc[(df["turn"] == turn) & (df["model"] == "prior"), "category_idx"].values
        for model in models:
            probs = np.vstack(df.loc[(df["turn"] == turn) & (df["model"] == model), "category_distribution"].values)
            target = df.loc[(df["turn"] == turn) & (df["model"] == model), "category_idx"].values
            metrics_results = {}
            for metric, _ in metrics:
                mean, std = compute_metric(probs, target, metric, prior=prior_probs, prior_target=prior_targets)
                metrics_results[f"{metric}:mean"] = mean
                metrics_results[f"{metric}:std"] = std
            results.append({
                "model": model,
                "turn": turn,
                **metrics_results,
            })
    results = pd.DataFrame(results)
    # print(results.set_index(["model", "turn"]).loc[("crsa", 5),:])
    # import pdb; pdb.set_trace()

    fig, ax = plt.subplots(2, 1, figsize=(6, 12))
    for i, (metric, metric_name) in enumerate(metrics):
        for model in models:
            model_df = results[results["model"] == model].sort_values("turn")
            ax[i].plot(model_df["turn"], model_df[f"{metric}:mean"], label=model2config[model]["label"], linestyle=model2config[model]["linestyle"], linewidth=3, color=model2config[model]["color"], marker="o", markersize=8)
            if plot_error_bars:
                if metric == "accuracy":
                    yerr_top = np.clip(model_df[f"{metric}:mean"] + model_df[f"{metric}:std"], 0, 1) - model_df[f"{metric}:mean"]
                    yerr_bottom = np.clip(model_df[f"{metric}:mean"] - model_df[f"{metric}:std"], 0, 1) + model_df[f"{metric}:mean"]
                else:
                    yerr_top = model_df[f"{metric}:std"]
                    yerr_bottom = model_df[f"{metric}:std"]
                ax[i].errorbar(
                    model_df["turn"], model_df[f"{metric}:mean"], yerr=[yerr_bottom, yerr_top],
                    fmt="none", color=model2config[model]["color"], capsize=5, capthick=2, elinewidth=2, alpha=0.5,
                )
        ax[i].set_title(metric_name, fontsize=14)
        ax[i].set_xlabel("Turn")
        ax[i].grid(True)
        ax[i].set_xticks(model_df["turn"].astype(int))
    ax[0].set_ylim(0, 1.05)
    ax[-1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), fontsize=12, ncol=4)
    # ax[0].legend(loc="upper left", fontsize=12, bbox_to_anchor=(1, 1))
    fig.tight_layout(pad=1)
    plt.savefig(output_dir / f"metrics_vs_turns.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_belief_example(df, sample_id, meanings, output_dir):
    sample = df.loc[(df["sample_id"] == sample_id) & (df["model"] == "crsa"), ["turn", "speaker", "belief_A", "belief_B"]].copy()
    sample = sample.set_index("turn").sort_index()
    turns = sample.index.to_numpy()
    round_meaning_A = df.loc[(df["sample_id"] == sample_id) & (df["model"] == "crsa"), "meaning_A"].values[0]
    round_meaning_B = df.loc[(df["sample_id"] == sample_id) & (df["model"] == "crsa"), "meaning_B"].values[0]
    dialog = df.loc[(df["sample_id"] == sample_id) & (df["model"] == "crsa"), ["turn","speaker","sampled_utt"]].set_index("turn").sort_index().copy()
    
    # Belief
    listener_belief = []
    listener_meanings = []
    for turn, (spk, belief_A, belief_B) in sample.iterrows():
        belief_L = belief_B if spk == "A" else belief_A
        belief_L = belief_L / np.sum(belief_L)
        listener_belief.append(belief_L)
        listener_meanings.append(meanings["B"] if spk == "A" else meanings["A"])
    listener_belief = np.vstack(listener_belief).T
    listener_meanings = np.array(listener_meanings).T
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    sns.heatmap(
        listener_belief, ax=ax, cmap="viridis", cbar=True, linewidths=0, linecolor=None, 
        annot=listener_meanings, fmt="s", cbar_kws={"aspect": 50}
    )
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks(turns.astype(int) - 0.5)
    ax.set_xticklabels(
        [f"Turn {turn}\n$S_{spk}$: {utt}" for turn, (spk,utt) in dialog.iterrows()], rotation=0, fontsize=8)
    ax.set_title(f"Speaker Belief\n$m_A={round_meaning_A},m_B={round_meaning_B}$", fontsize=14)

    fig.tight_layout()
    plt.savefig(output_dir / f"belief.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main(
    game_size: int = 3,
    models: List[str] = ["crsa"],
    alpha: float = 1.0,
    max_depth: int = None,
    tolerance: float = None,
    n_seeds: int = 1,
    seed: int = 0,
    plot_error_bars: bool = False,
    output_dir: Path = Path("outputs"),
    logger: logging.Logger = None,
):
    
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Init dataset
    dataset = FindA1Dataset(game_size, n_samples=n_seeds)
    
    results = []
    # Run for each model
    models_to_run = deepcopy(models)
    if "prior" not in models_to_run:
        models_to_run.append("prior")
    for model_name in models_to_run:
        # Check if model has already run
        if (output_dir / f"{model_name}_results.pkl").exists():
            results.append(pd.read_pickle(output_dir / f"{model_name}_results.pkl"))
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

            model.reset(meaning_A, meaning_B)
            cat_idx = dataset.world["categories"].index(cat)
            # Run the model for each turn
            for turn, speaker in zip(range(1, game_size + 1), cycle("AB")):
                model.run_turn(speaker)
                cat_dist = model.get_category_distribution()
                if model_name == "crsa":
                    belief_A = model.belief_A if model.belief_A is not None else np.ones(len(dataset.world["meanings_A"])) / len(dataset.world["meanings_A"])
                    belief_B = model.belief_B if model.belief_B is not None else np.ones(len(dataset.world["meanings_B"])) / len(dataset.world["meanings_B"])
                else:
                    belief_A = None
                    belief_B = None
                model_results.append({
                    "sample_id": idx,
                    "model": model_name,
                    "meaning_A": meaning_A,
                    "meaning_B": meaning_B,
                    "turn": turn,
                    "sampled_utt": model.past_utterances[-1]["utterance"] if model_name != "prior" else None,
                    "speaker": speaker,
                    "category_idx": cat_idx,
                    "category_distribution": cat_dist,
                    "belief_A": belief_A,
                    "belief_B": belief_B,
                })
        # Convert results to DataFrame
        model_results_df = pd.DataFrame(model_results)
        model_results_df.to_pickle(output_dir / f"{model_name}_results.pkl")
        results.append(model_results_df)
    
    # Concatenate all results
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_pickle(output_dir / "all_results.pkl")

    # Plot results
    plot_turns(all_results, models, output_dir, plot_error_bars=plot_error_bars)

    sample_id = 0
    meanings = {
        "A": dataset.world["meanings_A"],
        "B": dataset.world["meanings_B"],
    }
    plot_belief_example(all_results, sample_id, meanings, output_dir)



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
    parser.add_argument("--alpha", type=float, help="Alpha to run CRSA with", default=[1.0])
    parser.add_argument("--max_depth", type=int_or_inf, help="Max depth to run CRSA with", default=None)
    parser.add_argument("--tolerance", type=float, help="Tolerance to run CRSA with", default=None)
    parser.add_argument("--seed", type=int, help="Seed to run CRSA with", default=None)
    parser.add_argument("--n_seeds", type=int, help="Number of seeds to run each model", default=1)
    parser.add_argument("--plot_error_bars", action="store_true", help="Plot error bars in the plots")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / f"game_size={args.game_size}" / f"alpha={args.alpha}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = init_logger(__name__, output_dir)

    # Update configuration
    main_args = {
        "game_size": args.game_size,
        "models": args.models,
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "output_dir": output_dir,
        "plot_error_bars": args.plot_error_bars,
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




import argparse
from itertools import cycle
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import List, Optional
from tqdm import trange

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def init_model(model_name, model_args):
    if model_name in ["crsa_sample", "crsa_max"]:
        model = CRSA(**model_args)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model

def sample_new_utterance(model_name, meaning_S, last_turn_model):
    speaker = last_turn_model.speaker.as_df
    utt_dist = speaker.loc[meaning_S,:].squeeze()
    if model_name == "crsa_sample":
        new_utt = utt_dist.sample(n=1, weights=utt_dist.values).index[0]
    elif model_name == "crsa_max":
        new_utt = utt_dist.idxmax()
    return new_utt

def run_model_for_n_turns(model_name, model_args, turns, meaning_A, meaning_B):
    
    # Run the first turn
    new_past_utterances = []
    model_args["past_utterances"] = new_past_utterances
    model_args["speaker_now"] = "A"
    model = init_model(model_name, model_args)
    model.run()
    last_speaker = model_args["speaker_now"]
    for turn, speaker_now in zip(range(2, turns + 1),cycle("BA")):

        # last turn
        last_turn_model = model.turns_history[-1]

        # sample a new utterance
        meaning_S = meaning_A if last_speaker == "A" else meaning_B
        new_utt = sample_new_utterance(model_name, meaning_S, last_turn_model)
        new_past_utterances = [{"speaker": last_speaker, "utterance": new_utt}]

        # update the model
        model.continue_from_last_turn(new_past_utterances, speaker_now)
        last_speaker = speaker_now

    # sample last utterance and recreate the conversation
    last_turn_model = model.turns_history[-1]
    meaning_S = meaning_A if last_speaker == "A" else meaning_B
    new_utt = sample_new_utterance(model_name, meaning_S, last_turn_model)
    conversation = model.past_utterances + [{"speaker": last_speaker, "utterance": new_utt}]
    
    # Collect predictions
    predictions = {"turn": [], "speaker": [], "utterance": [], "category_dist": [], "utterance_dist": []}
    for turn, (turn_model, turn_sample) in enumerate(zip(model.turns_history, conversation), 1):
        
        # Get listener's category distribution for each turn
        listener = turn_model.listener.as_df
        meaning_L = meaning_B if turn_sample["speaker"] == "A" else meaning_A
        category_dist = listener.loc[(turn_sample["utterance"], meaning_L),:].squeeze()

        # Get speaker's utterance distribution for each turn
        meaning_S = meaning_A if turn_sample["speaker"] == "A" else meaning_B
        utterance_dist = turn_model.speaker.as_df.loc[meaning_S,:].squeeze()
        predictions["turn"].append(turn)
        predictions["speaker"].append(turn_sample["speaker"])
        predictions["utterance"].append(turn_sample["utterance"])
        predictions["category_dist"].append(category_dist)
        predictions["utterance_dist"].append(utterance_dist)

    return predictions

    
def plot_results(df, models, alpha, max_depth, tolerance, output_dir):
    metrics = ["accuracy", "nll"]
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    for i, metric in enumerate(metrics):
        # aggregate median, q1 and q3
        df_metric = df.groupby(["model","turn"]).agg(**{
            "median": (metric, "median"),
            "q1": (metric, lambda x: x.quantile(0.25)),
            "q3": (metric, lambda x: x.quantile(0.75)),
        }).reset_index()
        for c, model in enumerate(models):
            model_df = df_metric[df_metric["model"] == model].sort_values("turn")
            ax[i].plot(model_df["turn"], model_df[f"median"], label=model, linestyle="--", linewidth=2, color=f"C{c}")
            ax[i].errorbar(
                model_df["turn"], model_df["median"], yerr=[model_df["median"] - model_df["q1"], model_df["q3"] - model_df["median"]], 
                fmt="o", capsize=5, capthick=2, elinewidth=2, markersize=5, color=f"C{c}",
            )
            ax[i].set_ylabel(metric)
            ax[i].set_xlabel("Turn")
            ax[i].grid(True)
        df_metric.to_csv(output_dir / f"{metric}.csv", index=False)
    fig.suptitle(f"Model results over Turns for alpha={alpha}, max_depth={max_depth}, tolerance={tolerance}")
    ax[-1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(output_dir / f"scores.pdf", bbox_inches="tight", dpi=300)
    df.to_csv(output_dir / "results.csv", index=False)
    plt.close(fig)

def compute_metric(category_dist, y, metric="accuracy"):
    if metric == "accuracy":
        return [float(dist.idxmax() == y) for dist in category_dist]
    elif metric == "nll":
        return -np.log([dist[y] for dist in category_dist])
    raise ValueError(f"Metric {metric} not recognized.")


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
    n_turns: Optional[int] = 10,
    models: Optional[List[str]] = ["crsa_sample"],
    alpha: Optional[float] = 1.0,
    max_depth: Optional[int] = None,
    tolerance: Optional[float] = None,
    seed: Optional[int] = None,
    n_seeds: Optional[int] = 1,
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

    # Run models
    model_args = {
        "meanings_A": meanings_A,
        "meanings_B": meanings_B,
        "categories": categories,
        "utterances_A": utterances_A,
        "utterances_B": utterances_B,
        "lexicon_A": lexicon_A,
        "lexicon_B": lexicon_B,
        "prior": prior,
        "cost_A": cost_A,
        "cost_B": cost_B,
        "alpha": alpha,
        "max_depth": max_depth,
        "tolerance": tolerance,
    }
    results = []
    for seed in trange(n_seeds):

        meaning_A, meaning_B, y = sample_from_prior(prior, meanings_A, meanings_B, categories)

        for model_name in models:
            predictions = run_model_for_n_turns(model_name, model_args, n_turns, meaning_A, meaning_B)
            accs = compute_metric(predictions["category_dist"], y, "accuracy")
            nlls = compute_metric(predictions["category_dist"], y, "nll")
            for turn, acc, nll in zip(predictions["turn"], accs, nlls):
                results.append({
                    "model": model_name,
                    "seed": seed,
                    "turn": turn,
                    "accuracy": acc,
                    "nll": nll,
                })
    df = pd.DataFrame(results)

    plot_results(df, models, alpha, max_depth, tolerance, suboutput_dir)
            



def setup():

    def int_or_inf(value):
        if value.lower() == "inf":
            return float("inf")
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a integer or 'inf'.")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run models for a given configuration file")
    parser.add_argument("--world", type=str, help="Configuration file")
    parser.add_argument("--models", type=str, nargs="+", help="Models to run", default=["crsa_sample"])
    parser.add_argument("--n_turns", type=int, help="Number of turns", default=5)
    parser.add_argument("--alpha", type=float, help="Alpha to run CRSA with", default=[1.0])
    parser.add_argument("--max_depth", type=int_or_inf, help="Max depth to run CRSA with", default=None)
    parser.add_argument("--tolerance", type=float, help="Tolerance to run CRSA with", default=None)
    parser.add_argument("--seed", type=int, help="Seed to run CRSA with", default=None)
    parser.add_argument("--n_seeds", type=int, help="Number of seeds to run each model", default=1)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output", default=False)
    args = parser.parse_args()

    # Read configuration file
    config = read_config_file(f"worlds/{args.world}")

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / args.world
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update configuration
    config["models"] = args.models
    config["n_turns"] = args.n_turns
    config["alpha"] = args.alpha
    config["max_depth"] = args.max_depth
    config["tolerance"] = args.tolerance
    config["seed"] = args.seed
    config["n_seeds"] = args.n_seeds
    config["output_dir"] = output_dir
    config["verbose"] = args.verbose

    # Save configuration file
    shutil.copy(f"configs/worlds/{args.world}.yaml", output_dir / "config.yaml")

    return config

if __name__ == '__main__':
    config = setup()
    main(**config)
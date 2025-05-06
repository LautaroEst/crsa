


import argparse
from itertools import cycle
import logging
from pathlib import Path
import shutil
import sys
import time
from typing import List, Optional
from tqdm import tqdm, trange

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..src.crsa import CRSA
from ..src.multiturn_rsa import MultiTurnRSA
from ..src.literal import Literal
from ..src.utils import read_config_file
from ..src.infojigsaw import InfoJigsawDataset

def init_model(model_name, model_args):
    if "crsa" in model_name:
        model = CRSA(**model_args)
    elif "multi_rsa" in model_name:
        model = MultiTurnRSA(**model_args)
    elif "literal" in model_name:
        model = Literal(
            speaker_now=model_args["speaker_now"], 
            meanings_A=model_args["meanings_A"], 
            meanings_B=model_args["meanings_B"], 
            categories=model_args["categories"], 
            utterances_A=model_args["utterances_A"], 
            utterances_B=model_args["utterances_B"], 
            lexicon_A=model_args["lexicon_A"], 
            lexicon_B=model_args["lexicon_B"], 
            prior=model_args["prior"], 
            past_utterances=model_args["past_utterances"], 
            alpha=model_args["alpha"], 
            cost_A=model_args["cost_A"], 
            cost_B=model_args["cost_B"]
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model

def run_model(model_name, model_args, meaning_A, meaning_B, starter_speaker, utterances):
    
    cycle_iterator = cycle("BA") if starter_speaker == "A" else cycle("AB")

    # Run the first turn
    new_past_utterances = []
    model_args["past_utterances"] = new_past_utterances
    model_args["speaker_now"] = starter_speaker
    model = init_model(model_name, model_args)
    model.run()
    last_speaker = model_args["speaker_now"]
    turns = len(utterances)
    for turn, speaker_now, new_utt in zip(range(2, turns + 1),cycle_iterator, utterances):

        # update the model
        new_past_utterances = [{"speaker": last_speaker, "utterance": new_utt}]
        model.continue_from_last_turn(new_past_utterances, speaker_now)
        last_speaker = speaker_now

    # Get last turn
    new_utt = utterances[-1]
    turn_model = model.turns_history[-1]

    # Get listener's category distribution on the last turn
    listener = turn_model.listener.as_df
    meaning_L = meaning_B if last_speaker == "A" else meaning_A
    category_dist = listener.loc[(new_utt, meaning_L),:].squeeze()

    # Get speaker's utterance distribution on the last turn
    meaning_S = meaning_A if last_speaker == "A" else meaning_B
    utterance_dist = turn_model.speaker.as_df.loc[meaning_S,:].squeeze()

    return category_dist, utterance_dist

    
def plot_results(df, models, alpha, max_depth, tolerance, output_dir):
    metrics = ["accuracy", "nll"]
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    for i, metric in enumerate(metrics):
        # aggregate median, q1 and q3
        df_metric = df.groupby(["model","turn"]).agg(**{
            # "median": (metric, "median"),
            # "q1": (metric, lambda x: x.quantile(0.25)),
            # "q3": (metric, lambda x: x.quantile(0.75)),
            "mean": (metric, "mean"),
            "std": (metric, "std"),
        }).reset_index()
        for c, model in enumerate(models):
            model_df = df_metric[df_metric["model"] == model].sort_values("turn")
            # ax[i].plot(model_df["turn"], model_df[f"median"], label=model, linestyle="--", linewidth=2, color=f"C{c}")
            ax[i].plot(model_df["turn"], model_df["mean"], label=model, linestyle="--", linewidth=2, color=f"C{c}")
            # ax[i].errorbar(
            #     model_df["turn"], model_df["median"], yerr=[model_df["median"] - model_df["q1"], model_df["q3"] - model_df["median"]], 
            #     fmt="o", capsize=5, capthick=2, elinewidth=2, markersize=5, color=f"C{c}",
            # )
            ax[i].errorbar(
                model_df["turn"], model_df["mean"], yerr=model_df["std"], 
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
    models: Optional[List[str]] = ["crsa_sample"],
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

    dataset = InfoJigsawDataset(model="clicked_pos")
    world = dataset.world

    # Run models
    model_args = {
        "meanings_A": world["meanings_letter"],
        "meanings_B": world["meanings_number"],
        "categories": world["categories"],
        "utterances_A": world["utterances"],
        "utterances_B": world["utterances"],
        "lexicon_A": world["lexicon_letter"],
        "lexicon_B": world["lexicon_number"],
        "prior": world["prior"],
        "cost_A": None,
        "cost_B": None,
        "alpha": alpha,
        "max_depth": max_depth,
        "tolerance": tolerance,
    }
    results = []
    for idx, (meaning_letter, meaning_number, y, starter_speaker, utterances) in tqdm(dataset.samples(), total=len(dataset)):
        starter_speaker = "A" if starter_speaker == "playerChar" else "B"
        for model_name in models:
            category_dist, utterance_dist = run_model(model_name, model_args, meaning_letter, meaning_number, starter_speaker, utterances)
            cat_acc = compute_metric([category_dist], y, "accuracy")[0]
            cat_nll = compute_metric([category_dist], y, "nll")[0]
            utt_acc = compute_metric([utterance_dist], utterances[-1], "accuracy")[0]
            utt_nll = compute_metric([utterance_dist], utterances[-1], "nll")[0]
            results.append({"model": model_name, "sample_idx": idx, "cat_accuracy": cat_acc, "cat_nll": cat_nll, "utt_accuracy": utt_acc, "utt_nll": utt_nll})

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)

    # plot_results(df, models, alpha, max_depth, tolerance, suboutput_dir)
            



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
    parser.add_argument("--models", type=str, nargs="+", help="Models to run", default=["crsa"])
    parser.add_argument("--alpha", type=float, help="Alpha to run CRSA with", default=[1.0])
    parser.add_argument("--max_depth", type=int_or_inf, help="Max depth to run CRSA with", default=None)
    parser.add_argument("--tolerance", type=float, help="Tolerance to run CRSA with", default=None)
    parser.add_argument("--seed", type=int, help="Seed to run CRSA with", default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output", default=False)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update configuration
    config = {
        "models": args.models,
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "output_dir": output_dir,
        "verbose": args.verbose,
    }

    return config

if __name__ == '__main__':
    config = setup()
    main(**config)
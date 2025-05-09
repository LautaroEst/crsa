import argparse
from itertools import cycle, product
import logging
from pathlib import Path
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..src.crsa import CRSA
from ..src.prior_model import PriorModel
from ..src.memoryless_literal import MemorylessLiteral
from ..src.memoryless_rsa import MemorylessRSA
from ..src.llm_dialog import LLMDialog, LLM
from ..src.llm_rsa import LLMRSA

metric2name = {
    "accuracy": "Average of correct guessings",
    "nll": "Listener Cross-entropy"
}


def model2name(model_name):
    if model_name == "crsa":
        return "CRSA",
    elif model_name == "memoryless_rsa":
        return "RSA on each turn (no history)"
    elif model_name == "memoryless_literal":
        return "Literal model on each turn"
    elif model_name == "prior_model":
        return "Random (prior)"
    elif model_name.startswith("llm_"):
        return f"LLM {model_name[4:]}"
    elif model_name.startswith("llmrsa_"):
        return f"LLM-RSA {model_name[7:]}"
    else:
        raise ValueError(f"Model {model_name} not recognized.")


def sample_from_prior(prior, meanings_A, meanings_B, y):
    norm_prior = np.array(prior) / np.sum(prior)
    flat_p = norm_prior.flatten()
    sampled_index = np.random.choice(len(flat_p), p=flat_p)
    sampled_indices = np.unravel_index(sampled_index, norm_prior.shape)
    sampled_meaning_A = meanings_A[sampled_indices[0]]
    sampled_meaning_B = meanings_B[sampled_indices[1]]
    sampled_y = y[sampled_indices[2]]
    return sampled_meaning_A, sampled_meaning_B, sampled_y

def init_model(model_name, model_args, meaning_A, meaning_B, n_possitions):
    if model_name == "crsa":
        model = CRSA(
            meanings_A=model_args["meanings_A"],
            meanings_B=model_args["meanings_B"],
            categories=model_args["categories"],
            utterances=model_args["utterances"],
            lexicon_A=model_args["lexicon_A"],
            lexicon_B=model_args["lexicon_B"],
            prior=model_args["prior"],
            costs=model_args["costs"],
            alpha=model_args["alpha"],
            pov="listener",
            max_depth=model_args["max_depth"],
            tolerance=model_args["tolerance"],
        )
    elif model_name == "memoryless_rsa":
        model = MemorylessRSA(
            meanings_A=model_args["meanings_A"],
            meanings_B=model_args["meanings_B"],
            categories=model_args["categories"],
            utterances=model_args["utterances"],
            lexicon_A=model_args["lexicon_A"],
            lexicon_B=model_args["lexicon_B"],
            prior=model_args["prior"],
            costs=model_args["costs"],
            alpha=model_args["alpha"],
            pov="listener",
            max_depth=model_args["max_depth"],
            tolerance=model_args["tolerance"],
        )
    elif model_name == "memoryless_literal":
        model = MemorylessLiteral(
            meanings_A=model_args["meanings_A"],
            meanings_B=model_args["meanings_B"],
            categories=model_args["categories"],
            utterances=model_args["utterances"],
            lexicon_A=model_args["lexicon_A"],
            lexicon_B=model_args["lexicon_B"],
            prior=model_args["prior"],
            costs=model_args["costs"],
            alpha=model_args["alpha"],
        )
    elif model_name == "prior_model":
        model = PriorModel(
            meanings_A=model_args["meanings_A"],
            meanings_B=model_args["meanings_B"],
            categories=model_args["categories"],
            utterances=model_args["utterances"],
            costs=model_args["costs"],
            prior=model_args["prior"]
        )
    elif model_name.startswith("llm_"):
        model = LLMDialog(
            system_prompt_A=f"You are playing a game with the user. You are given {n_possitions} cards, "
            "each of which contains a letter (A, B, C, ...) and a number (1, 2, 3, ...). The goal "
            "is to find the possition of the card that contains the value A1. "
            "You only can see the letter of the card, not the number. The user can see the number, but not "
            "the letter. You have to communicate with the user to find out where is the card that contains the value A1. "
            "In each turn, you only can provide the user, one of the possible possitions that could potentially contatin "
            "the target card (i.e., 1st possition, 2nd possition, ...). Try to do it in the less possible turns. "
            f"You are given {n_possitions} cards containing the values {meaning_A}.",
            system_prompt_B=f"You are playing a game with the user. You are given {n_possitions} cards, "
            "each of which contains a letter (A, B, C, ...) and a number (1, 2, 3, ...). The goal "
            "is to find the possition of the card that contains the value A1. "
            "You only can see the number of the card, not the letter. The user can see the letter, but not "
            "the number. You have to communicate with the user to find out where is the card that contains the value A1. "
            "In each turn, you only can provide the user, one of the possible possitions that could potentially contatin "
            "the target card (i.e., 1st possition, 2nd possition, ...). Try to do it in the less possible turns. "
            f"You are given {n_possitions} cards containing the values {meaning_B}.",
            utterances=model_args["utterances"],
            categories=model_args["categories"],
            llm=model_args["llm"],
            game="findA1",
        )
    elif model_name.startswith("llmrsa_"):
        model = LLMRSA(
            meanings_A=model_args["meanings_A"],
            meanings_B=model_args["meanings_B"],
            system_prompt_template_A=f"You are playing a game with the user. You are given {n_possitions} cards, "
            "each of which contains a letter (A, B, C, ...) and a number (1, 2, 3, ...). The goal "
            "is to find the possition of the card that contains the value A1. "
            "You only can see the letter of the card, not the number. The user can see the number, but not "
            "the letter. You have to communicate with the user to find out where is the card that contains the value A1. "
            "In each turn, you only can provide the user, one of the possible possitions that could potentially contatin "
            "the target card (i.e., 1st possition, 2nd possition, ...). Try to do it in the less possible turns. "
            f"You are given {n_possitions} cards "
            "containing the values {meaning}.",
            system_prompt_template_B=f"You are playing a game with the user. You are given {n_possitions} cards, "
            "each of which contains a letter (A, B, C, ...) and a number (1, 2, 3, ...). The goal "
            "is to find the possition of the card that contains the value A1. "
            "You only can see the number of the card, not the letter. The user can see the letter, but not "
            "the number. You have to communicate with the user to find out where is the card that contains the value A1. "
            "In each turn, you only can provide the user, one of the possible possitions that could potentially contatin "
            "the target card (i.e., 1st possition, 2nd possition, ...). Try to do it in the less possible turns. "
            f"You are given {n_possitions} cards "
            "containing the values {meaning}.",
            categories=model_args["categories"],
            utterances=model_args["utterances"],
            prior=model_args["prior"],
            llm=model_args["llm"],
            alpha=model_args["alpha"],
            costs=model_args["costs"],
            pov="listener",
            max_depth=model_args["max_depth"],
            tolerance=model_args["tolerance"],
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model


def run_model_for_n_turns(model_name, model_args, turns, meaning_A, meaning_B, n_possitions):
    
    # Init the model
    model = init_model(model_name, model_args, meaning_A, meaning_B, n_possitions)

    # Run for each turn
    new_past_utterances = []
    category_dist = []
    for turn, speaker_now in zip(range(1, turns + 1),cycle("AB")):

        # update the model
        model.run(new_past_utterances, speaker_now)

        # sample a new utterance
        meaning_S = meaning_A if speaker_now == "A" else meaning_B
        new_utt = model.sample_new_utterance_from_last_speaker(meaning_S)
        new_past_utterances = [{"speaker": speaker_now, "utterance": new_utt}]

        # Get listener's category distribution for current turn
        meaning_L = meaning_B if speaker_now == "A" else meaning_A
        listener_dist = model.get_category_dist_from_last_listener(new_utt, meaning_L)
        category_dist.append(listener_dist)

    category_dist = np.vstack(category_dist)
    return category_dist

def create_world(n_possitions):
    world = {
        "meanings_A": ["".join(l) for l in product("AB", repeat=n_possitions)],
        "meanings_B": ["".join(n) for n in product("12", repeat=n_possitions)],
        "categories": ["There is no A1 card"] + [f"The card A1 is at possition {i+1}" for i in range(n_possitions)],
        "utterances": [f"The card A1 is at possition {i+1}" for i in range(n_possitions)],
    }

    prior = np.zeros((2**n_possitions,2**n_possitions,n_possitions+1))
    for i in range(2**n_possitions):
        for j in range(2**n_possitions):
            # check if (meaninings_A[i][k] == "A" and meanings_B[j][k] == "1") happens only once for a fixed i,j for k in range(p)
            count = 0
            A1_idx = None
            for k in range(n_possitions):
                if world["meanings_A"][i][k] == "A" and world["meanings_B"][j][k] == "1":
                    count += 1
                    A1_idx = k
            if count == 1:
                prior[i,j,A1_idx+1] = 1
            elif count == 0:
                prior[i,j,0] = 1
    prior = prior / np.sum(prior)

    lexicon_A = np.zeros((n_possitions,len(world["meanings_A"])))
    for u, utt in enumerate(world["utterances"]):
        for i, meaning in enumerate(world["meanings_A"]):
            if meaning[int(u)] == "A":
                lexicon_A[u,i] = 1
    lexicon_A[:,-1] = 1

    lexicon_B = np.zeros((n_possitions,len(world["meanings_B"])))
    for u, utt in enumerate(world["utterances"]):
        for i, meaning in enumerate(world["meanings_B"]):
            if meaning[int(u)] == "1":
                lexicon_B[u,i] = 1
    lexicon_B[:,-1] = 1

    world["prior"] = prior
    world["lexicon_A"] = lexicon_A
    world["lexicon_B"] = lexicon_B
    world["costs"] = np.zeros(len(world["utterances"]))
    
    return world

def compute_metric(probs, y, metric):
    if metric == "accuracy":
        return (probs.argmax(axis=1) == y).astype(float)
    elif metric == "nll":
        return -np.log(probs[np.arange(probs.shape[0]),y])
    else:
        raise ValueError(f"Metric {metric} not supported")


def compute_metrics(category_dist, y, categories, metrics):
    y_vec = categories.index(y) * np.ones(category_dist.shape[0],dtype=int)
    results = {}
    for metric in metrics:
        results[metric] = compute_metric(category_dist, y_vec, metric)
    return results


def plot_results(df, models, alpha, max_depth, tolerance, metrics, output_dir):
    
    fig, ax = plt.subplots(1, len(metrics), figsize=(12, 6))
    if len(metrics) == 1:
        ax = np.array([ax])
    for i, metric in enumerate(metrics):
        df_metric = df.groupby(["model","turn"]).mean().reset_index()
        for c, model in enumerate(models):
            model_df = df_metric[df_metric["model"] == model].sort_values("turn")
            ax[i].plot(model_df["turn"], model_df[metric], label=model2name(model), linestyle="--", linewidth=2, color=f"C{c}")
            # ax[i].errorbar(
            #     model_df["turn"], model_df["mean"], yerr=model_df["std"], 
            #     fmt="o", capsize=5, capthick=2, elinewidth=2, markersize=5, color=f"C{c}",
            # )
            ax[i].set_ylabel(metric2name[metric])
            ax[i].set_xlabel("Turn")
            ax[i].grid(True)
            ax[i].set_xticks(model_df["turn"].astype(int))
    fig.suptitle(f"Model results over Turns for alpha={alpha}, max_depth={max_depth}, tolerance={tolerance}")
    ax[-1].legend(loc="lower center", bbox_to_anchor=(-0.1, -0.2), fontsize=12, ncol=4)
    plt.savefig(output_dir / f"scores_alpha={alpha}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)



def main(
    n_possitions: Optional[int] = 3,
    n_turns: Optional[int] = 10,
    models: Optional[List[str]] = ["crsa"],
    metrics: Optional[List[str]] = ["accuracy", "nll"],
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
    suboutput_dir = output_dir
    suboutput_dir.mkdir(parents=True, exist_ok=True)

    # create world
    world = create_world(n_possitions)

    # Create scenarios
    scenarios = [
        sample_from_prior(world["prior"], world["meanings_A"], world["meanings_B"], world["categories"])
        for _ in range(n_seeds)
    ]

    # Run models
    model_args = {
        **world,
        "alpha": alpha,
        "max_depth": max_depth,
        "tolerance": tolerance,
    }
    llm_not_loaded = True
    model_dirs = []
    for model_name in models:
        if model_name.startswith("llmrsa_"):    
            model_dir = suboutput_dir / f"alpha={alpha}" / model_name.replace("/", "--")
        elif model_name.startswith("llm_"):
            model_dir = suboutput_dir / model_name.replace("/", "--")
        else:
            model_dir = suboutput_dir / f"alpha={alpha}" / model_name
        model_dirs.append(model_dir)
        if model_dir.exists():
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        all_model_results = []
        if model_name.startswith("llm_") and llm_not_loaded:
            model_args["llm"] = LLM.load(model_name[4:])
            model_args["llm"].distribute(devices="auto", precision="bf16-true")
            llm_not_loaded = False
        elif model_name.startswith("llmrsa_") and llm_not_loaded:
            model_args["llm"] = LLM.load(model_name[7:])
            model_args["llm"].distribute(devices="auto", precision="bf16-true")
            llm_not_loaded = False
        script_logger.info(f"Running model {model_name} for {n_seeds} seeds.")
        for meaning_A, meaning_B, y in tqdm(scenarios):
            category_dist = run_model_for_n_turns(model_name, model_args, n_turns, meaning_A, meaning_B, n_possitions)
            model_results = compute_metrics(category_dist, y, world["categories"], metrics)
            for metric in metrics:
                for turn in range(1, n_turns + 1):
                    all_model_results.append({
                        "seed": seed,
                        "model": model_name,
                        "turn": turn,
                        metric: model_results[metric][turn-1]
                    })
        df = pd.DataFrame(all_model_results)
        df.to_csv(model_dir / f"results.csv", index=False)

    results = []
    for model_dir in model_dirs:
        model_df = pd.read_csv(model_dir / f"results.csv", index_col=None, header=0)
        results.append(model_df)
    results = pd.concat(results, ignore_index=True, axis=0)
    results.to_csv(suboutput_dir / f"results_alpha={alpha}.csv", index=False)

    plot_results(results, models, alpha, max_depth, tolerance, metrics, suboutput_dir)
            



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
    parser.add_argument("--n_possitions", type=int, help="Number of positions of the Find A1 Game", default=3)
    parser.add_argument("--models", type=str, nargs="+", help="Models to run", default=["crsa_sample"])
    parser.add_argument("--metrics", type=str, nargs="+", help="Metrics to use", default=["accuracy"])
    parser.add_argument("--n_turns", type=int, help="Number of turns", default=5)
    parser.add_argument("--alpha", type=float, help="Alpha to run CRSA with", default=[1.0])
    parser.add_argument("--max_depth", type=int_or_inf, help="Max depth to run CRSA with", default=None)
    parser.add_argument("--tolerance", type=float, help="Tolerance to run CRSA with", default=None)
    parser.add_argument("--seed", type=int, help="Seed to run CRSA with", default=None)
    parser.add_argument("--n_seeds", type=int, help="Number of seeds to run each model", default=1)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output", default=False)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / f"p={args.n_possitions}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update configuration
    config = {
        "n_possitions": args.n_possitions,
        "models": args.models,
        "metrics": args.metrics,
        "n_turns": args.n_turns,
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "output_dir": output_dir,
        "verbose": args.verbose,
    }

    return config

if __name__ == '__main__':
    config = setup()
    main(**config)

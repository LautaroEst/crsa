
import argparse
from itertools import cycle
from pathlib import Path
from typing import List, Literal, Union
from lightning import seed_everything
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

from ..src.io import init_logger, close_logger, read_yaml, check_iter_args
from ..src.datasets import FindA1Dataset
from ..src.pragmatics import init_model
from ..src.speakers import StaticLexicon, DynamicLexicon
from ..src.evaluate import compute_metric



model2config = {
    "crsa_wm": {"label": "$L_1^{CRSA-WM}(y|u_t,w_t,m_{L_t})$", "color": "tab:blue", "linestyle": ":", "marker": "o"},
    "crsa": {"label": "$L_1^{CRSA}(y|u_t,w_t,m_{L_t})$", "color": "tab:blue", "linestyle": "-", "marker": "o"},
    "crsa-literal_wm": {"label": "$L_0^{CRSA-WM}(y|u_t,w_t,m_{L_t})$", "color": "tab:blue", "linestyle": ":", "marker": "x"},
    "crsa-literal": {"label": "$L_0^{CRSA}(y|u_t,w_t,m_{L_t})$", "color": "tab:blue", "linestyle": "-", "marker": "x"},
    "yrsa_wm": {"label": "$L_1^{YRSA-WM}(y|u_t,w_t,m_{L_t})$", "color": "tab:orange", "linestyle": ":", "marker": "o"},
    "yrsa": {"label": "$L_1^{YRSA}(y|u_t,w_t,m_{L_t})$", "color": "tab:orange", "linestyle": "-", "marker": "o"},
    "yrsa-literal_wm": {"label": "$L_0^{YRSA-WM}(y|u_t,w_t,m_{L_t})$", "color": "tab:orange", "linestyle": ":", "marker": "x"},
    "yrsa-literal": {"label": "$L_0^{YRSA}(y|u_t,w_t,m_{L_t})$", "color": "tab:orange", "linestyle": "-", "marker": "x"},
    "prior": {"label": "$P_t(y|m_{L_t})$", "color": "tab:green", "linestyle": "-", "marker": None},
}


def plot_turns(df, models, output_dir):

    metrics = [
        ("accuracy", "Accuracy of the listener"),
        ("igain", "Information Gain: $H_P(Y|M_{L_t})-H_L(Y|U_t,M_{L_t},W_t)$"),
    ]

    def retrieve_cat_dist(x):
        return x["prag_loglst"][x["utterance"], x[f"meaning_{'B' if x['speaker'] == 'A' else 'A'}"],:]

    turns = df["turn"].unique()
    df["category_distribution"] = df.apply(retrieve_cat_dist, axis=1)
    df["target"] = df.apply(lambda x: int(x["target"].item()), axis=1)

    results = []
    for turn in turns:
        prior_logprobs = torch.vstack(df.loc[(df["turn"] == turn) & (df["model"] == "prior"), "category_distribution"].to_list())
        prior_targets = torch.from_numpy(df.loc[(df["turn"] == turn) & (df["model"] == "prior"), "target"].values)
        for model in models:
            iter_num = df.loc[(df["turn"] == turn) & (df["model"] == model), "iter_num"]
            logprobs = torch.vstack(df.loc[(df["turn"] == turn) & (df["model"] == model), "category_distribution"].to_list())
            target = torch.from_numpy(df.loc[(df["turn"] == turn) & (df["model"] == model), "target"].values)
            metrics_results = {}
            for metric, _ in metrics:
                mean, std = compute_metric(logprobs, target, metric, prior_logprobs=prior_logprobs, prior_target=prior_targets)
                metrics_results[f"{metric}:mean"] = mean
                metrics_results[f"{metric}:std"] = std
            results.append({
                "model": model,
                "turn": turn,
                "iter_num:mean": iter_num.mean().item(),
                "iter_num:std": iter_num.std().item(),
                **metrics_results,
            })
    results = pd.DataFrame(results)
    # print(results.set_index(["model", "turn"]))
    # import pdb; pdb.set_trace()

    fig, ax = plt.subplots(2, 1, figsize=(6, 12))
    for i, (metric, metric_name) in enumerate(metrics):
        for model in models:
            model_df = results[results["model"] == model].sort_values("turn")
            ax[i].plot(model_df["turn"], model_df[f"{metric}:mean"], label=model2config[model]["label"], linestyle=model2config[model]["linestyle"], linewidth=3, color=model2config[model]["color"], marker=model2config[model]["marker"], markersize=8)
        ax[i].set_title(metric_name, fontsize=14)
        ax[i].set_xlabel("Turn")
        ax[i].grid(True)
        ax[i].set_xticks(model_df["turn"].astype(int))
    ax[0].set_ylim(0, 1.05)
    ax[-1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), fontsize=12, ncol=3)
    # ax[0].legend(loc="upper left", fontsize=12, bbox_to_anchor=(1, 1))
    fig.tight_layout(pad=1)
    plt.savefig(output_dir / f"metrics_vs_turns.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)



def plot_belief_example(df, sample_id, meanings, utterances, output_dir):
    sample = df.loc[(df["idx"] == sample_id) & (df["model"] == "crsa"), ["turn", "speaker", "logbelief_A", "logbelief_B"]].copy()
    sample = sample.set_index("turn").sort_index()
    turns = sample.index.to_numpy()
    round_meaning_A = meanings["A"][df.loc[(df["idx"] == sample_id) & (df["model"] == "crsa"), "meaning_A"].values[0]]
    round_meaning_B = meanings["B"][df.loc[(df["idx"] == sample_id) & (df["model"] == "crsa"), "meaning_B"].values[0]]
    dialog = df.loc[(df["idx"] == sample_id) & (df["model"] == "crsa"), ["turn","speaker","utterance"]].set_index("turn").sort_index().copy()

    # Belief
    listener_logbelief = []
    listener_meanings = []
    for turn, (spk, logbelief_A, logbelief_B) in sample.iterrows():
        logbelief_S = logbelief_A if spk == "A" else logbelief_B
        logbelief_S = torch.softmax(logbelief_S, dim=0).numpy()
        listener_logbelief.append(logbelief_S)
        listener_meanings.append(meanings["B"] if spk == "A" else meanings["A"])
    listener_logbelief = np.vstack(listener_logbelief).T
    listener_meanings = np.array(listener_meanings).T
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    sns.heatmap(
        listener_logbelief, ax=ax, cmap="viridis", cbar=True, linewidths=0, linecolor=None, 
        annot=listener_meanings, fmt="s", cbar_kws={"aspect": 50}
    )
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks(turns.astype(int) - 0.5)
    ax.set_xticklabels(
        [f"Turn {turn}\n$S_{spk}$: {utterances[utt]}" for turn, (spk,utt) in dialog.iterrows()], rotation=0, fontsize=8)
    ax.set_title(f"Speaker Belief\n$m_A={round_meaning_A},m_B={round_meaning_B}$", fontsize=14)

    fig.tight_layout()
    plt.savefig(output_dir / f"belief.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main(
    game_size: int = 3,
    n_rounds: int = 100,
    models: List[str] = ["crsa"],
    alpha: float = 1.0,
    max_depth: Union[int,Literal['inf']] = None,
    tolerance: float = None,
    sampling_strategy: str = "greedy",
    seed: int = 0,
    print_every_n_turns: int = 10,
    **kwargs
):
    
    # Validate iteration parameters
    max_depth, tolerance = check_iter_args(max_depth, tolerance)

    # Set up the logger and output directory
    output_dir = kwargs["output_dir"]
    logger = kwargs["logger"]

    # Set random seed for reproducibility
    seed_everything(seed, verbose=False)

    # Init dataset
    dataset = FindA1Dataset(game_size, n_rounds)

    # Model is the combination of literal speaker and pragmatic model
    results = []
    for model_name in models:

        # Check if model has already run
        if (output_dir / f"{model_name}_results.pkl").exists():
            results.append(pd.read_pickle(output_dir / f"{model_name}_results.pkl"))
            logger.info(f"Model {model_name} already run. Skipping.")
            continue
        logger.info(f"Running model {model_name}")

        # Init literal speaker
        if "_wm" in model_name:
            speaker = DynamicLexicon(dataset.world["lexicon_A"], dataset.world["lexicon_B"])
        else:
            speaker = StaticLexicon(dataset.world["lexicon_A"], dataset.world["lexicon_B"])

        # Init pragmatic model
        model = init_model(model_name.split("_")[0], dataset.world["logprior"], max_depth=max_depth, tolerance=tolerance)       

        # Generate pragmatic dialogs
        model_results = []
        turns_count = 0
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
                if turns_count % print_every_n_turns == 0:
                    logger.info(f"Round {i+1}/{len(dataset)}, Turn {turn}, Speaker {spk_name}")

                # Get the literal speaker
                lit_logspk, costs = speaker(past_utterances, spk_name)

                # Run the pragmatic model
                prag_logspk, prag_loglst = model.run_turn(lit_logspk, spk_name, costs, alpha)

                # Sample an utterance from the pragmatic speaker
                meaning_S = meaning_A if spk_name == "A" else meaning_B
                new_utt = model.sample_utterance(meaning_S, sampling_strategy)
                if "crsa" in model_name:
                    model.update_belief_(new_utt)
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
                    "prag_logspk": prag_logspk,
                    "prag_loglst": prag_loglst,
                    "logbelief_A": model.logbeliefs[-1]["A"] if model_name in ["crsa", "crsa_wm"] else None,
                    "logbelief_B": model.logbeliefs[-1]["B"] if model_name in ["crsa", "crsa_wm"] else None,
                    "iter_num": model.turns[-1].iter_num,
                })

                # Update turns count
                turns_count += 1

        # Convert results to DataFrame
        model_results_df = pd.DataFrame(model_results)
        model_results_df.to_pickle(output_dir / f"{model_name}_results.pkl")
        results.append(model_results_df)
    
    # Concatenate all results
    results = pd.concat(results, ignore_index=True)

    # Plot results
    plot_turns(results, models, output_dir)
    sample_id = 0
    meanings = {
        "A": dataset.world["meanings_A"],
        "B": dataset.world["meanings_B"],
    }
    plot_belief_example(results, sample_id, meanings, dataset.world["utterances"], output_dir)


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
    except KeyboardInterrupt:
        logger.info(f"Script {Path(__file__).stem}  with config {config_path.stem} interrupted by user")
    except Exception:
        import traceback
        logger.error(f"Error running script {Path(__file__).stem}  with config {config_path.stem}:\n\n{traceback.format_exc()}")
            
    close_logger(logger)



if __name__ == "__main__":
    setup()
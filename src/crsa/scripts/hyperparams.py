
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
from ..src.pragmatics.crsa import CRSA
from ..src.speakers import StaticLexicon


def create_world(
    meanings_A: List[str],
    meanings_B: List[str],
    categories: List[str],
    utterances: List[str],
    lexicon_A: List[List[int]],
    lexicon_B: List[List[int]],
    prior: List[List[List[float]]],
):
    prior = torch.tensor(prior, dtype=torch.float32)
    logprior = torch.log(prior / prior.sum())
    world = {
        "meanings_A": torch.arange(len(meanings_A)),
        "meanings_B": torch.arange(len(meanings_B)),
        "categories": torch.arange(len(categories)),
        "utterances": torch.arange(len(utterances)),
        "logprior": logprior,
    }
    lexicon_A = torch.tensor(lexicon_A, dtype=torch.float32)
    lexicon_B = torch.tensor(lexicon_B, dtype=torch.float32)
    speaker = StaticLexicon(lexicon_A, lexicon_B)

    return world, speaker

def plot_initial_final(root_results_dir, dialog, dialog_meaning_A, dialog_meaning_B, meanings_A, meanings_B, utterances, categories, alphas, max_depths, tolerances):

    turns = len(dialog)

    fig, ax = plt.subplots(turns, len(alphas)+1, figsize=(16, turns*5))
    vmin, vmax = 0, 1
    if turns == 1:
        ax = ax.reshape(1, len(alphas)+1)

    for i, (alpha, max_depth, tolerance) in enumerate(zip(alphas, max_depths, tolerances)):
        max_depth, tolerance = check_iter_args(max_depth, tolerance)
        results_dir = root_results_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        model = CRSA.load(results_dir)
                
        for turn, (utt, turn_model) in enumerate(zip(dialog, model.turns)):
            utt_idx = utterances.index(utt)
            meanings_L = meanings_B if turn_model.spk_name == "A" else meanings_A

            if i == 0:
                # Plot initial listener
                literal_listener = pd.DataFrame(torch.exp(turn_model.listener.literal_as_tensor[utt_idx,:,:]).numpy(), index=meanings_L, columns=categories)
                sns.heatmap(literal_listener, ax=ax[turn,0], cmap='viridis', vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cbar=False, yticklabels=literal_listener.index)
            
            # Plot final listener
            final_listener = pd.DataFrame(torch.exp(turn_model.listener.as_tensor[utt_idx,:,:]).numpy(), index=meanings_L, columns=categories)
            sns.heatmap(final_listener, ax=ax[turn,i+1], cmap='viridis', vmin=vmin, vmax=vmax, annot=True, fmt=".2f", cbar=False, yticklabels=final_listener.index)

    for j, name in enumerate(["Literal listener"] + [f"alpha={alpha}" for alpha in alphas]):
        ax[0,j].set_title(name)
        ax[-1,j].set_xlabel("Meanings")

    for turn in range(turns):
        ax[turn,0].set_ylabel(f"Turn {turn+1}")

    # Unified legend and title
    fig.suptitle(f"Dialog: {dialog}\n$m_A={dialog_meaning_A},m_B={dialog_meaning_B}$", fontsize=16)
    fig.tight_layout(pad=2.3)
    plt.savefig(root_results_dir / "initial_final.pdf")

def plot_history(root_results_dir, dialog, dialog_meaning_A, dialog_meaning_B, alphas, max_depths, tolerances):

    titles2key = [
        ("Conditional entropy", "cond_entropy_history"),
        ("Listener value", "listener_value_history"), 
        ("Gain function", "gain_history"), 
    ]

    turns = len(dialog)

    fig, ax = plt.subplots(turns, len(titles2key), figsize=(16, turns*4))
    if turns == 1:
        ax = ax.reshape(1, len(titles2key))

    for i, (alpha, max_depth, tolerance) in enumerate(zip(alphas, max_depths, tolerances)):
        max_depth, tolerance = check_iter_args(max_depth, tolerance)
        results_dir = root_results_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        model = CRSA.load(results_dir)
        for turn, turn_model in enumerate(model.turns):
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
    fig.suptitle(f"Dialog: {dialog}\n$m_A={dialog_meaning_A},m_B={dialog_meaning_B}$", fontsize=16)
    ax[0,-1].legend([f"$\\alpha={alpha},\,D_{{max}}={max_depth},\,tol={tolerance:.2g}$" for (alpha, max_depth, tolerance) in zip(alphas, max_depths, tolerances)], loc="upper right", bbox_to_anchor=(1.8, 1))
    fig.tight_layout(pad=2.3)
    plt.savefig(root_results_dir / "asymptotic_analysis.pdf")


def main(
    dialog: List[str],
    dialog_meaning_A: int,
    dialog_meaning_B: int,
    dialog_target: int,
    meanings_A: List[str],
    meanings_B: List[str],
    categories: List[str],
    utterances: List[str],
    lexicon_A: List[List[int]],
    lexicon_B: List[List[int]],
    prior: List[List[List[float]]],
    alphas: List[float] = [1.0],
    max_depths: List[Union[int,Literal['inf']]] = [None],
    tolerances: List[float] = [None],
    seed: int = 0,
    print_every_n_turns: int = 10,
    **kwargs
):
    
    output_dir = kwargs["output_dir"]
    script_logger = kwargs["logger"]

    # Set random seed
    seed_everything(seed)

    # Create world and speaker
    world, speaker = create_world(
        meanings_A=meanings_A,
        meanings_B=meanings_B,
        categories=categories,
        utterances=utterances,
        lexicon_A=lexicon_A,
        lexicon_B=lexicon_B,
        prior=prior,
    )

    # Sample meanings from the prior
    meaning_A = torch.tensor(meanings_A.index(dialog_meaning_A))
    meaning_B = torch.tensor(meanings_B.index(dialog_meaning_B))
    dialog_target = torch.tensor(categories.index(dialog_target))
    turns = len(dialog)

    for alpha, max_depth, tolerance in zip(alphas, max_depths, tolerances):

        # Validate and convert max_depth and tolerance
        max_depth, tolerance = check_iter_args(max_depth, tolerance)

        # Check if results already exist
        results_dir = output_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
        if (results_dir / "model.pt").exists():
            script_logger.info(f"Results for alpha={alpha}, max_depth={max_depth}, tolerance={tolerance} already exist. Skipping.")
            continue
        script_logger.info(f"Running model with alpha={alpha}, max_depth={max_depth}, tolerance={tolerance}")

        # Create results directory
        results_dir.mkdir(parents=True, exist_ok=True)
        model_logger = init_logger(f"crsa_alpha={alpha}_max_depth={max_depth}_tolerance={tolerance}", results_dir, log_to_console=False)

        # Init model
        model = CRSA(world["logprior"], max_depth, tolerance, save_memory=False)
        model_logger.info(f"Meaning A: {dialog_meaning_A}, Meaning B: {dialog_meaning_B}, Target: {dialog_target}\n")

        # Generate dialog with the model
        model.reset()
        for turn, spk_name in zip(range(1, turns + 1), cycle("AB")):

            # Get the literal speaker
            past_utterances = [utterances.index(utt) for utt in dialog[:turn-1]]
            lit_logspk, costs = speaker(past_utterances, spk_name)

            # Run the pragmatic model
            prag_logspk, prag_loglst = model.run_turn(lit_logspk, spk_name, costs, alpha)

            # Sample an utterance from the pragmatic speaker
            meaning_S = meaning_A if spk_name == "A" else meaning_B
            new_utt = utterances.index(dialog[turn-1])
            utt_dist = torch.exp(prag_logspk[meaning_S, :])
            model.update_belief_(new_utt)

            # Log the results
            meanings_S = meanings_A if spk_name == "A" else meanings_B
            meanings_L = meanings_B if spk_name == "A" else meanings_B
            model_logger.info(f"Turn {turn} - Agent {spk_name} speaks\n")
            utt_dist = pd.Series(utt_dist.numpy(), index=utterances)
            model_logger.info(f"Utterances distribution: {utt_dist.to_dict()}\n")
            model_logger.info(f"Sampled utterance: {utterances[new_utt]}.\n")
            df = pd.DataFrame(torch.exp(prag_logspk).numpy(), index=meanings_S, columns=utterances)
            model_logger.info(f"Literal speaker:\n{df}\n")
            df = pd.DataFrame(torch.exp(prag_loglst[new_utt,:,:]).numpy(), index=meanings_L, columns=categories)
            model_logger.info(f"Pragmatic listener:\n{df}\n")

        # Save results
        close_logger(model_logger)
        script_logger.info(f"Saving results to {results_dir}")
        model.save(results_dir)

    # Plot initial and final listeners
    plot_initial_final(output_dir, dialog, dialog_meaning_A, dialog_meaning_B, meanings_A, meanings_B, utterances, categories, alphas, max_depths, tolerances)

    # Plot training history
    plot_history(output_dir, dialog, dialog_meaning_A, dialog_meaning_B, alphas, max_depths, tolerances)

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
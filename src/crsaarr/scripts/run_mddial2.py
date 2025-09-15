
import argparse
import logging
from pathlib import Path
import pickle
import os
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import log_softmax, softmax

from ..src.llm_models import LLM, LLMCRSA, LLMRSA, LLMLiteral, ZERO
from ..src.utils import init_logger, Predictions
from ..src.evaluate import compute_metric
from ..src.datasets.mddial2 import MDDialDataset


model2config = {
    "crsa_wm": {"label": "CRSA $\\mathcal{L}(u,m_S,w)$", "color": "tab:blue", "linestyle": "--"},
    "crsa": {"label": "CRSA $\\mathcal{L}(u,m_S)$", "color": "tab:blue", "linestyle": "-"},
    "rsa_wm": {"label": "RSA $\\mathcal{L}(u,m_S,w)$", "color": "tab:orange", "linestyle": "--"},
    "rsa": {"label": "RSA $\\mathcal{L}(u,m_S)$", "color": "tab:orange", "linestyle": "-"},
    "literal_wm": {"label": "Literal $\\mathcal{L}(u,m_S,w)$", "color": "tab:green", "linestyle": "--"},
    "literal": {"label": "Literal $\\mathcal{L}(u,m_S)$", "color": "tab:green", "linestyle": "-"},
    "prior": {"label": "Prior $P(y|m_L)$", "color": "tab:red", "linestyle": "-"},
}



def run_llm(
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    output_dir: Path = Path("outputs"),
    logger: logging.Logger = None,
):
    
    # Init model
    logger.info(f"Loading model {base_model}")
    llm = LLM.load(base_model)
    llm.distribute(accelerator="auto", precision="bf16-true")

    # Init dataset
    dataset = MDDialDataset(prompt_style=llm.prompt_style, split="train")
    
    predictions = Predictions(output_dir / "predictions")
    for sample in dataset.iter_samples():

        # Check if sample already processed
        if sample in predictions:
            logger.info(f"Sample {sample['idx']} already processed, skipping.")
            continue

        # Log progress
        logger.info(f"Running sample {len(predictions)+1}/{len(dataset)}")

        # Run the model for each turn to compute the lexicon
        turns_data = []
        for turn, utterance in enumerate(sample["utterances"][1:-1], start=2):
            # Prompts contains the possible meanings
            # endings can be:
            # - if speaker is patient: "yes" or "no" answers (in one of its variants)
            # - if speaker is doctor: the question about the symptom
            prompts, endings, true_ending_idx = dataset.create_prompts_from_past_utterances(
                past_turns=sample["utterances"][:turn - 1],
                current_turn=utterance,
            )
            all_logits = []
            for prompt in prompts:
                logits = llm.predict(prompt, endings)
                all_logits.append(logits)
            all_logits = np.vstack(all_logits).T # L(u,m)

            turns_data.append({
                "turn": turn,
                "speaker": utterance["speaker"],
                "speaker_logprob": all_logits,
                "true_utt": true_ending_idx,
            })

        category_prompt, endings = dataset.create_category_prompt_from_dialog(sample["utterances"], sample["symptoms"])
        category_distribution = llm.predict(category_prompt, endings)
        predictions.add({
            "idx": sample["idx"],
            "disease": sample["disease"],
            "category_distribution": category_distribution,
            "turns": turns_data,
        })


def check_iter_args(alpha, max_depth, tolerance):
    if max_depth is None and tolerance is None:
        raise ValueError("Either max_depth or tolerance must be provided.")
    if max_depth is None and tolerance is not None:
        max_depth = float("inf")
    if max_depth is not None and tolerance is None:
        tolerance = 0.
    return alpha, max_depth, tolerance


def init_model(model_name, meanings_A, meanings_B, categories, prior, alpha, max_depth, tolerance):
    if model_name == "crsa":
        model = LLMCRSA(
            meanings_A=meanings_A,
            meanings_B=meanings_B,
            categories=categories,
            prior=prior,
            alpha=alpha,
            max_depth=max_depth,
            tolerance=tolerance,
        )
    elif model_name == "rsa":
        model = LLMRSA(
            meanings_A=meanings_A,
            meanings_B=meanings_B,
            categories=categories,
            prior=prior,
            alpha=alpha,
            max_depth=max_depth,
            tolerance=tolerance,
        )
    elif model_name == "literal":
        model = LLMLiteral(
            meanings_A=meanings_A,
            meanings_B=meanings_B,
            categories=categories,
            prior=prior,
            alpha=alpha,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def run_rsa(models, alpha, max_depth, tolerance, output_dir: Path, logger: logging.Logger = None):
   
   # Init dataset
    dataset = MDDialDataset(prompt_style=None, split="train")
    world = dataset.world

    # Load the data
    predictions = Predictions(output_dir / "predictions")
   
    results = []
    alpha, max_depth, tolerance = check_iter_args(alpha, max_depth, tolerance)
    for model_name in models:
        model = init_model(
            model_name,
            meanings_A=world["diseases"],
            meanings_B=["You are a doctor"],
            categories=world["diseases"],
            prior=world["prior"],
            alpha=alpha, 
            max_depth=max_depth, 
            tolerance=tolerance,
        )
        logger.info(f"Running {model_name} with alpha={alpha}, max_depth={max_depth}, tolerance={tolerance}")
        # Run the model for each sample
        for i, sample in enumerate(dataset.iter_samples()):
            if sample["idx"] == 734:
                import pdb; pdb.set_trace()
            if sample not in predictions:
                continue
            sample_data = predictions[sample["idx"]]
            model.reset(world["diseases"][sample["disease"]], "You are a doctor")
            # for (_, turn), (utt, speaker, log_lexicon) in df_lex.loc[(sample["dialog_id"], slice(None)), :].iterrows():
            for turn_data in sample_data["turns"]:
                turn = turn_data["turn"]
                speaker = turn_data["speaker"]
                speaker_logprob = turn_data["speaker_logprob"]
                true_utt = world[f"{speaker}_utterances"][turn_data["true_utt"]]
                utterances = world[f"{speaker}_utterances"]
                model.run_turn(true_utt, log_lexicon=speaker_logprob, utterances=utterances, speaker="A" if speaker == "patient" else "B")
                results.append({
                    "model_name": model_name,
                    "alpha": alpha,
                    "sample_id": sample["idx"],
                    "meaning_A": world["diseases"][sample["disease"]],
                    "meaning_B": "You are a doctor",
                    "turn": turn,
                    "sampled_utt": model.sample_utterance("A" if speaker == "patient" else "B"),
                    "true_utt": true_utt,
                    "true_utt_idx": utterances.index(true_utt),
                    "speaker": speaker,
                    "speaker_dist": model.past_speaker_dist[-1],
                    "lexicon_dist": model.past_lexicon_dist[-1],
                    "category_idx": sample["disease"],
                    "category_distribution": sample_data["category_distribution"],
                    "listener_dist": model.get_category_distribution(),
                    "belief_A": model.belief_A if model_name == "crsa" else None,
                    "belief_B": model.belief_B if model_name == "crsa" else None,
                })
    
    # Concatenate all results
    all_results = pd.DataFrame(results)
    all_results.to_pickle(output_dir / "all_results.pkl")
    return all_results


def compute_results(results, models, alpha, output_dir: Path):

    results_ce = []
    for model_name in models:
        speaker_logprobs = []
        lexicon_logprobs = []
        category_logprobs = []
        category_accuracy = []
        lexicon_cat_logprobs = []
        lexicon_accuracy = []
        dialogs = results[(results["alpha"] == alpha) & (results["model_name"] == model_name)].sort_values(["sample_id","turn"])
        for dialog_id, dialog in dialogs.groupby("sample_id"):
            for i, row in dialog.iterrows():
                speaker_logprobs.append(-np.log(row["speaker_dist"][row["true_utt_idx"]]))
                lexicon_logprobs.append(-np.log(row["lexicon_dist"] + ZERO)[row["true_utt_idx"]])
                category_logprobs.append(-np.log(row["listener_dist"])[row["category_idx"]])
                argmax_classes = np.arange(len(row["listener_dist"]))[row["listener_dist"] == np.max(row["listener_dist"])]
                argmax = np.random.permutation(argmax_classes)[0] if len(argmax_classes) > 1 else argmax_classes[0]    
                category_accuracy.append(argmax == row["category_idx"])
                lexicon_cat_logprobs.append(-log_softmax(row["category_distribution"])[row["category_idx"]])
                lexicon_accuracy.append(np.argmax(row["category_distribution"]) == row["category_idx"])
        
        ce_speaker = np.mean(speaker_logprobs)
        ce_lexicon = np.mean(lexicon_logprobs)
        ce_category = np.mean(category_logprobs)
        acc_category = np.mean(category_accuracy)
        ce_lexicon_cat = np.mean(lexicon_cat_logprobs)
        acc_lexicon = np.mean(lexicon_accuracy)

        results_ce.append({
            "model_name": model_name,
            "$H_S$": ce_speaker,
            "$H_L$": ce_category,
            "Task success rate": acc_category,
        })               
    results_ce.append({
        "model_name": "Literal",
        "$H_S$": ce_lexicon,
        "$H_L$": ce_lexicon_cat,
        "Task success rate": acc_lexicon,
    })
    results_ce = pd.DataFrame(results_ce)
    results_ce = results_ce.set_index("model_name")
    results_ce.to_latex(output_dir / f"results_ce_{alpha}.tex", index=True, float_format="%.2f")





def main(
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    models: List = ["crsa", "rsa", "literal"],
    seed: int = None,
    alpha: float = 1.0,
    max_depth: int = float("inf"),
    tolerance: float = 0.01,
    output_dir: Path = Path("outputs"),
    logger: logging.Logger = None,
):
    
    # Set random seed for reproducibility
    np.random.seed(seed)

    # run_llm(base_model, output_dir, logger)

    results = run_rsa(models, alpha=alpha, max_depth=max_depth, tolerance=tolerance, output_dir=output_dir, logger=logger)
    
    compute_results(results, models, alpha, output_dir)
    # plot_turns(results, models, output_dir)
    


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
    parser.add_argument("--base_model", type=str, help="LLM base model to run", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--models", type=str, nargs="+", help="Models to run", default=["crsa", "rsa", "literal"])
    parser.add_argument("--alpha", type=float, help="Alpha to run", default=1.0)
    parser.add_argument("--max_depth", type=int_or_inf, help="Max depth to run CRSA with", default=None)
    parser.add_argument("--tolerance", type=float, help="Tolerance to run", default=0.01)
    parser.add_argument("--seed", type=int, help="Seed to run", default=None)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / args.base_model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = init_logger(__name__, output_dir)

    # Update configuration
    main_args = {
        "base_model": args.base_model,
        "models": args.models,
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "output_dir": output_dir,
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

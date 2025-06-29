
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

from ..src.llm_models import LLM, LLMCRSA, LLMRSA, LLMLiteral
from ..src.datasets import MDDialDataset
from ..src.utils import init_logger
from ..src.evaluate import compute_metric


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
    save_every: int = 100,
    output_dir: Path = Path("outputs"),
    logger: logging.Logger = None,
):

    if (output_dir / "lexicon_data.pkl").exists() and (output_dir / "categories_data.pkl").exists():
        logger.info("Model already run, skipping...")
        return

    # Initialize the model
    logger.info(f"Loading model {base_model}")
    model = LLM.load(base_model)
    model.distribute(accelerator="auto", precision="bf16-true")

    # Init dataset
    dataset = MDDialDataset(prompt_style=model.prompt_style, split="train")
    world = dataset.world

    if (output_dir / "lexicon_data_part.pkl").exists() and (output_dir / "categories_data_part.pkl").exists():
        with open(output_dir / "lexicon_data_part.pkl", "rb") as f:
            lexicon_data = pickle.load(f)
        with open(output_dir / "categories_data_part.pkl", "rb") as f:
            categories_data = pickle.load(f)
        current_sample = categories_data[-1]["training_sample_id"] + 1
        logger.info(f"Resuming from sample {current_sample}")
    else:
        lexicon_data = []
        categories_data = []
        current_sample = 0

    # Run the model for each sample
    for i, sample in enumerate(dataset.iter_samples()):
        if i < current_sample:
            continue
        
        # Log and save progress
        logger.info(f"Running sample {i}/{len(dataset)}")
        if i % save_every == 0:
            with open(output_dir / "lexicon_data_part.pkl", "wb") as f:
                pickle.dump(lexicon_data, f)
            with open(output_dir / "categories_data_part.pkl", "wb") as f:
                pickle.dump(categories_data, f)
            logger.info(f"Saved progress at sample {i}")

        # Run the model for each turn to compute the lexicon
        for turn, utterance in enumerate(sample["utterances"], start=1):
            past_utterances = sample["utterances"][:turn - 1]
            prompts = dataset.create_prompts_from_past_utterances(
                past_utterances=past_utterances,
                speaker=utterance["speaker"]
            )
            all_logits = []
            unique_utterances = world["patient_utterances"] if utterance["speaker"] == "patient" else world["doctor_utterances"]
            for prompt in prompts:
                logits = model.predict(prompt, unique_utterances)
                all_logits.append(logits)
            all_logits = np.vstack(all_logits).T # L(u,m)

            lexicon_data.append({
                "sample_id": sample["dialog_id"],
                "turn": turn,
                "utt": utterance["content"],
                "speaker": utterance["speaker"],
                "lexicon": all_logits,
            })
        category_prompt, endings = dataset.create_category_prompt_from_dialog(sample["utterances"], sample["symptoms"])

        category_dist = model.predict(category_prompt, endings)
        categories_data.append({
            "training_sample_id": i,
            "sample_id": sample["dialog_id"],
            "category_dist": category_dist,
            "disease": sample["disease"],
        })
    
    with open(output_dir / "lexicon_data.pkl", "wb") as f:
        pickle.dump(lexicon_data, f)
    with open(output_dir / "categories_data.pkl", "wb") as f:
        pickle.dump(categories_data, f)
    os.remove(output_dir / "lexicon_data_part.pkl")
    os.remove(output_dir / "categories_data_part.pkl")
    logger.info("Finished running model")


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
    with open(output_dir / "lexicon_data_part.pkl", "rb") as f:
        df_lex = pd.DataFrame(pickle.load(f)).set_index(["sample_id", "turn"]).sort_index()
    with open(output_dir / "categories_data_part.pkl", "rb") as f:
        df_cat = pd.DataFrame(pickle.load(f)).set_index(["sample_id"]).sort_index()
   
    results = []
    alpha, max_depth, tolerance = check_iter_args(alpha, max_depth, tolerance)
    for model_name in models:
        model = init_model(
            model_name,
            meanings_A=world["symptoms"],
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
            if sample["dialog_id"] not in df_lex.index.get_level_values(0):
                continue
            model.reset(sample["symptoms_id"], "You are a doctor")
            for (_, turn), (utt, speaker, log_lexicon) in df_lex.loc[(sample["dialog_id"], slice(None)), :].iterrows():
                utterances = world[f"{speaker}_utterances"]
                model.run_turn(utt, log_lexicon=log_lexicon, utterances=utterances, speaker="A" if speaker == "patient" else "B")
                results.append({
                    "model_name": model_name,
                    "alpha": alpha,
                    "sample_id": sample["dialog_id"],
                    "meaning_A": sample["symptoms_id"],
                    "meaning_B": "You are a doctor",
                    "turn": turn,
                    "sampled_utt": model.sample_utterance("A" if speaker == "patient" else "B"),
                    "true_utt": utt,
                    "true_utt_idx": utterances.index(utt),
                    "speaker": speaker,
                    "speaker_dist": model.past_speaker_dist[-1],
                    "lexicon_dist": model.past_lexicon_dist[-1],
                    "category_idx": world["diseases"].index(df_cat.loc[sample["dialog_id"], "disease"]),
                    "category_distribution": df_cat.loc[sample["dialog_id"], "category_dist"],
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
                lexicon_logprobs.append(-np.log(row["lexicon_dist"])[row["true_utt_idx"]])
                category_logprobs.append(-np.log(row["listener_dist"])[row["category_idx"]])
                category_accuracy.append(np.argmax(row["listener_dist"]) == row["category_idx"])
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
    save_every: int = 100,
    alpha: float = 1.0,
    max_depth: int = float("inf"),
    tolerance: float = 0.01,
    output_dir: Path = Path("outputs"),
    logger: logging.Logger = None,
):
    
    # Set random seed for reproducibility
    np.random.seed(seed)

    # run_llm(base_model, save_every, output_dir, logger)

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
    parser.add_argument("--save_every", type=int, help="Save every N samples", default=100)
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
        "save_every": args.save_every,
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

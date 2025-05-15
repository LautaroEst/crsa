
import argparse
import logging
from pathlib import Path
import pickle
import os

import numpy as np
import pandas as pd

from ..src.llm_models import LLM
from ..src.datasets import MDDialDataset
from ..src.utils import init_logger


model2config = {
    "crsa_wm": {"label": "CRSA $\mathcal{L}(u,m_S,w)$", "color": "tab:blue", "linestyle": "--"},
    "crsa": {"label": "CRSA $\mathcal{L}(u,m_S)$", "color": "tab:blue", "linestyle": "-"},
    "rsa_wm": {"label": "RSA $\mathcal{L}(u,m_S,w)$", "color": "tab:orange", "linestyle": "--"},
    "rsa": {"label": "RSA $\mathcal{L}(u,m_S)$", "color": "tab:orange", "linestyle": "-"},
    "literal_wm": {"label": "Literal $\mathcal{L}(u,m_S,w)$", "color": "tab:green", "linestyle": "--"},
    "literal": {"label": "Literal $\mathcal{L}(u,m_S)$", "color": "tab:green", "linestyle": "-"},
    "prior": {"label": "Prior $P(y|m_L)$", "color": "tab:red", "linestyle": "-"},
}



def main(
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    seed: int = 0,
    save_every: int = 100,
    output_dir: Path = Path("outputs"),
    logger: logging.Logger = None,
):
    
    # Set random seed for reproducibility
    np.random.seed(seed)

    if (output_dir / "lexicon_data.pkl").exists() and (output_dir / "categories_data.pkl").exists():
        logger.info("Model already run, skipping...")
        return

    # Initialize the model
    logger.info(f"Loading model {base_model}")
    model = LLM.load(base_model)
    model.distribute(accelerator="auto", precision="bf16-true")

    # Init dataset
    dataset = MDDialDataset(prompt_style=model.prompt_style)
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
            for prompt in prompts:
                logits = model.predict(prompt, world["unique_utterances"])
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

def parse_args():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run models for a given configuration file")
    parser.add_argument("--base_model", type=str, help="LLM base model to run", default="meta-llama/Llama-3.2-1B-Instruct")
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
        "save_every": args.save_every,
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

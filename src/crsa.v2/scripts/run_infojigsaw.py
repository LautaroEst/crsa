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
from ..src.infojigsaw import InfojigsawDataset
from ..src.utils import compute_metrics, plot_results

def create_lexicon(system_prompt, meanings, utterances, llm):
    lexicon = np.zeros((len(utterances), len(meanings)))
    for j, meaning in enumerate(meanings):
        prompt = llm.prompt_style.apply([{"role": "system", "content": system_prompt.format(meaning=meaning)}])
        endings = [llm.prompt_style.apply([{"role": "assistant", "content": u}]) for u in utterances]
        preds = llm.predict(prompt=prompt, endings=endings)
        lexicon[:, j] = np.exp(preds)
    lexicon /= lexicon.sum()
    return lexicon
            

def create_system_prompt(meaning, world, llm):
    return f"You are playing the game with board {meaning}"


def init_model(model_name, world, iter_args, llm, system_prompt_A, system_prompt_B):
    if model_name == "crsa":
        model = CRSA(
            meanings_A=world["meanings_A"],
            meanings_B=world["meanings_B"],
            categories=world["categories"],
            utterances=world["utterances"],
            lexicon_A=create_lexicon(system_prompt_A, world["meanings_A"], world["utterances"], llm),
            lexicon_B=create_lexicon(system_prompt_B, world["meanings_B"], world["utterances"], llm),
            prior=world["prior"],
            costs=iter_args["costs"],
            alpha=iter_args["alpha"],
            pov="listener",
            max_depth=iter_args["max_depth"],
            tolerance=iter_args["tolerance"],
        )
    elif model_name == "memoryless_rsa":
        model = MemorylessRSA(
            meanings_A=world["meanings_A"],
            meanings_B=world["meanings_B"],
            categories=world["categories"],
            utterances=world["utterances"],
            lexicon_A=create_lexicon(system_prompt_A, world["meanings_A"], world["utterances"], llm),
            lexicon_B=create_lexicon(system_prompt_B, world["meanings_B"], world["utterances"], llm),
            prior=world["prior"],
            costs=iter_args["costs"],
            alpha=iter_args["alpha"],
            pov="listener",
            max_depth=iter_args["max_depth"],
            tolerance=iter_args["tolerance"],
        )
    elif model_name.startswith("llmrsa_"):
        model = LLMRSA(
            meanings_A=world["meanings_A"],
            meanings_B=world["meanings_B"],
            system_prompt_template_A=system_prompt_A,
            system_prompt_template_B=system_prompt_B,
            categories=world["categories"],
            utterances=world["utterances"],
            prior=world["prior"],
            llm=llm,
            alpha=iter_args["alpha"],
            costs=iter_args["costs"],
            pov="listener",
            max_depth=iter_args["max_depth"],
            tolerance=iter_args["tolerance"],
        )
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model


def run_model_for_n_turns(model_name, world, iter_args, llm, messages, meaning_A, meaning_B):

    system_prompt_A = create_system_prompt(meaning_A, world, llm)
    system_prompt_B = create_system_prompt(meaning_B, world, llm)
    
    # Init the model
    model = init_model(model_name, world, iter_args, llm, system_prompt_A, system_prompt_B)

    # Run for each turn
    utterances = []
    for msg in messages:
        model.run(utterances, msg["speaker"])
        utterances.append(msg["utterance"])

    meaning_L = meaning_B if messages[-1]["speaker"] == "playerChar" else meaning_A
    category_dist = model.get_category_dist_from_last_listener(utterances[-1], meaning_L)
    return category_dist



def main(
    models: Optional[List[str]] = ["crsa"],
    metrics: Optional[List[str]] = ["accuracy", "nll"],
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
    suboutput_dir = output_dir
    suboutput_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dataset
    dataset = InfojigsawDataset()
    world = dataset.world

    # Run models
    iter_args = {
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
            llm = LLM.load(model_name[4:])
            llm.distribute(devices="auto", precision="bf16-true")
            llm_not_loaded = False
        elif model_name.startswith("llmrsa_") and llm_not_loaded:
            llm = LLM.load(model_name[7:])
            llm.distribute(accelerator="cuda", precision="bf16-true")
            llm_not_loaded = False
        else:
            llm = None
        script_logger.info(f"Running model {model_name}.")
        for sample_id, meaning_A, meaning_B, y, messages in tqdm(dataset.iter_samples(), total=len(dataset)):
            category_dist = run_model_for_n_turns(model_name, world, iter_args, llm, messages, meaning_A, meaning_B)
            model_results = compute_metrics(category_dist.reshape(1,-1), y, world["categories"], metrics)
            for metric in metrics:
                all_model_results.append({
                    "sample_id": sample_id,
                    "model": model_name,
                    metric: model_results[metric][0],
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
    parser.add_argument("--models", type=str, nargs="+", help="Models to run", default=["crsa_sample"])
    parser.add_argument("--metrics", type=str, nargs="+", help="Metrics to use", default=["accuracy"])
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
        "metrics": args.metrics,
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



import argparse
from pathlib import Path
import shutil
from typing import List

import pandas as pd
from ..src.llms.llm import LitGPTLLM
from ..src.utils import init_logger
from ..src.evaluate import compute_metric
from ..llms.find_a1 import FindA1Dataset


def load_dataset(dataset_name: str, prompt_style):
    if dataset_name.startswith("findA1"):
        return FindA1Dataset(prompt_style)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


def init_model(model_name: str, dataset, alpha: float, max_depth: int, tolerance: float):
    if model_name == "crsa_sample":
        model = CRSA(
            meanings_A, 
            meanings_B, 
            categories, 
            lexicon_S, 
            dataset.prior, 
            alpha=alpha, 
            max_depth=max_depth, 
            tolerance=tolerance,
        )


def main(
    dataset: str,
    llm: str,
    models: str,
    metrics: List[str],
    alpha: float,
    max_depth: int,
    tolerance: float,
    output_dir: str,
):

    # Load the model
    litgpt_llm = LitGPTLLM.load(model=llm)
    litgpt_llm.distribute(accelerator="auto", precision="bf16-true")

    # Load the dataset
    dataset = load_dataset(dataset, litgpt_llm.prompt_style)
    
    results = []
    # Run for each model
    for model_name in models:
        # Check if model has already run
        if (output_dir / f"{model_name}_results.csv").exists():
            results.append(pd.read_csv(output_dir / f"{model_name}_results.csv"))
            logger.info(f"Model {model_name} already run. Skipping.")
            continue
        elif (output_dir / f"{model_name}_results_part.pkl").exists():
            model_results = pd.read_pickle(output_dir / f"{model_name}_results_part.pkl").to_dict(orient="records")
            logger.info(f"Model {model_name} partially run. Loading previous results.")
        else:
            model_results = []
            logger.info(f"Model {model_name} not run. Starting from scratch.")

        # Initialize the model
        model = init_model(model_name, dataset.world, alpha=alpha, max_depth=max_depth, tolerance=tolerance)
        
        # Run the model for each sample
        for i, (idx, meaning_A, meaning_B, turns, cat) in enumerate(dataset.iter_samples()):

            # Check if the model has already run for this sample
            if idx in [res["sample_id"] for res in model_results]:
                continue
            
            # Log and save progress
            if i % (len(dataset) // 20) == 0:
                logger.info(f"Running sample {i}/{len(dataset)}")
                model_results_df = pd.DataFrame(model_results)
                model_results_df.to_pickle(output_dir / f"{model_name}_results_part.pkl")

            cat_idx = dataset.world["categories"].index(cat)
            model.reset(meaning_A, meaning_B)

            # Run the model for each turn
            for turn in turns:
                model.run_turn(turn["speaker"])

                
            cat_dist = model.get_category_distribution()
            model_results.append({
                "sample_id": idx,
                "model": model_name,
                "category_dist": cat_dist,
                **{metric: compute_metric(cat_dist, cat_idx, metric) for metric in metrics},
            })

        # Convert results to DataFrame
        model_results_df = pd.DataFrame(model_results)
        model_results_df.to_csv(output_dir / f"{model_name}_results.csv", index=False)
        results.append(model_results_df)
        shutil.remove(output_dir / f"{model_name}_results_part.pkl")
        logger.info(f"Model {model_name} finished. Results saved to {output_dir / f'{model_name}_results.csv'}")
    
    # Concatenate all results
    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(output_dir / "all_results.csv", index=False)





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
    parser.add_argument("--models", type=str, nargs="+", help="Models to run", default=["crsa_sample"])
    parser.add_argument("--metrics", type=str, nargs="+", help="Metrics to use", default=["accuracy"])
    parser.add_argument("--alpha", type=float, help="Alpha to run CRSA with", default=[1.0])
    parser.add_argument("--max_depth", type=int_or_inf, help="Max depth to run CRSA with", default=None)
    parser.add_argument("--tolerance", type=float, help="Tolerance to run CRSA with", default=None)
    parser.add_argument("--seed", type=int, help="Seed to run CRSA with", default=None)
    parser.add_argument("--n_seeds", type=int, help="Number of seeds to run each model", default=1)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / f"game_size={args.game_size}" / f"n_turns={args.n_turns}" / f"alpha={args.alpha}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = init_logger(__name__, output_dir)

    # Update configuration
    main_args = {
        "game_size": args.game_size,
        "models": args.models,
        "metrics": args.metrics,
        "n_turns": args.n_turns,
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
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

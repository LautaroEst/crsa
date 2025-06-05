
import argparse
from pathlib import Path
import pickle
from typing import List, Literal, Union
from lightning import seed_everything
import pandas as pd
import torch

from ..src.io import init_logger, read_yaml, check_iter_args
from ..src.datasets import MDDialDataset
from ..src.datasets.utils import Predictions
from ..src.pragmatics import init_model
from ..src.speakers import LLMSpeaker



def predict(speaker: LLMSpeaker, dataset: MDDialDataset, log_every: int = 10, logger=None, output_dir: Path = None):

    if (predictions_dir := output_dir / "processed").exists():
        predictions = Predictions.from_directory(predictions_dir)
    else:
        predictions = Predictions([], predictions_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)


    # Start processing samples
    for i, sample in enumerate(dataset.iter_samples()):
        if sample in predictions:
            logger.info(f"Sample {sample['idx']} already processed. Skipping.")
            continue

        if (i + 1) % log_every == 0:
            logger.info(f"Processing sample {i+1}/{len(dataset)} (idx: {sample['idx']})")

        utterances = speaker.get_dialog_speakers(
            sample["utterances"],
            dataset.world["system_prompts"],
            dataset.world["speakers"]
        )
                
        prediction = {
            "idx": sample["idx"], 
            "meaning_patient": sample["meaning_patient"],
            "meaning_doctor": sample["meaning_doctor"],
            "target": sample["target"],
            "utterances": utterances,
        }

        # Update computed predictions
        predictions.add(prediction)

    return predictions
        

def run_pragmatic_models(dataset: MDDialDataset, predictions: Predictions, models: List[str], alpha: float, max_depth: Union[int, Literal['inf']], tolerance: float, output_dir: Path, logger=None, log_every: int = 10):

    results = []
    for model_name in models:

        # Check if model has already run
        if (output_dir / f"{model_name}_results.pkl").exists():
            results.append(pd.read_pickle(output_dir / f"{model_name}_results.pkl"))
            logger.info(f"Model {model_name} already run. Skipping.")
            continue
        logger.info(f"Running model {model_name}")

        # Init pragmatic model
        model = init_model(model_name.split("_")[0], dataset.world["logprior"], max_depth=max_depth, tolerance=tolerance)       

        # Iterate over samples in the dataset
        model_results = []
        for i, sample in enumerate(predictions):

            # Get the meanings and target
            meaning_patient = sample["meaning_patient"]
            meaning_doctor = sample["meaning_doctor"]
            target = sample["target"]

            # Compute the LL of each utterance
            model.reset()

            # Log the sample index
            if (i + 1) % log_every == 0:
                logger.info(f"Runnung model {model_name} on sample {i+1}/{len(predictions)}")

            # Run each turn
            for turn, utterance in enumerate(sample["utterances"], start=1):
                spk_name = "A" if utterance["speaker"] == "patient" else "B"
                utt_idx = utterance["content"]
                lit_logspk = utterance["logits"]
                costs = torch.zeros(lit_logspk.shape[1], dtype=torch.float32)

                # Run the pragmatic model
                prag_logspk, prag_loglst = model.run_turn(lit_logspk, spk_name, costs, alpha)

                # Save results
                model_results.append({
                    "model": model_name,
                    "idx": sample["idx"],
                    "turn": turn,
                    "utterance": utt_idx,
                    "speaker": spk_name,
                    "meaning_A": meaning_patient,
                    "meaning_B": meaning_doctor,
                    "target": target,
                    "prag_spk_dist": prag_logspk[meaning_patient if spk_name == "A" else meaning_doctor, :],
                    "prag_lst_dist": prag_loglst[meaning_doctor if spk_name == "A" else meaning_patient, :],
                    "logbelief_A": model.logbeliefs[-1]["A"] if model_name == "crsa" else None,
                    "logbelief_B": model.logbeliefs[-1]["B"] if model_name == "crsa" else None,
                    "iter_num": model.turns[-1].iter_num,
                })

        # Convert results to DataFrame
        model_results_df = pd.DataFrame(model_results)
        model_results_df.to_pickle(output_dir / f"{model_name}_results.pkl")
        results.append(model_results_df)
    
    # Concatenate all results
    results = pd.concat(results, ignore_index=True)

    # Show results
    logger.info(f"Results for all models:\n{results.head()}")



def main(
    llm: str = "Llama-3.2-1B-Instruct",
    models: List[str] = ["crsa"],
    alpha: float = 1.0,
    max_depth: Union[int,Literal['inf']] = None,
    tolerance: float = None,
    seed: int = 0,
    log_every: int = 10,
    save_every: int = 10,
    **kwargs
):
    # Validate iteration parameters
    max_depth, tolerance = check_iter_args(max_depth, tolerance)

    # Set up the logger and output directory
    output_dir = kwargs["output_dir"]
    logger = kwargs["logger"]

    # Set random seed for reproducibility
    seed_everything(seed, verbose=False)

    # Initialize the dataset
    dataset = MDDialDataset(split="train")

    # Initialize the LLM speaker
    speaker = LLMSpeaker.load(model=llm)
    speaker.distribute(accelerator="auto", precision="bf16-true")

    # Predict literal speakers
    predictions = predict(
        speaker, 
        dataset, 
        log_every=log_every, 
        logger=logger, 
        output_dir=output_dir
    )

    # Run RSA models
    run_pragmatic_models(dataset, predictions, models, alpha, max_depth, tolerance, output_dir, logger, log_every)




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
        
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    setup()
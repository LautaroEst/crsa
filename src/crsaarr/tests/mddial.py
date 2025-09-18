
from pathlib import Path
import numpy as np

from ..src.datasets.mddial import MDDialDataset
from ..src.llm_models.base_llm import LLM
from ..src.utils import init_logger, Predictions


def run_llm(dataset, llm, output_dir, logger):
    
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
            import pdb; pdb.set_trace()
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

        predictions.add({
            "idx": sample["idx"],
            "disease": sample["disease"],
            "turns": turns_data,
        })




def main():

    seed = 1234
    base_model = "EleutherAI/pythia-70m"
    # base_model="meta-llama/Llama-3.2-1B-Instruct"

    # Create output directory
    output_dir = Path("outputs/tests") / Path(__file__).stem / base_model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = init_logger(__name__, output_dir)

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Init model
    logger.info(f"Loading model {base_model}")
    llm = LLM.load(base_model)
    llm.distribute(accelerator="auto", precision="bf16-true")

    # Init dataset
    dataset = MDDialDataset(prompt_style=llm.prompt_style, split="train")

    # Run predictions on LLM
    run_llm(dataset, llm, output_dir, logger)

        


if __name__ == "__main__":
    main()
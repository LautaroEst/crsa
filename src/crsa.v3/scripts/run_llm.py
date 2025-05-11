

import numpy as np
from ..src.llms.llm import LitGPTLLM
from ..src.llms.datasets.find_a1 import FindA1Dataset

def load_dataset(dataset_name: str, prompt_style):
    if dataset_name.startswith("findA1"):
        return FindA1Dataset(prompt_style)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


def main(
    dataset: str,
    model: str,
    num_responses: int,
):
    # Load the model
    litgpt_llm = LitGPTLLM.load(model=model)
    litgpt_llm.distribute(accelerator="auto", precision="bf16-true")

    # Load the dataset
    dataset = load_dataset(dataset, litgpt_llm.prompt_style)

    results = []
    # Run the model for each sample
    for i, (idx, meaning_A, meaning_B, turns_utterances, cat) in enumerate(dataset.iter_samples()):

        for turn_num, turn_data in enumerate(turns_utterances, start=1):
            meaning_S = meaning_A if turn_data["speaker"] == "A" else meaning_B
            meaning_S_prompt = dataset.create_prompt_from_meaning_and_past(
                meaning=meaning_S, 
                past_utterances=turns_utterances[:turn_num-1],
                speaker=turn_data["speaker"],
            )
            responses = []
            logits = []
            for _ in range(num_responses):
                response, logprob, num_tokens = litgpt_llm.generate(
                    meaning_S_prompt,
                    max_new_tokens=100,
                    top_p=0.9,
                    top_k=50,
                )
                responses.append(response)
                logits.append(logprob / num_tokens)
            results.append({
                "idx": idx,
                "turn_num": turn_num,
                "meaning_A": meaning_A,
                "meaning_B": meaning_B,
                "cat": cat,
                "utterance": turn_data["content"],
                "speaker": turn_data["speaker"],
                "responses": responses,
                "logits": logits,
            })

            



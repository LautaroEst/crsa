
from copy import deepcopy
import json

import torch
from tqdm import tqdm


class MDDialDataset:

    data_dir = "data/mddial/"

    def __init__(self, split="train"):
        if split not in ["train", "test"]:
            raise ValueError("split must be either 'train' or 'test'")
        self.data, self.world, self.shots = self._load_data(f"{split}.json")
        self.sampled_indices = torch.randperm(len(self.data))

    def _load_data(self, filename):
        with open(self.data_dir + filename, "r") as f:
            dialogs = json.load(f)
        data = {"dialogs": [dialogs[f"Dialog {i}"] for i in range(1, len(dialogs) + 1)]}
        with open(self.data_dir + "disease.txt", "r") as f:
            diseases = [line.strip() for line in f.readlines()]
        with open(self.data_dir + "symptom.txt", "r") as f:
            symptoms = [line.strip() for line in f.readlines()]
        with open(self.data_dir + "disease_symptoms.txt", "r") as f:
            disease2symptoms = json.load(f)

        diseases_count = {}
        valid_data = []
        for i, dialog in enumerate(tqdm(data["dialogs"])):
            
            found = False
            d_idx = 0
            for d in diseases:
                if d in dialog[-1]["doctor"]:
                    found = True
                    break
                d_idx += 1

            if found:
                sample = {}
                sample["idx"] = torch.tensor(i)
                sample["target"] = torch.tensor(d_idx)

                # Utterances
                utterances = []
                for t, turn in enumerate(dialog):
                    utterances.append({"speaker": "patient", "content": turn["patient"]})
                    utterances.append({"speaker": "doctor", "content": turn["doctor"]})
                sample["utterances"] = utterances
                
                # Patient meanings    
                sample["meaning_patient"] = torch.tensor(d_idx)
                
                # Doctor meanings (Always the same meaning)
                sample["meaning_doctor"] = torch.tensor(0)  

                # Count diseases
                if d_idx not in diseases_count:
                    diseases_count[d_idx] = 0
                diseases_count[d_idx] += 1  

                valid_data.append(sample)

        
        prior = torch.zeros((len(diseases), 1, len(diseases)))
        for i in range(len(diseases)):
            prior[i, 0, i] = diseases_count.get(i, 0)
        logprior = torch.log(prior / torch.sum(prior))

        shots_ids = [43, 1204]
        shots = []
        for i in shots_ids:
            record = deepcopy(valid_data[i])
            shots.append({
                "idx": record["idx"],
                "utterances": record["utterances"],
                "meaning_patient": disease2symptoms[diseases[record["meaning_patient"].item()]],
                "meaning_doctor": record["meaning_doctor"],
                "target": record["target"],
            })

        world = {
            "speakers": ["patient", "doctor"],
            "diseases": diseases,
            "symptoms": symptoms,
            "system_prompts": {
                "patient": [self._create_patient_prompt(disease2symptoms[d], shots) for d in diseases],
                "doctor": [self._create_doctor_prompt(disease2symptoms, diseases, shots)],
            },
            "logprior": logprior,
        }

        return valid_data, world, shots
    

    def iter_samples(self):
        for idx in self.sampled_indices:
            yield self.data[idx]


    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[idx]


    def __len__(self):
        return len(self.data)
    

    def _create_patient_prompt(self, patient_symptoms, shots):
        prompt = (
            "You are an assistant that simulates to be a patient "
            "who has a disease and describes the symptoms to the user, "
            "which is a medical doctor.\n\nBelow are examples of conversations.\n\n"
            ""
        )
        for example in shots:
            prompt += "--- Example: ---\n\n"
            prompt += "You are experiencing some of the following symptoms:\n"
            for s in example["meaning_patient"]:
                prompt += f"- {s}\n"
            prompt += "\nConversation:\n"
            for utterance in example["utterances"]:
                if utterance["speaker"] == "patient":
                    prompt += f"Assistant: {utterance['content']}\n"
                else:
                    prompt += f"User: {utterance['content']}\n"
            prompt += "\n--- End of example ---\n\n"
        prompt += (
            "Now, begin a new conversation with the doctor.\n\n"
            f"You are experiencing some of the following symptoms:\n"
        )
        for s in patient_symptoms:
            prompt += f"- {s}\n"
        prompt += "\nBegin the conversation by describing some of your symptoms to the doctor.\n"

        return prompt

    def _create_doctor_prompt(self, disease2symptoms, diseases, shots):
        prompt = (
            "You are an assistant simulating a medical doctor interacting with a patient. "
            "The patient will describe their symptoms gradually in a conversation. "
            "Your task is to ask relevant questions to identify which disease the patient most likely has. "
            "You have access to the following mapping between diseases and their typical symptoms:"
            "\n\n--- Disease-Symptom Mapping ---\n\n"
        )
        for i, disease in enumerate(diseases):
            prompt += f"- {disease}: {', '.join(disease2symptoms[disease])}.\n"
        prompt += "\n--- End of Disease-Symptom Mapping ---\n\n"
        prompt += "Below are examples of conversations with patients.\n\n"
        for example in shots:
            prompt += (
                "--- Example: ---\n\n"
            )
            for utterance in example["utterances"]:
                if utterance["speaker"] == "patient":
                    prompt += f"User: {utterance['content']}\n"
                else:
                    prompt += f"Assistant: {utterance['content']}\n"
            prompt += "\n--- End of example ---\n\n"
        prompt += "Now, begin a new conversation. The patient will describe symptoms gradually. Use the disease-symptom mapping to guide your questions.\n"
        return prompt

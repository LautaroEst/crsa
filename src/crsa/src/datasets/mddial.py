
from copy import deepcopy
import json

import torch


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

        valid_data = []
        diseases_count = {}
        for i, dialog in enumerate(data["dialogs"]):
            
            found = False
            for d_idx, d in enumerate(diseases):
                if d in dialog[-1]["doctor"]:
                    found = True
                    break

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

        shots_ids = [5, 43, 1204, 864]
        shots = []
        for i in shots_ids:
            record = deepcopy(valid_data[i])
            shots.append({
                "idx": record["idx"],
                "utterances": record["utterances"],
                "meaning_patient": record["meaning_patient"],
                "meaning_doctor": record["meaning_doctor"],
                "target": record["target"],
            })

        world = {
            "speakers": ["patient", "doctor"],
            "diseases": diseases,
            "symptoms": symptoms,
            "system_prompts": {
                "patient": [self._create_patient_prompt(d, diseases, shots) for d in diseases],
                "doctor": [self._create_doctor_prompt(diseases, shots)],
            },
            "logprior": logprior,
        }

        return valid_data, world, shots_ids
    
    def iter_samples(self):
        for idx in self.sampled_indices:
            yield self.data[idx]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    

    def _create_patient_prompt(self, meaning, diseases, shots):
        prompt = (
            "You are an assistant that simulates to be a patient "
            "who has a disease and describes the symptoms to the user, "
            "which is a medical doctor.\n\n"
        )
        for example in shots:
            prompt += (
                "Here is an example of a conversation between the assitant and the user. "
                f"You are experiencing the symptoms corresponding to {diseases[example['meaning_patient'].item()]}:\n"
            )
            for utterance in example["utterances"]:
                if utterance["speaker"] == "patient":
                    prompt += f"Assistant: {utterance['content']}\n"
                else:
                    prompt += f"User: {utterance['content']}\n"
            prompt += "\n"
        prompt += (
            "Now, you describe your symptoms to the doctor in a natural way. "
            f"You are experiencing the symptoms corresponding to {meaning}.\n"
        )
        return prompt

    def _create_doctor_prompt(self, diseases, shots):
        prompt = (
            "You are an assistant that simulates to be a doctor "
            "who is diagnosing a patient based on the symptoms that he or she describes. "
            "You can ask questions to the patient, but ultimately, "
            "you have provide a diagnosis based on the symptoms described by the patient.\n\n"
        )
        for i, disease in enumerate(diseases):
            prompt += f"{i + 1}. {disease}\n"
        
        for example in shots:
            prompt += (
                "Here is an example of a conversation "
                "between you and the patient. "
            )
            for utterance in example["utterances"]:
                if utterance["speaker"] == "patient":
                    prompt += f"User: {utterance['content']}\n"
                else:
                    prompt += f"Assistant: {utterance['content']}\n"
        prompt += (
            "Now, you are given a conversation with a patient. "
            "The possible diseases are:\n"
        )
        for i, disease in enumerate(diseases):
            prompt += f"{i + 1}. {disease}\n"
        prompt += (
            "You need to diagnose the patient based on the symptoms described by the patient.\n"
        )
        return prompt

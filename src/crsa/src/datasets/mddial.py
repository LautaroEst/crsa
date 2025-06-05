
from copy import deepcopy
import json

import torch
from tqdm import tqdm


class MDDialDataset:

    data_dir = "data/mddial/"

    def __init__(self, split="train"):
        if split not in ["train", "test"]:
            raise ValueError("split must be either 'train' or 'test'")
        self.data, self.world = self._load_data(f"{split}.json")
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

        d_count = {}
        valid_data = []
        unique_meanings_patient = []
        meaning_id = 0
        for i, dialog in enumerate(tqdm(data["dialogs"])):
            
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
                symptoms_in_dialog = torch.zeros(len(symptoms))
                for t, turn in enumerate(dialog):
                    if t == 0:
                        for s_id, s in enumerate(symptoms):
                            if s in turn["patient"]:
                                symptoms_in_dialog[s_id] = 1
                            if s in turn["doctor"]:
                                symptom_q = s
                    elif 0 < t < len(dialog) - 1:
                        if turn["patient"] in ["Yes, sometimes", "I am experiencing that sometimes",
                                               "Yes Doctor, I am feeling that as well", "Yes most of the times"]:
                            for s_id, s in enumerate(symptoms):
                                if s == symptom_q:
                                    symptoms_in_dialog[s_id] = 1
                    utterances.append({"speaker": "patient", "content": turn["patient"]})
                    utterances.append({"speaker": "doctor", "content": turn["doctor"]})
                sample["utterances"] = utterances
                
                # Patient meanings (One-hot encoding of symptoms)
                meaning_patient = torch.zeros(len(symptoms))
                for s_id, s in enumerate(symptoms):
                    if s in disease2symptoms[d]:
                        meaning_patient[s_id] = 1
                meaning_patient = meaning_patient * symptoms_in_dialog

                # Keep track of the number of times each meaning appears
                if (meaning_id, d_idx) in d_count:
                    d_count[(meaning_id, d_idx)] += 1
                else:
                    d_count[(meaning_id, d_idx)] = 1

                # Patient meanings    
                sample["meaning_patient"] = torch.tensor(meaning_id)
                
                # Doctor meanings (Always the same meaning)
                sample["meaning_doctor"] = torch.tensor(0)    

                if not any([all(meaning_patient == m) for m in unique_meanings_patient]): # if not already in unique_meanings_patient
                    unique_meanings_patient.append(meaning_patient)
                    meaning_id += 1
                    
                
                valid_data.append(sample)

        
        prior = torch.zeros((len(unique_meanings_patient), 1, len(diseases)))
        for i, d in enumerate(diseases):
            for j, meaning in enumerate(unique_meanings_patient):
                prior[j, 0, i] = d_count[(j, i)] 
        logprior = torch.log(prior / torch.sum(prior))

        world = {
            "speakers": ["patient", "doctor"],
            "diseases": diseases,
            "symptoms": symptoms,
            "system_prompts": {
                "patient": [self._create_patient_prompt(s, symptoms) for s in unique_meanings_patient],
                "doctor": [self._create_doctor_prompt(diseases)],
            },
            "unique_meanings_patient": unique_meanings_patient,
            "logprior": logprior,
        }

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
        self.shots = shots

        return valid_data, world
    
    def iter_samples(self):
        for idx in self.sampled_indices:
            yield self.data[idx]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    

    def _create_patient_prompt(self, meaning, symptoms):
        prompt = (
            "You are an assistant that simulates to be a patient "
            "who has a disease and describes the symptoms to the user, "
            "which is a medical doctor.\n\n"
        )
        for example in self.shots:
            system_prompt += (
                "Here is an example of a conversation "
                "between the assitant (i.e., the patient) and the user (i.e., the doctor). "
                "You are experiencing the following symptoms:\n"
            )
            for s_id, s_name in zip(self.world["unique_meanings_patient"][example["meaning_patient"].item()], symptoms):
                if s_id == 1:
                    system_prompt += f"- {s_name}\n"
        system_prompt += (
            "Now, you are experiencing the following symptoms:\n"
        )
        for s_id, s_name in zip(self.world["unique_meanings_patient"][meaning], symptoms):
            if s_id == 1:
                prompt += f"- {s_name}\n"
        prompt += (
            "Now, please describe your symptoms to the doctor in a natural way.\n"
        )
        return prompt

    def _create_doctor_prompt(self, diseases):
        pass
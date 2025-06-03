
import json

import torch


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
        with open(self.data_dir + "symptoms.txt", "r") as f:
            symptoms = [line.strip() for line in f.readlines()]
        with open(self.data_dir + "disease_symptoms.txt", "r") as f:
            disease2symptoms = json.load(f)

        d_count = {}
        valid_data = []
        unique_meanings_patient = []
        meaning_id = 0
        for i, dialog in enumerate(data["dialogues"]):
            
            found = False
            for d in diseases:
                if d in dialog[-1]["doctor"]:
                    found = True
                    break

            if found:
                sample = {}
                sample["idx"] = torch.tensor(i)
                sample["target"] = torch.tensor(d)

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
                if (meaning_id, d) in d_count:
                    d_count[(meaning_id, d)] += 1
                else:
                    d_count[(meaning_id, d)] = 1

                # Patient meanings    
                sample["meaning_patient"] = meaning_id
                
                # Doctor meanings (Always the same meaning)
                sample["meaning_doctor"] = torch.tensor(0)    

                if not any([all(meaning_patient == m) for m in unique_meanings_patient]): # if not already in unique_meanings_patient
                    unique_meanings_patient.append(meaning_patient)
                    meaning_id += 1
                    
                
                valid_data.append(sample)

        
        prior = torch.zeros((len(unique_meanings_patient), 1, len(diseases)))
        for i, d in enumerate(diseases):
            for j, meaning in enumerate(unique_meanings_patient):
                prior[j, 0, i] = d_count[(j, d)] 
        logprior = torch.log(prior / torch.sum(prior))

        world = {
            "diseases": diseases,
            "symptoms": symptoms,
            "unique_meanings_patient": unique_meanings_patient,
            "logprior": logprior,
        }

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

from copy import deepcopy
import json

import torch
from tqdm import tqdm

from ..utils import cosine_similarity


class MDDialDataset:

    data_dir = "data/mddial/"

    def __init__(self, split="train", n_patient_distractors=1):
        if split not in ["train", "test"]:
            raise ValueError("split must be either 'train' or 'test'")
        self.data, self.world, self.shots = self._load_data(f"{split}.json")
        self.sampled_indices = torch.randperm(len(self.data))
        if n_patient_distractors < 1 or n_patient_distractors > len(self.world["unique_meanings_patient"]):
            raise ValueError("n_distrctactors must be at least 1")
        self.n_patient_distractors = n_patient_distractors

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
                symptoms_in_dialog = torch.zeros(len(symptoms))
                for t, turn in enumerate(dialog):
                    if t == 0:
                        for s_id, s in enumerate(symptoms):
                            if s in turn["patient"]:
                                symptoms_in_dialog[s_id] = 1
                            if s in turn["doctor"]:
                                symptom_q = s
                    else:
                        if turn["patient"] in ["Yes, sometimes", "I am experiencing that sometimes",
                                               "Yes Doctor, I am feeling that as well", "Yes most of the times"]:
                            for s_id, s in enumerate(symptoms):
                                if s == symptom_q:
                                    symptoms_in_dialog[s_id] = 1
                        for s in symptoms:
                            if s in turn["doctor"]:
                                symptom_q = s
                    utterances.append({"speaker": "patient", "content": turn["patient"]})
                    utterances.append({"speaker": "doctor", "content": turn["doctor"]})
                sample["utterances"] = utterances
                
                # Patient meanings (One-hot encoding of symptoms)
                meaning_patient = torch.zeros(len(symptoms))
                for s_id, s in enumerate(symptoms):
                    if s in disease2symptoms[d]:
                        meaning_patient[s_id] = 1
                meaning_patient = meaning_patient * symptoms_in_dialog

                if len(unique_meanings_patient) == 0:
                    meaning_id = 0
                    unique_meanings_patient.append(meaning_patient)
                else:
                    mask = (torch.vstack(unique_meanings_patient) == meaning_patient).all(dim=1)
                    if not mask.any():
                        # If the meaning is not already in unique_meanings_patient, add it
                        meaning_id = len(unique_meanings_patient)
                        unique_meanings_patient.append(meaning_patient)
                    else:
                        # If it exists, get the index
                        meaning_id = mask.nonzero(as_tuple=True)[0].item()

                # Keep track of the number of times each meaning appears
                if (meaning_id, d_idx) in d_count:
                    d_count[(meaning_id, d_idx)] += 1
                else:
                    d_count[(meaning_id, d_idx)] = 1

                # Patient meanings    
                sample["meaning_patient"] = torch.tensor(meaning_id)
                
                # Doctor meanings (Always the same meaning)
                sample["meaning_doctor"] = torch.tensor(0)    
                
                valid_data.append(sample)

        
        prior = torch.zeros((len(unique_meanings_patient), 1, len(diseases)))
        for i, d in enumerate(diseases):
            for j, meaning in enumerate(unique_meanings_patient):
                if (j, i) in d_count:
                    prior[j, 0, i] = d_count[(j, i)]

        shots_ids = [43, 1204]
        shots = []
        for i in shots_ids:
            record = deepcopy(valid_data[i])
            shots.append({
                "idx": record["idx"],
                "utterances": record["utterances"],
                "meaning_patient": [symptoms[i] for i, s in enumerate(unique_meanings_patient[record["meaning_patient"].item()]) if s == 1],
                "meaning_doctor": record["meaning_doctor"],
                "target": record["target"],
            })

        world = {
            "speakers": ["patient", "doctor"],
            "diseases": diseases,
            "symptoms": symptoms,
            "system_prompts": {
                "patient": [self._create_patient_prompt([symptoms[i] for i, s in enumerate(m) if s == 1], shots) for m in unique_meanings_patient],
                "doctor": [self._create_doctor_prompt(disease2symptoms, diseases, shots)],
            },
            "unique_meanings_patient": unique_meanings_patient,
            "prior": prior,
        }

        return valid_data, world, shots
    

    def iter_samples(self):
        unique_meanings = torch.vstack(self.world["unique_meanings_patient"]) # (n_meanings, r)
        n_meanings = unique_meanings.shape[0]
        n_distractors = self.n_patient_distractors
        for idx in self.sampled_indices[:100]:
            sample = self.data[idx]
            meaning_patient = sample["meaning_patient"].item()  # Get the meaning index
            meaning_patient_vec = unique_meanings[meaning_patient,:] # (r,)

            # Construct distractors
            similarities = cosine_similarity(unique_meanings, meaning_patient_vec) # (m,)
            similarities = similarities / similarities.sum()  # Normalize to create a probability distribution
            indices = torch.multinomial(similarities, num_samples=n_meanings, replacement=False)
            distractors = indices[:n_distractors+1]  # Get the first n_distractors indices
            if meaning_patient not in distractors: # Ensure the selected meaning is included in the distractors
                distractors = torch.cat((distractors[:n_distractors], torch.tensor([meaning_patient])))
            distractors = distractors[torch.randperm(len(distractors))]  # Shuffle the distractors

            sample["system_prompts"] = {
                "patient": [self.world["system_prompts"]["patient"][i] for i in distractors.tolist()],
                "doctor": self.world["system_prompts"]["doctor"]
            }
            prior = self.world["prior"][distractors, :, :]
            sample["logprior"] = torch.log(prior / prior.sum())
            sample["meaning_patient"] = torch.tensor(distractors.tolist().index(sample["meaning_patient"].item()))
            sample["original_meaning_patient"] = sample["meaning_patient"].item()  # Store the original meaning index
            sample["original_meaning_doctor"] = sample["meaning_doctor"].item()  # Store the original meaning index

            yield sample


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
            prompt += "You are experiencing the following symptoms:\n"
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
            f"You are experiencing the following symptoms:\n"
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

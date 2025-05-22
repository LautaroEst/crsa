


from copy import deepcopy
import json

import numpy as np


class MDDialDataset:

    data_dir = "data/mddial/"

    def __init__(self, prompt_style, split="train"):
        self.prompt_style = prompt_style
        if split not in ["train", "test"]:
            raise ValueError("split must be either 'train' or 'test'")
        self.data, self.world = self._load_data(f"{split}.json")

    def _load_data(self, filename):
        with open(self.data_dir + filename, "r") as f:
            dialogs = json.load(f)
        data = {"dialogues": [dialogs[f"Dialog {i}"] for i in range(1, len(dialogs) + 1)]}
        with open(self.data_dir + "disease.txt", "r") as f:
            diseases = [line.strip() for line in f.readlines()]
        with open(self.data_dir + "disease_symptoms.txt", "r") as f:
            disease2symptoms = json.load(f)
        id2symptoms = {}
        for i, d in enumerate(diseases, start=1):
            id2symptoms[f"symptoms_group_{i}"] = disease2symptoms[d]
        self.id2symptoms = id2symptoms
        self.disease2symptoms = disease2symptoms

        d_count = {}
        valid_data = []
        for i, dialog in enumerate(data["dialogues"]):
            sample = {}
            found = False
            for d in diseases:
                if d in dialog[-1]["doctor"]:
                    utterances = []
                    for turn in dialog:
                        utterances.append({"speaker": "patient", "content": turn["patient"]})
                        utterances.append({"speaker": "doctor", "content": turn["doctor"]})
                    sample["dialog_id"] = i
                    sample["dialog"] = utterances
                    sample["target_disease"] = d
                    sample["symptoms"] = disease2symptoms[d]
                    sample["symptoms_id"] = f"symptoms_group_{diseases.index(d) + 1}"
                    found = True
                    if d in d_count:
                        d_count[d] += 1
                    else:
                        d_count[d] = 1
                    break
            if found:
                valid_data.append(sample)

        prior = np.zeros((len(diseases), 1, len(diseases)))
        for i, d in enumerate(diseases):
            prior[i, 0, i] = d_count[d]
        prior = prior / np.sum(prior)

        world = {
            "diseases": diseases,
            "symptoms": [f"symptoms_group_{i}" for i in range(1, len(diseases) + 1)],
            "prior": prior,
        }

        doctor_utterances = []
        patient_utterances = []
        for sample in valid_data:
            for turn in sample["dialog"]:
                if turn["speaker"] == "patient" and turn["content"] not in patient_utterances:
                    patient_utterances.append(turn["content"])
                elif turn["speaker"] == "doctor" and turn["content"] not in doctor_utterances:
                    doctor_utterances.append(turn["content"])
        world["patient_utterances"] = patient_utterances
        world["doctor_utterances"] = doctor_utterances

        shots_ids = [5, 43, 1204, 864]
        shots = []
        for i in shots_ids:
            record = deepcopy(valid_data[i])
            shots.append({
                "dialog_id": record["dialog_id"],
                "symptoms": record["symptoms"],
                "utterances": record["dialog"],
                "disease": record["target_disease"],
            })
        self.shots = shots

        return valid_data, world

    def __getitem__(self, idx):
        record = self.data[idx]
        return {
            "dialog_id": record["dialog_id"],
            "symptoms": record["symptoms"],
            "symptoms_id": record["symptoms_id"],
            "utterances": record["dialog"],
            "disease": record["target_disease"],
        }

    def iter_samples(self):
        for record in self.data:
            yield {
                "dialog_id": record["dialog_id"],
                "symptoms": record["symptoms"],
                "symptoms_id": record["symptoms_id"],
                "utterances": record["dialog"],
                "disease": record["target_disease"],
            }

    def __len__(self):
        return len(self.data)
    

    def create_prompts_from_past_utterances(self, past_utterances, speaker):
        if speaker == "patient":
            # Patient speaks. Create prompt from past utterances for each possible set of symptoms
            prompts = []
            for symptoms_id in self.world["symptoms"]:
                symptoms = self.id2symptoms[symptoms_id]
                messages = [{'role': 'system', 'content': self._create_patient_system_prompt(symptoms)}]
                for turn in past_utterances:
                    if turn["speaker"] == "patient":
                        messages.append({'role': 'assistant', 'content': turn["content"]})
                    else:
                        messages.append({'role': 'user', 'content': turn["content"]})
                prompts.append(self.prompt_style.apply(messages))
        elif speaker == "doctor":
            # Doctor speaks. Create prompt from past utterances for each possible set of symptoms
            prompts = []
            for symptoms_id in self.world["symptoms"]:
                symptoms = self.id2symptoms[symptoms_id]
                messages = [{'role': 'system', 'content': self._create_doctor_system_prompt()}]
                for turn in past_utterances:
                    if turn["speaker"] == "doctor":
                        messages.append({'role': 'assistant', 'content': turn["content"]})
                    else:
                        messages.append({'role': 'user', 'content': turn["content"]})
                prompts.append(self.prompt_style.apply(messages))
        return prompts

    def _create_patient_system_prompt(self, symptoms):
        system_prompt = (
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
            system_prompt += ", ".join(example['symptoms']) + "\n"
            for turn in example["utterances"]:
                if turn["speaker"] == "patient":
                    system_prompt += f"Assistant: {turn['content']}\n"
                else:
                    system_prompt += f"User: {turn['content']}\n"
            system_prompt += "\n"
        system_prompt += (
            "Now, participate in a real conversation with the user. "
            "You are experiencing the following symptoms:\n"
        )
        system_prompt += ", ".join(symptoms) + "\n"
        return system_prompt
    
    def _create_doctor_system_prompt(self):
        system_prompt = (
            "You are an assistant that simulates to be a doctor "
            "who is diagnosing a patient based on the symptoms that he or she describes. "
            "You can ask questions to the patient, but ultimately, "
            "you have provide a diagnosis based on the symptoms described by the patient.\n\n"
        )
        for example in self.shots:
            system_prompt += (
                "Here is an example of a conversation "
                "between the assitant (i.e., the doctor) and the user (i.e., the patient). "
                "The patient is experiencing the following symptoms:\n"
            )
            for turn in example["utterances"]:
                if turn["speaker"] == "patient":
                    system_prompt += f"User: {turn['content']}\n"
                else:
                    system_prompt += f"Assistant: {turn['content']}\n"
            system_prompt += "\n"
        system_prompt += (
            "Now, participate in a real conversation with the user. "
            "You can ask questions to the patient, but ultimately, "
            "you have provide a diagnosis based on the symptoms described by the patient.\n"
        )
        return system_prompt

            
    def create_category_prompt_from_dialog(self, utterances, symptoms):
        messages = [{"role": "system", "content": self._create_patient_system_prompt(symptoms)}]
        for utterance in utterances[:-1]:
            if utterance["speaker"] == "patient":
                messages.append({"role": "assistant", "content": utterance["content"]})
            else:
                messages.append({"role": "user", "content": utterance["content"]})
        category_prompt = self.prompt_style.apply(messages)
        
        if utterances[-1]["speaker"] != "doctor":
            raise ValueError("The last utterance must be from the doctor.")
        last_utterance = utterances[-1]["content"]
        for disease in self.world["diseases"]:
            if " " + disease in last_utterance:
                last_utterance = last_utterance.replace(disease, "{disease}")
                break
        endings = [
            self.prompt_style.apply([
                {"role": "user", "content": last_utterance.format(disease=disease)}
            ]) for disease in self.world["diseases"]
        ]
        return category_prompt, endings



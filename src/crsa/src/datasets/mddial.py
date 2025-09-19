


from copy import deepcopy
import json
import re

import numpy as np
from tqdm import tqdm


class MDDialDataset:

    data_dir = "data/mddial/"

    YES_ANSWERS = [
        "Yes, sometimes",
        "I am experiencing that sometimes",
        "Yes Doctor, I am feeling that as well",
        "Yes most of the times"
    ]

    NO_ANSWERS = [
        "No, I don't have that",
        "No, I never had anything like that.",
        "Well not in my knowledge",
        "Not that I know of",
    ]

    TRAIN_SAMPLES = 1878

    def __init__(self, prompt_style, split="train"):
        self.prompt_style = prompt_style
        if split not in ["train", "test"]:
            raise ValueError("split must be either 'train' or 'test'")
        self.data, self.world, self.shots = self._load_data(f"{split}.json")
        self.sampled_indices = np.random.permutation(sorted(self.data.keys()))

    def _read_symptoms(self):
        with open(self.data_dir + "symptom.txt", "r") as f:
            symptoms = [line.strip() for line in f.readlines()]
        return symptoms

    def _read_diseases(self):
        with open(self.data_dir + "disease.txt", "r") as f:
            diseases = [line.strip() for line in f.readlines()]
        return diseases
    
    def _get_disease_name(self, doctor_utterance, diseases):
        match = re.search(r"((?:In that case, you have|This could probably be|I believe you are having from|Ok, this means you might be having)) ([^\.]+)\.", doctor_utterance)
        if match:
            template = match.group(1).strip() + " {disease}."
            disease_name = match.group(2).strip()
            # Check if the disease name is in the list of diseases
            if disease_name in diseases:
                return template, disease_name
        return None, None
    
    def _get_symptoms(self, sample, symptoms):
        # First check explicit symptoms
        valid_explicit_symptoms = []
        match = re.search(r"(?:Recently, I am experiencing|I have been feeling|Hi Doctor, I am having|I have) ([^.,]+)", sample[0]["patient"])
        if match:
            explicit_symptom = match.group(1).strip()
            if explicit_symptom in symptoms:
                valid_explicit_symptoms.append(explicit_symptom)
            else:
                explicit_symptoms = re.findall("(" + "|".join(symptoms) + ")", explicit_symptom)
                if all(symptom in symptoms for symptom in explicit_symptoms):
                    valid_explicit_symptoms.extend(explicit_symptoms)
                else:
                    raise ValueError(f"Explicit symptom '{explicit_symptoms}' not found in symptoms list.")
        
        # Then check implicit symptoms
        valid_implicit_symptoms = []
        negated_symptoms = []
        for i, turn in enumerate(sample):
            # patient answers
            if i != 0:
                if turn["patient"] in self.YES_ANSWERS:
                    valid_implicit_symptoms.append(potential_symptom)
                elif turn["patient"] in self.NO_ANSWERS:
                    negated_symptoms.append(potential_symptom)
                else:
                    raise ValueError(f"Unexpected patient response: {turn['patient']}")
            # doctor asks
            if i != len(sample) - 1:
                _, potential_symptom = self._get_template_and_symptom(turn["doctor"])
                if not potential_symptom:
                    raise ValueError(f"Could not find potential symptom in doctor utterance: {turn['doctor']}")
        
        return valid_explicit_symptoms, valid_implicit_symptoms, negated_symptoms
    
    def _get_template_and_symptom(self, doctor_utterance):
        match = re.search(r"((?:Is it\? Then do you experience|In that case, do you have any|What about|Oh, do you have any)) ([^.,]+)\?", doctor_utterance)
        if match:
            template = match.group(1).strip() + " {symptom}?"
            symptom = match.group(2).strip()
        else:
            raise ValueError(f"Could not find template and symptom in doctor utterance: {doctor_utterance}")
        return template, symptom
    
    def _init_prior(self, dialogs, diseases, symptoms, disease2symptoms):
        
        prior = np.zeros((len(diseases), 1, len(diseases)), dtype=float)
        symptom2disease = {s_idx: [d_idx for d_idx, d in enumerate(diseases) if disease2symptoms[d][s_idx]] for s_idx, symptom in enumerate(symptoms)}
        for dialog_id, dialog in dialogs.items():
            for s_idx in dialog["symptoms"]:
                prior[dialog['disease'],0,symptom2disease[s_idx]] += 1.0
        prior += 1e-4  # smoothing
        prior = prior / np.sum(prior)
        return prior

    def _load_data(self, filename):
        diseases = self._read_diseases()
        symptoms = self._read_symptoms()

        with open(self.data_dir + filename, "r") as f:
            data = json.load(f)
        
        dialogs = {}
        disease2symptoms = {}
        for dialog_id in range(1,len(data)+1):
            sample = data[f"Dialog {dialog_id}"]
            
            # Read disease name from the last doctor's utterance
            diagnostic_template, disease = self._get_disease_name(sample[-1]["doctor"], diseases)
            if not disease:
                continue

            # Read symptoms from the sample
            valid_explicit_symptoms, valid_implicit_symptoms, negated_symptoms = self._get_symptoms(sample, symptoms)
            if disease not in disease2symptoms:
                disease2symptoms[disease] = np.zeros(len(symptoms), dtype=bool)
            for symptom in valid_explicit_symptoms + valid_implicit_symptoms:
                disease2symptoms[disease][symptoms.index(symptom)] = True

            # Read utterances
            utterances = []
            for t, turn in enumerate(sample):
                utterances.append({
                    "speaker": "patient",
                    "content": turn["patient"],
                    "dialog_act": "explicit_symptom" if t == 0 else "implicit_symptom"
                })

                utterance = {
                    "speaker": "doctor",
                    "content": turn["doctor"],
                }
                if t != len(sample) - 1:  # Last turn is diagnosis
                    template, symptom = self._get_template_and_symptom(turn["doctor"])
                    utterance["dialog_act"] = {
                        "type": "question",
                        "template": template,
                        "symptom": symptom,
                    }
                else:
                    utterance["dialog_act"] = {
                        "type": "diagnosis",
                        "template": diagnostic_template,
                        "disease": disease,
                    }
                utterances.append(utterance)

            dialogs[dialog_id] = {
                "idx": dialog_id,
                "disease": diseases.index(disease),
                "valid_explicit_symptoms": [symptoms.index(s) for s in valid_explicit_symptoms],
                "valid_implicit_symptoms": [symptoms.index(s) for s in valid_implicit_symptoms],
                "negated_symptoms": [symptoms.index(s) for s in negated_symptoms],
                "symptoms": [symptoms.index(s) for s in valid_explicit_symptoms + valid_implicit_symptoms],
                "utterances": utterances,
            }

        world = {
            "diseases": diseases,
            "symptoms": symptoms,
            "disease2symptoms": np.stack([disease2symptoms[disease] for disease in diseases], axis=0),
            "prior": self._init_prior(dialogs, diseases, symptoms, disease2symptoms),
            "doctor_utterances": deepcopy(symptoms),  # Doctor can ask about any symptom
            "patient_utterances": ["yes", "no"],  # Patient can only answer yes or no
        }

        shots_ids = [8, 1706]
        shots = [dialogs[i] for i in shots_ids]

        return dialogs, world, shots
            

    def __getitem__(self, idx):
        return deepcopy(self.data[idx])

    def iter_samples(self):
        for idx in tqdm(self.sampled_indices):
            sample = self[idx]
            yield sample

    def __len__(self):
        return len(self.data)
    
    @property
    def indices(self):
        return sorted(self.data.keys())

    def create_prompts_from_past_utterances(self, past_turns, current_turn):
        
        if current_turn["speaker"] == "patient":
            # Patient speaks. Create prompt from past utterances for each possible set of symptoms
            prompts = []
            for diseases_id in range(len(self.world["diseases"])):
                symptoms = [s for i, s in enumerate(self.world["symptoms"]) if self.world["disease2symptoms"][diseases_id,i]]
                messages = [{'role': 'system', 'content': self._create_patient_system_prompt(symptoms)}]
                for turn in past_turns:
                    if turn["speaker"] == "patient":
                        messages.append({'role': 'assistant', 'content': turn["content"]})
                    else:
                        messages.append({'role': 'user', 'content': turn["content"]})
                prompts.append(self.prompt_style.apply(messages))
            if current_turn["content"] in self.YES_ANSWERS:
                endings = [
                    self.prompt_style.apply([{"role": "assistant", "content": current_turn["content"]}]), 
                    self.prompt_style.apply([{"role": "assistant", "content": np.random.choice(self.NO_ANSWERS)}])
                ]
                true_ending_idx = 0
            elif current_turn["content"] in self.NO_ANSWERS:
                endings = [
                    self.prompt_style.apply([{"role": "assistant", "content": np.random.choice(self.YES_ANSWERS)}]), 
                    self.prompt_style.apply([{"role": "assistant", "content": current_turn["content"]}])
                ]
                true_ending_idx = 1
            else:
                raise ValueError(f"Unexpected patient response: {current_turn['content']}")

        elif current_turn["speaker"] == "doctor":
            # Doctor speaks. Create prompt from past utterances.
            messages = [{'role': 'system', 'content': self._create_doctor_system_prompt()}]
            for turn in past_turns:
                if turn["speaker"] == "doctor":
                    messages.append({'role': 'assistant', 'content': turn["content"]})
                else:
                    messages.append({'role': 'user', 'content': turn["content"]})
            prompts = [self.prompt_style.apply(messages)]
            endings = [self.prompt_style.apply([{"role": "assistant", "content": current_turn["dialog_act"]["template"].format(symptom=s)}]) for s in self.world["symptoms"]]
            true_ending_idx = self.world["symptoms"].index(current_turn["dialog_act"]["symptom"])

        return prompts, endings, true_ending_idx
    
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
            system_prompt += ", ".join([self.world['symptoms'][s] for s in example['symptoms']]) + "\n"
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
        symptoms_str = ", ".join([self.world["symptoms"][s] for s in symptoms])
        messages = [{"role": "system", "content": self._create_patient_system_prompt(symptoms_str)}]
        for utterance in utterances[:-1]:
            if utterance["speaker"] == "patient":
                messages.append({"role": "assistant", "content": utterance["content"]})
            else:
                messages.append({"role": "user", "content": utterance["content"]})
        category_prompt = self.prompt_style.apply(messages)
        
        if utterances[-1]["speaker"] != "doctor":
            raise ValueError("The last utterance must be from the doctor.")
        endings = [
            self.prompt_style.apply([
                {"role": "user", "content": utterances[-1]["dialog_act"]["template"].format(disease=disease)}
            ]) for disease in self.world["diseases"]
        ]
        return category_prompt, endings

            
    



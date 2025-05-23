
import pandas as pd
from scipy.special import softmax


class Listener:

    def __init__(self, system_prompt, categories, past_utterances, llm):
        self.categories = categories
        self.system_prompt = system_prompt
        self.past_utterances = past_utterances
        self.llm = llm
        self.has_not_run_yet = True

    def run(self, new_utt):
        prompt = self.create_prompt(new_utt)
        endings = self.create_endings()
        self._logits = self.llm.predict(prompt=prompt, endings=endings)
        self.has_not_run_yet = False

    def create_prompt(self, new_utt):
        prompt = self.llm.prompt_style.apply([{"role": "system", "content": self.system_prompt}])
        for turn in self.past_utterances:
            if turn["speaker"] == "S":
                prompt += self.llm.prompt_style.apply([{"role": "assistant", "content": turn["utterance"]}])
            elif turn["speaker"] == "L":
                prompt += self.llm.prompt_style.apply([{"role": "user", "content": turn["utterance"]}])
        prompt += self.llm.prompt_style.apply([{"role": "user", "content": new_utt}])
        return prompt
    
    def create_endings(self):
        endings = []
        for category in self.categories:
            endings.append(self.llm.prompt_style.apply([{"role": "assistant", "content": category}]))
        return endings
    
    @property
    def as_df(self):
        posteriors = softmax(self._logits, axis=0)
        return pd.Series(posteriors, index=self.categories)


class Speaker:

    def __init__(self, system_prompt, utterances, past_utterances, llm):
        self.system_prompt = system_prompt
        self.utterances = utterances
        self.past_utterances = past_utterances
        self.llm = llm
        self.has_not_run_yet = True

    def create_prompt(self):
        prompt = self.llm.prompt_style.apply([{"role": "system", "content": self.system_prompt}])
        for turn in self.past_utterances:
            if turn["speaker"] == "S":
                prompt += self.llm.prompt_style.apply([{"role": "user", "content": turn["utterance"]}])
            elif turn["speaker"] == "L":
                prompt += self.llm.prompt_style.apply([{"role": "assistant", "content": turn["utterance"]}])
        return prompt
    
    def create_endings(self):
        endings = []
        for utterance in self.utterances:
            endings.append(self.llm.prompt_style.apply([{"role": "assistant", "content": utterance}]))
        return endings

    def run(self):
        prompt = self.create_prompt()
        endings = self.create_endings()
        self._logits = self.llm.predict(prompt=prompt, endings=endings)
        self.has_not_run_yet = False

    @property
    def as_df(self):
        posteriors = softmax(self._logits, axis=0)
        return pd.Series(posteriors, index=self.utterances)
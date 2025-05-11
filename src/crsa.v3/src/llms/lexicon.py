
import numpy as np


class LLMLexicon:

    def __init__(self, dataset, llm, num_utt=10, max_utt_length=20, top_k=50, top_p=0.95):
        self.dataset = dataset
        self.llm = llm
        self.num_utt = num_utt
        self.max_utt_length = max_utt_length
        self.top_k = top_k
        self.top_p = top_p

    def create_from_past_utterances_and_meanings(self, past_utterances, speaker, meanings):
        utterances = []
        prompts = []
        for meaning in meanings:
            prompt = self.dataset.create_prompt_from_meaning_and_past(meaning, past_utterances, agent=speaker)
            prompts.append(prompt)
            responses = []
            for i in range(self.num_utt):
                responses.append(self.llm.generate(prompt, max_new_tokens=self.max_utt_length, top_k=self.top_k, top_p=self.top_p))
            utterances.extend(responses)

        lexicon = []
        for prompt in prompts:
            lexicon.append(self.llm.predict(prompt, utterances))
        lexicon = np.exp(np.vstack(lexicon).T)
        lexicon = lexicon / lexicon.sum()
        return lexicon, utterances
            
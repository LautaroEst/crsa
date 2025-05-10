
import numpy as np
from .memoryless_rsa import MemorylessRSA, MemorylessRSATurn


def create_lexicon(system_prompt, meanings, utterances, past_utterances, llm, speaker_now="A"):
    lexicon = np.zeros((len(utterances), len(meanings)))
    for meaning_idx, meaning in enumerate(meanings):
        prompt = llm.prompt_style.apply([{"role": "system", "content": system_prompt.format(meaning=meaning)}])
        for turn in past_utterances:
            if turn["speaker"] == speaker_now:
                prompt += llm.prompt_style.apply([{"role": "assistant", "content": turn["utterance"]}])
            else:
                prompt += llm.prompt_style.apply([{"role": "user", "content": turn["utterance"]}])
        endings = []
        for utterance in utterances:
            endings.append(llm.prompt_style.apply([{"role": "assistant", "content": utterance}]))
        logits = llm.predict(prompt=prompt, endings=endings)
        lexicon[:, meaning_idx] = np.exp(logits)
    lexicon /= lexicon.sum()
    return lexicon


class LLMRSA(MemorylessRSA):

    def __init__(self, meanings_A, meanings_B, system_prompt_template_A, system_prompt_template_B, categories, utterances, prior, llm="llama3", alpha=1.0, costs=None, pov="listener", max_depth=100, tolerance=1e-5):
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.system_prompt_template_A = system_prompt_template_A
        self.system_prompt_template_B = system_prompt_template_B
        self.categories = categories
        self.utterances = utterances
        self.prior = prior # P(a,b,y)
        self.llm = llm
        self.alpha = alpha
        self.costs = costs if costs is not None else np.zeros(len(utterances))
        self.pov = pov
        self.max_depth = max_depth
        self.tolerance = tolerance
        self.past_utterances = []
        self.speaker_now = None
        self.turns_history = []

    def _run_turn(self, past_utterances, speaker_now):
        prior = self.prior.copy() if speaker_now == "A" else self.prior.copy().transpose(1, 0, 2)
        meanings_S = self.meanings_A if speaker_now == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker_now == "A" else self.meanings_A

        if (speaker_now == "A" and self.pov == "listener") or (speaker_now == "B" and self.pov == "speaker"):
            lexicon = create_lexicon(
                system_prompt=self.system_prompt_template_A,
                meanings=self.meanings_A,
                utterances=self.utterances,
                past_utterances=past_utterances,
                llm=self.llm,
                speaker_now=speaker_now
            )
        else:
            lexicon = create_lexicon(
                system_prompt=self.system_prompt_template_B,
                meanings=self.meanings_B,
                utterances=self.utterances,
                past_utterances=past_utterances,
                llm=self.llm,
                speaker_now=speaker_now
            )
        
        model = MemorylessRSATurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=self.utterances,
            prior=prior,
            lexicon=lexicon,
            alpha=self.alpha,
            costs=self.costs,
            pov=self.pov,
            max_depth=self.max_depth,
            tolerance=self.tolerance
        )
        model.run()
        return model

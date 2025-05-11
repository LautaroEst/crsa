


from itertools import product
import numpy as np


class FindA1Dataset:

    def __init__(self, prompt_style):
        self.n_samples = 5
        self.world = self._create_world()
        self.prompt_style = prompt_style

    def create_prompt_from_meaning_and_past(self, meaning, past_utterances, speaker="A"):
        meaning = ", ".join(meaning)
        if speaker == "A":
            messages = [{"role": "system", "content": (
                "You are playing a collaborative game with the user. Both of you are given a set of 4 cards "
                "but you can see only the letter on the cards and the user can see only the number. Letters can be "
                "either A or B and numbers can be either 1 or 2. The goal of the game is to find the position of the card A1. "
                "You and the user can only say one word at a time representing the position (1-4) of a card "
                "and 0 if there is no card A1. There is at most one card A1.\n\n"

                "Examples:\n\n"

                "You can see the following letters: A, A, B, A\n"
                "User: Let's begin the game.\n"
                "Assistant: Position 1\n"
                "User: Position 3\n"
                "Assistant: Position 2\n"
                "User: Position 4\n"
                "Assistant: Position 4\n\n"

                "You can see the following letters: B, A, B, A\n"
                "User: Let's begin the game.\n"
                "Assistant: Position 4\n"
                "User: Position 3\n"
                "Assistant: Position 2\n"
                "User: Position 0\n"
                "Assistant: Position 0\n\n"

                "You can see the following letters: {meaning}\n\n"
            )}, {"role": "user", "content": "Let's begin the game."}]
        elif speaker == "B":
            messages = [{"role": "system", "content": (
                "You are playing a collaborative game with the user. Both of you are given a set of 4 cards "
                "but you can see only the numbers on the cards and the user can see only the letter. Numbers can be "
                "either 1 or 2 and letters can be either A or B. The goal of the game is to find the position of the card A1. "
                "You and the user can only say one word at a time representing the position (1-4) of a card "
                "and 0 if there is no card A1. There is at most one card A1.\n\n"
                
                "Examples:\n\n"
                
                "You can see the following numbers: 1, \n"
                "User: Let's begin the game.\n"
                "Assistant: Position 1\n"
                "User: Position 3\n"
                "Assistant: Position 2\n"
                "User: Position 4\n"
                "Assistant: Position 4\n\n"

                "You can see the following letters: B, B, B, A\n"
                "User: Let's begin the game.\n"
                "Assistant: Position 4\n"
                "User: Position 3\n"
                "Assistant: Position 0\n"
                "User: Position 0\n\n"

                "Now, start the game.\n\nYou can see the following letters: {meaning}\n\n"
            )}, {"role": "user", "content": "Let's begin the game."}]
        for utterance in past_utterances:
            role = "assistant" if utterance["speaker"] == speaker else "user"
            messages.append({"role": role, "content": utterance["content"]})
        prompt += self.prompt_style.apply(messages)
        prompt += self.prompt_style.ASSISTANT_HEADER
        return prompt

    def _create_world(self):
        world = {
            "meanings_A": [", ".join(l) for l in product("AB", repeat=self.game_size)],
            "meanings_B": [", ".join(n) for n in product("12", repeat=self.game_size)],
            "categories": ["There is no A1 card"] + [f"The card A1 is at possition {i+1}" for i in range(self.game_size)],
            "utterances": [f"The card A1 is at possition {i+1}" for i in range(self.game_size)],
        }

        prior = np.zeros((2**self.game_size,2**self.game_size,self.game_size+1))
        for i in range(2**self.game_size):
            for j in range(2**self.game_size):
                # check if (meaninings_A[i][k] == "A" and meanings_B[j][k] == "1") happens only once for a fixed i,j for k in range(p)
                count = 0
                A1_idx = None
                for k in range(self.game_size):
                    if world["meanings_A"][i][k] == "A" and world["meanings_B"][j][k] == "1":
                        count += 1
                        A1_idx = k
                if count == 1:
                    prior[i,j,A1_idx+1] = 1
                elif count == 0:
                    prior[i,j,0] = 1
        prior = prior / np.sum(prior)

        lexicon_A = np.zeros((self.game_size,len(world["meanings_A"])))
        for u, utt in enumerate(world["utterances"]):
            for i, meaning in enumerate(world["meanings_A"]):
                if meaning[int(u)] == "A":
                    lexicon_A[u,i] = 1
        lexicon_A[:,-1] = 1

        lexicon_B = np.zeros((self.game_size,len(world["meanings_B"])))
        for u, utt in enumerate(world["utterances"]):
            for i, meaning in enumerate(world["meanings_B"]):
                if meaning[int(u)] == "1":
                    lexicon_B[u,i] = 1
        lexicon_B[:,-1] = 1

        world["prior"] = prior
        world["lexicon_A"] = lexicon_A
        world["lexicon_B"] = lexicon_B
        world["costs"] = np.zeros(len(world["utterances"]))
        return world

    def _sample_from_prior(self):
        flat_p = self.world["prior"].flatten()
        sampled_index = np.random.choice(len(flat_p), p=flat_p)
        sampled_indices = np.unravel_index(sampled_index, self.world["prior"].shape)
        sampled_meaning_A = self.world["meanings_A"][sampled_indices[0]]
        sampled_meaning_B = self.world["meanings_B"][sampled_indices[1]]
        sampled_cat = self.world["categories"][sampled_indices[2]]
        return sampled_meaning_A, sampled_meaning_B, sampled_cat
    
    def iter_samples(self):
        for idx in range(self.n_samples):
            sampled_meaning_A, sampled_meaning_B, sampled_cat = self._sample_from_prior()
            yield idx, sampled_meaning_A, sampled_meaning_B, sampled_cat

    def __len__(self):
        return self.n_samples
            
        
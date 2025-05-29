


from itertools import product
import numpy as np
import torch


class FindA1Dataset:

    def __init__(self, game_size, n_rounds: int = 1000):
        self.n_rounds = n_rounds
        self.game_size = game_size
        self.world = self._create_world()
        self.data = self._load_data()
        self.sampled_indices = torch.randperm(self.n_rounds)

    def _load_data(self):
        data = []
        indices = torch.arange(self.n_rounds)
        for idx in indices:
            sampled_meaning_A, sampled_meaning_B, sampled_cat = self._sample_from_prior()
            data.append({
                "idx": idx,
                "meaning_A": sampled_meaning_A.squeeze(),
                "meaning_B": sampled_meaning_B.squeeze(),
                "target": sampled_cat.squeeze(),
                "utterances": None,
            })
        return data

    def _create_world(self):
        world = {
            "meanings_A": ["".join(l) for l in product("AB", repeat=self.game_size)],
            "meanings_B": ["".join(n) for n in product("12", repeat=self.game_size)],
            "targets": ["There is no A1 card"] + [f"The card A1 is at position {i+1}" for i in range(self.game_size)],
            "utterances": [f"Position {i+1}" for i in range(self.game_size)],
        }

        prior = torch.zeros((2**self.game_size,2**self.game_size,self.game_size+1))
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
        prior = prior / torch.sum(prior)
        logprior = torch.log(prior)

        lexicon_A = torch.zeros((self.game_size,len(world["meanings_A"])))
        for u, utt in enumerate(world["utterances"]):
            for i, meaning in enumerate(world["meanings_A"]):
                if meaning[int(u)] == "A":
                    lexicon_A[u,i] = 1
        lexicon_A[:,-1] = 1

        lexicon_B = torch.zeros((self.game_size,len(world["meanings_B"])))
        for u, utt in enumerate(world["utterances"]):
            for i, meaning in enumerate(world["meanings_B"]):
                if meaning[int(u)] == "1":
                    lexicon_B[u,i] = 1
        lexicon_B[:,-1] = 1

        world["logprior"] = logprior
        world["lexicon_A"] = lexicon_A
        world["lexicon_B"] = lexicon_B
        return world

    def _sample_from_prior(self):
        sampled_index = torch.multinomial(torch.exp(self.world["logprior"]).flatten(), 1)
        sampled_indices = torch.unravel_index(sampled_index, self.world["logprior"].shape)
        return sampled_indices[0], sampled_indices[1], sampled_indices[2]
    
    def iter_samples(self):
        for idx in self.sampled_indices:
            yield self.data[idx]

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_rounds:
            raise IndexError("Index out of range")
        return self.data[idx]

    def __len__(self):
        return self.n_rounds
            
        
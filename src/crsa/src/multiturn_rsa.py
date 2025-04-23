
from .crsa import CRSA
from .crsa_turn import CRSATurn




class MultiTurnRSA(CRSA):

    def _run_turn(self, past_utterances, speaker="A", output_dir=None, verbose=False, prefix=""):
        meanings_S = self.meanings_A if speaker == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker == "A" else self.meanings_A
        utterances = self.utterances_A if speaker == "A" else self.utterances_B
        lexicon = self.lexicon_A if speaker == "A" else self.lexicon_B
        cost = self.cost_A if speaker == "A" else self.cost_B
        prior = self.prior.copy() if speaker == "A" else self.prior.copy().transpose(1, 0, 2)

        model = CRSATurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=utterances,
            prior=prior,
            lexicon=lexicon,
            past_utterances=[],
            ds=None,
            dl=None,
            cost=cost,
            alpha=self.alpha,
            max_depth=self.max_depth,
            tolerance=self.tolerance
        )
        model.run(output_dir, verbose, prefix)
        return model
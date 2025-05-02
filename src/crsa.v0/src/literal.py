
import numpy as np
from .crsa import CRSA
from .crsa_turn import CRSATurn
from .utils import INF
from scipy.special import softmax




class Literal(CRSA):

    def __init__(self, speaker_now, meanings_A, meanings_B, categories, utterances_A, utterances_B, lexicon_A, lexicon_B, prior, past_utterances, alpha, cost_A, cost_B):
        super().__init__(
            speaker_now, meanings_A, meanings_B, categories, utterances_A, utterances_B, lexicon_A, lexicon_B, prior, past_utterances, 
            cost_A, cost_B, alpha, max_depth=0, tolerance=0.0
        )

    def _run_turn(self, past_utterances, speaker="A", output_dir=None, verbose=False, prefix=""):
        meanings_S = self.meanings_A if speaker == "A" else self.meanings_B
        meanings_L = self.meanings_B if speaker == "A" else self.meanings_A
        utterances = self.utterances_A if speaker == "A" else self.utterances_B
        lexicon = self.lexicon_A if speaker == "A" else self.lexicon_B
        cost = self.cost_A if speaker == "A" else self.cost_B
        prior = self.prior.copy() if speaker == "A" else self.prior.copy().transpose(1, 0, 2)
        ds, dl = self._compute_dialog_model(past_utterances, speaker)

        model = CRSATurn(
            meanings_S=meanings_S,
            meanings_L=meanings_L,
            categories=self.categories,
            utterances=utterances,
            prior=prior,
            lexicon=lexicon,
            past_utterances=past_utterances,
            ds=ds,
            dl=dl,
            cost=cost,
            alpha=self.alpha,
            max_depth=self.max_depth,
            tolerance=self.tolerance
        )
        model.run(output_dir, verbose, prefix)
        
        # Compute the literal speaker
        mask = lexicon > 0
        log_lexicon = np.zeros_like(lexicon, dtype=float)
        log_lexicon[mask] = np.log(lexicon[mask])
        log_lexicon[~mask] = -INF
        literal_speaker = softmax(self.alpha * log_lexicon.T - cost.reshape(1,-1), axis=1)
        model.speaker.history.append(literal_speaker)
        
        return model
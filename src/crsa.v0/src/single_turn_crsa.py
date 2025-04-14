
from pathlib import Path
import numpy as np
from scipy.special import softmax
import logging
import pandas as pd
import yaml
import pickle

from .utils import (
    is_list_of_strings, 
    is_numeric_ndarray, 
    is_list_of_numbers, 
    is_positive_number, 
    is_positive_integer, 
    is_list_of_list_of_numbers,
    save_yaml,
    ZERO, INF,
)


class Listener:

    # Listener(u,w,b,y) = L(y|u,w,b)
    # Prior(a,b,y) = P(a,b,y)
    # Lexicon(u,w,a) = Lexicon_A(u,w,a)

    def __init__(self, categories, past_utterances, utterances, meanings_B, prior, lexicon):
        self.categories = categories
        self.meanings_B = meanings_B
        self.past_utterances = past_utterances
        self.utterances = utterances
        self.prior = prior
        self.lexicon = lexicon

        # Extract array from lexicon
        lexicon_arr = lexicon.to_numpy()
        groups = lexicon.columns.levels[0]
        n_groups = len(groups)
        n_features_per_group = len(lexicon.columns) // n_groups
        lexicon_arr = lexicon_arr.reshape(len(lexicon), n_groups, n_features_per_group)

        # Initialize with literal listener
        literal_listener = np.einsum('uwa,aby->uwby', lexicon_arr, prior)
        literal_listener = literal_listener / literal_listener.sum(axis=3, keepdims=True)
        self.history = [literal_listener]

    def update(self, speaker, ps):
        """
        Update the listener based on the speaker
        """
        pragmatic_listener = np.einsum("abyw,awu,aby->uwby", ps, speaker, self.prior)
        pragmatic_listener = pragmatic_listener / pragmatic_listener.sum(axis=3, keepdims=True)
        self.history.append(pragmatic_listener)

    def get_literal_as_df(self):
        return pd.DataFrame(self.history[0].reshape(-1,len(self.categories)), index=pd.MultiIndex.from_product([self.utterances, self.past_utterances, self.meanings_B], names=['utterances','past_utterances','meanings_B']), columns=self.categories)
    
    def get_literal_as_array(self):
        return self.history[0]
      
    @property
    def as_df(self):
        return pd.DataFrame(self.as_array.reshape(-1,len(self.categories)), index=pd.MultiIndex.from_product([self.utterances, self.past_utterances, self.meanings_B], names=['utterances','past_utterances','meanings_B']), columns=self.categories)
    
    @property
    def as_array(self):
        return self.history[-1]
    

class Speaker:

    # Speaker(a,w,u) = S(u|a,w)
    # Listener(u,w,b,y) = L(y|u,w,b)
    # Prior(a,b,y) = P(a,b,y)
    # Cost(u) = C(u)

    def __init__(self, meanings_A, past_utterances, utterances, prior, cost, alpha):
        self.meanings_A = meanings_A
        self.past_utterances = past_utterances
        self.utterances = utterances
        self.prior = prior
        self.cost = cost
        self.alpha = alpha

        literal_speaker = np.ones((len(meanings_A), len(past_utterances), len(utterances))) * np.nan
        self.history = [literal_speaker]

    def update(self, listener, ps):
        """
        Update the speaker based on the listener
        """

        # Log listener with -inf for zero probabilities
        mask = listener > 0
        log_listener = np.zeros_like(listener)
        log_listener[mask] = np.log(listener[mask])
        log_listener[~mask] = -INF

        # Safe x
        prior = self.prior.copy()
        prior[prior == 0] = ZERO
        cond_prior = prior / prior.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True) # Cond(a,b,y) = P(b,y|a) = P(a,b,y) / P(a)
        cond_prior[cond_prior == 0] = ZERO

        # Safe x/y        
        ps_num = ps.copy()
        ps_num[ps_num == 0] = ZERO
        ps_den = np.einsum("abyw,aby->aw", ps, cond_prior)[:,np.newaxis,np.newaxis,:]
        ps_frac = ps_num / ps_den
        exp_term = np.einsum("abyw,aby,uwby->awu", ps_frac, cond_prior, log_listener)
        pragmatic_speaker = softmax(self.alpha * (exp_term - self.cost.reshape(1,1,-1)), axis=2)
        self.history.append(pragmatic_speaker)

    @property
    def as_df(self):
        return pd.DataFrame(self.as_array, index=self.meanings_A, columns=self.utterances)
    
    @property
    def as_array(self):
        return self.history[-1]


class CRSAGain:

    # Ps(a,b,y,w) = Ps(w|a,b,y)
    # Prior(a,b,y) = P(a,b,y)
    # Speaker(a,w,u) = S(u|a,w)
    # Listener(u,w,b,y) = L(y|u,w,b)
    # Cost(u) = C(u)
    # Gain = alpha * E[L(Y|U,WA,MB)] + Hs(U|MA,WA)

    def __init__(self, ps):
        self.ps = ps.get_value()
        self.cond_entropy_history = []
        self.listener_value_history = []
        self.gain_history = []
        self.coop_index_history = []

    def H_S_of_U_given_MA_WA(self, prior, speaker):
        """
        Compute the conditional mutual information of the utterances given the meanings.

        Parameters
        ----------
        prior : np.array (meanings_A, meanings_B, categories)
            The prior probability of each meaning.
        speaker : np.array (meanings_A, utterances)
            The speaker probability of each meaning given each utterance.

        """

        # Compute safe x * log(x)
        mask = speaker > 0
        speaker_times_log_speaker = np.zeros_like(speaker) 
        speaker_times_log_speaker[mask] = np.log(speaker[mask])
        speaker_times_log_speaker[~mask] = ZERO # approximate x * log(x) to 0
        speaker_times_log_speaker = speaker * speaker_times_log_speaker

        # Safe x
        prior = prior.copy()
        prior[prior == 0] = ZERO
        
        # speaker = S(a,w,u) = S(u|a,w), prior = P(a,b,y) = P(a,b,y), ps = Ps(a,b,y,w) = Ps(w|a,b,y)
        cond_entropy = - np.einsum("awu,aby,abyw->", speaker_times_log_speaker, prior, self.ps) # Hs(U|MA,WA)
        self.cond_entropy_history.append(cond_entropy)
        return cond_entropy

    def expected_V_L_over_S(self, listener, speaker, prior, cost):
        """
        Compute the expected value of the listener over the speaker.

        Parameters
        ----------
        listener : np.array (utterances, meanings_B, categories)
            The listener probability of each meaning given each utterance.
        speaker : np.array (meanings_A, utterances)
            The speaker probability of each meaning given each utterance.
        prior : np.array (meanings_A, meanings_B, categories)
            The prior probability of each meaning.
        cost : np.array (utterances,)
            The cost of each utterance.
        """
        
        # Safe logarithm
        log_listener = np.zeros_like(listener)
        mask = listener > 0
        log_listener[mask] = np.log(listener[mask])
        log_listener[~mask] = -INF
        V_L = log_listener - cost.reshape(-1,1,1,1) # V_L(u,w,b,y) = log(P(y|u,w,b)) - C(u)

        # Safe x
        prior = prior.copy()
        prior[prior == 0] = ZERO
        Ps = self.ps.copy()
        Ps[Ps == 0] = ZERO
        
        # V_L = V_L(u,w,b,y), prior = P(a,b,y) = P(a,b,y), ps = Ps(a,b,y,w) = Ps(w|a,b,y)
        expected_V_L = np.einsum("uwby,aby,abyw->", V_L, prior, self.ps)
        self.listener_value_history.append(expected_V_L)
        return expected_V_L
    
    def compute_gain(self, listener, speaker, prior, cost, alpha):
        """
        Compute the gain function.

        Parameters
        ----------
        listener : np.array (U,M)
            The listener probability of each meaning given each utterance.
        speaker : np.array (M,U)
            The speaker probability of each meaning given each utterance.
        prior : np.array (M,)
            The prior probability of each meaning.
        cost : np.array (U,)
            The cost of each utterance.
        alpha : float
            The rationality parameter.
        """
        gain = alpha * self.expected_V_L_over_S(listener, speaker, prior, cost) + self.H_S_of_U_given_MA(prior, speaker)
        self.gain_history.append(gain)
        return gain
    
    def get_diff(self):
        if len(self.gain_history) < 2:
            return float("inf")
        return abs(self.gain_history[-1] - self.gain_history[-2]) / abs(self.gain_history[-2])



class Ps:

    # Ps(a,b,y,w) = P_S(w|a,b,y)

    def __init__(self, meanings_A, meanings_B, past_utterances):
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.past_utterances = past_utterances
        self.history = []
        self._value = None
    
    def get_value(self):
        if self._value is None:
            return np.ones((len(self.meanings_A), len(self.meanings_B), len(self.past_utterances)), dtype=float)
        return self._value

    def update(self, speaker, agent_speaking="a"):
        # speaker = S(a_or_b,w,u)
        self._value = np.einsum(f"{agent_speaking}wu,abyw->abyw", speaker, self.get_value())

    def as_array(self):
        pass



class SingleCRSA:
    """
    Y-Rational Speech Act (RSA) model from the listener's perspective

    Parameters
    ----------
    meanings : List[str]
        List of possible meanings
    utterances : List[str]
        List of possible utterances
    lexicon : Union[np.ndarray, list[list[int]]]
        Lexicon matrix of shape (len(utterances),  len((meanings))
    prior : Optional[Union[np.ndarray,list[int]]]
        Prior probability of meanings. If None, uniform prior is assumed.
    cost : Optional[Union[np.ndarray,list[float]]]
        Cost of utterances. If None, uniform cost (zero) is assumed.
    alpha : Optional[float]
        Rationality parameter
    max_depth : Optional[int]
        Maximum depth of the model (number of iterations)
    tolerance : Optional[float]
        Tolerance for convergence
    """

    def __init__(self, ps, meanings_A, meanings_B, categories, past_utterances, utterances, lexicon, prior=None, cost=None, alpha=1., max_depth=None, tolerance=1e-6):

        # Ps(a,b,y,w): Conditional prob for the past utterances
        self.ps = ps 

        # Check meanings and utterances
        if not is_list_of_strings(meanings_A) or not is_list_of_strings(meanings_B) or not is_list_of_strings(categories):
            raise ValueError("meanings and categories should be a list of strings")
        self.meanings_A = meanings_A
        self.meanings_B = meanings_B
        self.categories = categories
        if not is_list_of_strings(utterances):
            raise ValueError("utterances should be a list of strings")
        self.utterances = utterances
        if not is_list_of_strings(past_utterances):
            raise ValueError("past_utterances should be a list of strings")
        self.past_utterances = past_utterances

        # Check lexicon
        if not isinstance(lexicon, pd.DataFrame) or lexicon.shape != (len(utterances), len(meanings_A) * max(1,len(past_utterances))):
            raise ValueError("lexicon should be a pandas DataFrame of shape (len(utterances), len(meanings_A) * len(past_utterances))")
        self.lexicon = lexicon

        # Check prior
        if prior is None:
            prior = np.ones((len(meanings_A), len(meanings_B), len(categories)), dtype=float)
            prior /= prior.sum()
        else:
            try:
                prior = np.asarray(prior).astype(float)
            except:
                raise ValueError("prior should be an array-like object of shape (len(meanings_A), len(meanings_B), len(categories))")
        self.prior = prior

        # Check cost
        if cost is None:
            cost = np.zeros(len(utterances), dtype=float)
        elif is_numeric_ndarray(cost) and cost.shape == (len(utterances),):
            pass
        elif is_list_of_numbers(cost) and len(cost) == len(utterances):
            cost = np.asarray(cost).astype(float)
        else:
            raise ValueError("cost should be a list of floats or a numpy array of shape (len(utterances),)")
        self.cost = cost

        # Check alpha
        if not is_positive_number(alpha):
            raise ValueError("alpha should be a positive number")
        self.alpha = float(alpha)

        # Check max_depth and tolerance
        if not is_positive_integer(max_depth) and max_depth != float("inf"):
            raise ValueError("depth should be a positive integer or inf")
        if not is_positive_number(tolerance) and tolerance != 0:
            raise ValueError("tolerance should be a positive number or None")
        if max_depth == float("inf") and tolerance == 0:
            raise ValueError("Either max_depth or tolerance should be provided")
        self.max_depth = max_depth
        self.tolerance = tolerance

        self.listener = None
        self.speaker = None
        self.gain = None


    def run(self, output_dir: Path, verbose: bool = False, prefix: str = ""):
        """
        Run the RSA model for a given number of iterations

        Parameters
        ----------
        output_dir : Path
            Output directory to save the results
        verbose : bool
            If True, print the results to the console
        """
               
        # Configure logging
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter("%(message)s")
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        file_handler = logging.FileHandler(output_dir / f"{prefix}history.log", mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        # Log configuration
        logger.info(f"Running singleturn CRSA model for max depth {self.max_depth} and tolerance {self.tolerance:.2e}")
        logger.info("-" * 40)
        logger.info(
            f"\nLexicon:\n\n{self.lexicon}\n\n"
            f"Prior:\n\n{pd.DataFrame(self.prior.reshape(-1,len(self.categories)), index=pd.MultiIndex.from_product([self.meanings_A, self.meanings_B], names=['meanings_A','meanings_B']), columns=self.categories)}\n\n"
            f"Cost:\n\n{pd.Series(self.cost, index=self.utterances).to_string()}\n\n"
            f"Alpha: {self.alpha}\n"
        )
        logger.info("-" * 40 + "\n")
        
        # Init listener and speaker
        self.listener = Listener(self.categories, self.past_utterances, self.utterances, self.meanings_B, self.prior, self.lexicon)
        self.speaker = Speaker(self.meanings_A, self.past_utterances, self.utterances, self.prior, self.cost, self.alpha)
        self.ps = Ps(self.meanings_A, self.meanings_B, self.past_utterances)
        self.gain = CRSAGain(self.ps)
        gain = self.gain.compute_gain(self.listener.as_array, self.speaker.as_array, self.prior, self.cost, self.alpha)
        logger.info(f"Literal listener:\n{self.listener.get_literal_as_df()}\n\n")
        logger.info(f"Initial gain: {gain:.4f}\n")
        logger.info("-" * 40 + "\n")

        # Run the model for the given number of iterations
        i = 0
        while i < self.max_depth:
            # Update speaker
            self.speaker.update(self.listener.as_array, self.ps.as_array)
            logger.info(f"Pragmatic speaker at step {i+1}:\n{self.speaker.as_df}\n")
            
            # Update listener
            self.listener.update(self.speaker.as_array)
            logger.info(f"Pragmatic listener at step {i+1}:\n{self.listener.as_df}\n")

            # Check for convergence
            gain = self.gain.compute_gain(self.listener.as_array, self.speaker.as_array, self.prior, self.cost, self.alpha)
            logger.info(f"Step: {i+1} | Gain: {gain:.4f}")
            logger.info("\n" + "-" * 40 + "\n")
            if self.gain.get_diff() < self.tolerance:
                logger.info(f"Converged after {i+1} iterations")
                break
            i += 1

        # Close logging
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    @property
    def history(self):
        return {
            "listener": [pd.DataFrame(l, index=pd.MultiIndex.from_product([self.utterances, self.meanings_B], names=['utterances','meanings_B']), columns=self.categories) for l in self.listener.history],
            "speaker": [pd.DataFrame(s, index=self.meanings_A, columns=self.utterances) for s in self.speaker.history],
            "cond_entropy": self.gain.cond_entropy_history,
            "listener_value": self.gain.listener_value_history,
            "gain": self.gain.gain_history,
            "coop_index": self.gain.coop_index_history,
        }
    
    @history.setter
    def history(self, value):
        raise AttributeError("history is a read-only property")
    
    def save(self, output_dir: Path, prefix: str = ""):
        args = {
            "meanings_A": self.meanings_A,
            "meanings_B": self.meanings_B,
            "categories": self.categories,
            "utterances": self.utterances,
            "lexicon": self.lexicon.to_dict(orient="records"),
            "prior": self.prior.tolist(),
            "cost": self.cost.tolist(),
            "alpha": self.alpha,
            "max_depth": self.max_depth,
            "tolerance": self.tolerance
        }
        save_yaml(args, output_dir / f"{prefix}args.yaml")

        with open(output_dir / f"{prefix}history.pkl", "wb") as f:
            pickle.dump({
                "listeners": np.asarray(self.listener.history),
                "speakers": np.asarray(self.speaker.history),
                "cond_entropy": np.asarray(self.gain.cond_entropy_history),
                "listener_value": np.asarray(self.gain.listener_value_history),
                "gain": np.asarray(self.gain.gain_history),
                "coop_index": np.asarray(self.gain.coop_index_history),
            }, f)

    @classmethod
    def load(cls, output_dir: Path, prefix: str = ""):
        with open(output_dir / "args.yaml", "r") as f:
            args = yaml.safe_load(f)
        with open(output_dir / f"{prefix}history.pkl", "rb") as f:
            history = pickle.load(f)
        rsa = cls(**args)
        rsa.listener = Listener(rsa.categories, rsa.utterances, rsa.meanings_B, rsa.prior, rsa.lexicon)
        rsa.listener.history = [l for l in history["listeners"]]
        rsa.speaker = Speaker(rsa.meanings_A, rsa.utterances, rsa.prior, rsa.cost, rsa.alpha)
        rsa.speaker.history = [s for s in history["speakers"]]
        rsa.gain = CRSAGain()
        rsa.gain.cond_entropy_history = history["cond_entropy"].tolist()
        rsa.gain.listener_value_history = history["listener_value"].tolist()
        rsa.gain.gain_history = history["gain"].tolist()
        rsa.gain.coop_index_history = history["coop_index"].tolist()
        return rsa
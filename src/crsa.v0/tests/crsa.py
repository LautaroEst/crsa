
import numpy as np
from scipy.special import softmax
from ..src.crsa import CRSA


meanings_A = ["AA", "AB", "BA", "BB"]
meanings_B = ["11", "12", "21", "22"]
categories = ["None", "1st", "2nd"]

prior_list = [
    [[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
    [[0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
]

utterances_A = ["1st", "2nd"]
utterances_B = ["1st", "2nd"]

lexicon_A_list = [
    [1, 1, 0, 1],
    [1, 0, 1, 1],
]
lexicon_B_list = [
    [1, 1, 0, 1],
    [1, 0, 1, 1],
]

past_utterances = ["1st", "2nd"]

alpha = 2

def main():

    
    model = CRSA(
        meanings_A, 
        meanings_B, 
        categories, 
        utterances_A, 
        utterances_B, 
        lexicon_A_list, 
        lexicon_B_list, 
        prior_list, 
        past_utterances, 
        cost_A=None, 
        cost_B=None, 
        alpha=alpha, 
        max_depth=1, 
        tolerance=0.,
    )
    model.run()

    prior = model.prior
    lexicon_A = model.lexicon_A
    lexicon_B = model.lexicon_B

    ## TURN 1
    turn_1 = model.turns_history[0]
    # Literal listener
    l_0 = np.einsum('aby,ua->uby', prior, lexicon_A)
    l_0 = l_0 / l_0.sum(axis=-1, keepdims=True)
    # Speaker at step 1
    cond_prior = prior.copy()
    cond_prior = cond_prior / cond_prior.sum(axis=1, keepdims=True).sum(axis=2, keepdims=True)
    mask = l_0 > 0
    ll_0 = np.zeros_like(l_0)
    ll_0[mask] = np.log(l_0[mask])
    ll_0[~mask] = -np.inf
    s_1 = np.einsum('aby,uby->abyu', cond_prior, ll_0)
    s_1[np.isnan(s_1)] = 0
    s_1 = np.einsum('abyu->au', s_1)
    s_1 = softmax(s_1, axis=1)
    # Listener at step 1
    l_1 = np.einsum('aby,au->uby', prior, s_1)
    l_1 = l_1 / l_1.sum(axis=-1, keepdims=True)

    ## TURN 2
    turn_2 = model.turns_history[1]
    # Literal listener
    dm_a = turn_1.speaker.as_array[:,utterances_A.index(past_utterances[0])]
    print("out:", dm_a)
    l_0 = np.einsum('a,aby,ub->uay', dm_a, prior, lexicon_B)
    l_0 = l_0 / l_0.sum(axis=-1, keepdims=True)
    # Speaker at step 1
    cond_prior = prior.copy()
    cond_prior = cond_prior / cond_prior.sum(axis=0, keepdims=True).sum(axis=2, keepdims=True)
    mask = l_0 > 0
    ll_0 = np.zeros_like(l_0)
    ll_0[mask] = np.log(l_0[mask])
    ll_0[~mask] = -np.inf
    s_1 = np.einsum('aby,uay->abyu', cond_prior, ll_0)
    s_1[np.isnan(s_1)] = 0
    s_1 = np.einsum('abyu->bu', s_1)
    s_1 = softmax(alpha * s_1, axis=1)
    print(s_1)
    print(turn_2.speaker.history[0])


if __name__ == "__main__":
    main()
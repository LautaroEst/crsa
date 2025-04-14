
import numpy as np
from ..src.rsa import RSA


meanings  = ["David Lewis", "Paul Grice", "Claude Shannon"]
utterances = ["beard", "glasses", "tie"]

lexicon = np.array([
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 1]
])
prior = np.array([1/3, 1/3, 1/3])
cost = np.array([0, 0, 0])
alpha = 1


def main():
    rsa = RSA(meanings, utterances, lexicon, prior, cost, alpha, depth=1)
    rsa.run(verbose=False)

    assert np.allclose(rsa.speaker.value, np.array([
        [0.66666667, 0.33333333, 0.],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.]
    ]))

    assert np.allclose(rsa.listener.value, np.array([
        [1., 0., 0.],
        [.4, .6, 0.],
        [0., 0.33333, 0.66667]
    ]))

    print("Test passed!")




if __name__ == '__main__':
    main()
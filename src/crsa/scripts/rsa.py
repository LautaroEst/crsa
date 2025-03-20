
from pathlib import Path
import sys
import numpy as np
from typing import List
import shutil

from ..src.rsa import RSA
from ..src.utils import read_config_file


def main(
    meanings: List[str],
    utterances: List[str],
    lexicon: List[List[int]],
    prior: List[float],
    cost: List[float],
    alpha: float = 1.,
    depth: int = 2,
    output_dir: Path = Path("outputs"),
    verbose: bool = False
):
    # Run RSA
    rsa = RSA(meanings, utterances, lexicon, prior, cost, alpha, depth)
    rsa.run(output_dir, verbose)

    # Save history
    np.save(output_dir / "history.npy", rsa.history)


def setup():

    # Read configuration file
    if len(sys.argv) < 2:
        raise ValueError("Please provide a configuration file")
    config_file = sys.argv[1]
    config = read_config_file(f"{Path(__file__).stem}/{config_file}")

    # Check for verbose flag
    if len(sys.argv) > 2:
        if sys.argv[2] in ["--verbose", "-v"]:
            verbose = True
        else:
            raise ValueError(f"Unknown argument: {sys.argv[2]}")
    else:
        verbose = False

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / config_file
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copy(f"configs/{Path(__file__).stem}/{config_file}.yaml", output_dir / "config.yaml")

    return config, output_dir, verbose

if __name__ == '__main__':
    config, output_dir, verbose = setup()
    main(**config, output_dir=output_dir, verbose=verbose)    

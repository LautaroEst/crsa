
from pathlib import Path
import sys
import numpy as np
from typing import List, Union
import shutil
import logging
import time

from ..src.rsa import RSA
from ..src.utils import read_config_file


def main(
    meanings: List[str],
    utterances: List[str],
    lexicon: List[List[int]],
    prior: List[float],
    cost: List[float],
    alphas: Union[int,float,List[float]] = [],
    depths: Union[int,List[Union[List[int],int]]] = [[]],
    output_dir: Path = Path("outputs"),
    verbose: bool = False
):
    
    # Configure logging
    logger = logging.getLogger(__name__)
    console = logging.StreamHandler()
    logger.addHandler(console)
    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    file_handler = logging.FileHandler(output_dir / f"{now}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Validate alphas and depths
    if isinstance(alphas, float) or isinstance(alphas, int):
        alphas = [alphas]
    elif isinstance(alphas, list):
        if isinstance(depths, list) and len(alphas) != len(depths):
            raise ValueError("Alphas and depths must have the same length")
    else:
        raise ValueError("Alphas must be a number or a list of numbers")
    if isinstance(depths, int):
        depths = [depths]
        if len(alphas) != len(depths):
            raise ValueError("Alphas and depths must have the same length")
    elif not isinstance(depths, list):
        raise ValueError("Depth must be an integer or a list of integers")

    # Run RSA for each alpha and depth
    for alpha, alphas_depths in zip(alphas, depths):

        # Check if depth is a list
        if isinstance(alphas_depths, int):
            alphas_depths = [alphas_depths]
        elif not isinstance(alphas_depths, list):
            raise ValueError("Depth must be an integer or a list of integers")
        
        for depth in alphas_depths:

            # Create output directory
            suboutput_dir = output_dir / f"alpha={float(alpha)}" / f"depth={depth}"
            if suboutput_dir.exists():
                logger.warning(f"Experiment already run for alpha={alpha} and depth={depth}. Skipping.")
                continue
            else:
                logger.info(f"Running experiment for alpha={alpha} and depth={depth}")
            suboutput_dir.mkdir(parents=True, exist_ok=True)

            # Run RSA
            rsa = RSA(meanings, utterances, lexicon, prior, cost, alpha, depth)
            rsa.run(suboutput_dir, verbose)

    # Close logging
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


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

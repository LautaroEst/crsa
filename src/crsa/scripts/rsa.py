
from pathlib import Path
import sys
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
    alphas: List[float] = [1.0],
    max_depths: List[List[Union[int,None]]] = [[None]],
    tolerances: List[List[Union[float,None]]] = [[None]],
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

    # Run RSA for each alpha and depth
    for alpha, alphas_max_depths, alphas_tolerances in zip(alphas, max_depths, tolerances):
        alpha = float(alpha)

        for max_depth, tolerance in zip(alphas_max_depths, alphas_tolerances):

            max_depth = None if max_depth is None else int(max_depth)
            tolerance = None if tolerance is None else float(tolerance)

            # Create output directory
            suboutput_dir = output_dir / f"alpha={alpha}" / f"max_depth={max_depth}_tolerance={tolerance}"
            if suboutput_dir.exists():
                logger.warning(f"Experiment already run for alpha={alpha}, max_depth={max_depth} and tolerance={tolerance}. Skipping.")
                continue
            else:
                logger.info(f"Running experiment for alpha={alpha}, max_depth={max_depth} and tolerance={tolerance}.")
            suboutput_dir.mkdir(parents=True, exist_ok=True)

            # Run RSA
            rsa = RSA(meanings, utterances, lexicon, prior, cost, alpha, max_depth, tolerance)
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

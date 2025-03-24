
import argparse
from pathlib import Path
import sys
from typing import List, Optional
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
    max_depths: Optional[List[int]] = None,
    tolerances: Optional[List[float]] = None,
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

    # Check if RSA is possible
    if max_depths is None and tolerances is None:
        logger.error("Either max_depths or tolerances must be provided.")
        sys.exit(1)
    if max_depths is None and tolerances is not None:
        max_depths = [float("inf")] * len(tolerances)
    if max_depths is not None and tolerances is None:
        tolerances = [0.] * len(max_depths)
    if len(alphas) != len(max_depths) or len(alphas) != len(tolerances):
        logger.error("Alphas, max_depths and tolerances must have the same length.")
        sys.exit(1)

    # Run RSA for each alpha and depth
    for alpha, max_depth, tolerance in zip(alphas, max_depths, tolerances):

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
        rsa.save(suboutput_dir)

    # Close logging
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def setup():

    def int_or_inf(value):
        if value.lower() == "inf":
            return float("inf")
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a integer or 'inf'.")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run RSA for a given configuration file")
    parser.add_argument("config_file", type=str, help="Configuration file")
    parser.add_argument("--alphas", type=float, nargs="+", help="Alphas to run RSA with", default=[1.0])
    parser.add_argument("--max_depths", type=int_or_inf, nargs="+", help="Max depths to run RSA with", default=None)
    parser.add_argument("--tolerances", type=float, nargs="+", help="Tolerances to run RSA with", default=None)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output", default=False)
    args = parser.parse_args()

    # Read configuration file
    config = read_config_file(f"{Path(__file__).stem}/{args.config_file}")

    # Create output directory
    output_dir = Path("outputs") / Path(__file__).stem / args.config_file
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update configuration
    config["alphas"] = args.alphas
    config["max_depths"] = args.max_depths
    config["tolerances"] = args.tolerances
    config["output_dir"] = output_dir
    config["verbose"] = args.verbose

    # Save configuration file
    shutil.copy(f"configs/{Path(__file__).stem}/{args.config_file}.yaml", output_dir / "config.yaml")

    return config

if __name__ == '__main__':
    config = setup()
    main(**config)

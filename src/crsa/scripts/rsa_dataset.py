
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..src.rsa import RSA



def main(
    dataset,
    output_dir = Path("outputs"), 
    alpha = 1.0, 
    max_depth = float("inf"), 
    tolerance = 1e-7,
):

    # Check if RSA is possible
    if max_depth == float("inf") and tolerance == 0:
        raise ValueError("Either max_depth or tolerance must be provided.")
    
    

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
    parser.add_argument("--dataset", type=str, help="Dataset in which to run RSA")
    parser.add_argument("--alpha", type=float, help="Alpha parameter of RSA model", default=1.0)
    parser.add_argument("--max_depth", type=int_or_inf, help="Maximum depth of RSA model", default=float("inf"))
    parser.add_argument("--tolerance", type=float, help="Tolerance of RSA model", default=1e-7)
    args = parser.parse_args()

    # Create configuration
    config = {
        "dataset": args.dataset,
        "output_dir": Path("outputs") / Path(__file__).stem / args.dataset,
        "alpha": args.alpha,
        "max_depth": args.max_depth,
        "tolerance": args.tolerance
    }

    return config

if __name__ == '__main__':
    config = setup()
    main(**config)
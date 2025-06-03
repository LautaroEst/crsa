

import logging
from pathlib import Path
import time
import yaml


def init_logger(name, output_dir: Path):
    script_logger = logging.getLogger(name)
    console = logging.StreamHandler()
    script_logger.addHandler(console)
    now = time.strftime("%Y-%m-%d-%H-%M-%S")
    file_handler = logging.FileHandler(output_dir / f"{now}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    script_logger.addHandler(file_handler)
    script_logger.setLevel(logging.INFO)
    return script_logger


def read_yaml(path: Path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config




def check_iter_args(max_depth, tolerance):
    if isinstance(max_depth, (int, str)):
        max_depth = float(max_depth)
    if isinstance(tolerance, (int, str)):
        tolerance = float(tolerance)

    if max_depth is None and tolerance is None:
        raise ValueError("Either max_depth or tolerance must be provided.")
    elif max_depth is None and isinstance(tolerance, (int, float)):
        max_depth = float("inf")
    elif isinstance(max_depth, (int, str)) and tolerance is None:
        tolerance = 0.
    elif isinstance(max_depth, float) and isinstance(tolerance, float):
        if max_depth <= 0:
            raise ValueError("max_depth must be a positive integer or 'inf'.")
        if tolerance < 0:
            raise ValueError("tolerance must be a non-negative number.")
    else:
        raise ValueError("Invalid combination of max_depth and tolerance.")
    return max_depth, tolerance
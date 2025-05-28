

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
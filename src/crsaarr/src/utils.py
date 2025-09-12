

import logging
from pathlib import Path
import time
import numpy as np
import torch



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



class Predictions:

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, prediction: dict):
        if prediction["idx"] in self.predictions_ids:
            raise ValueError(f"Prediction with idx {prediction['idx']} already exists.")
        output_file = self.output_dir / f"sample_{prediction['idx']}.npy"
        np.save(output_file, prediction, allow_pickle=True)

    @property
    def predictions_ids(self):
        return [int(f.stem.split("_")[1]) for f in self.output_dir.glob("sample_*.npy")]

    def __contains__(self, prediction):
        return prediction["idx"] in self.predictions_ids
    
    def __len__(self):
        return len(self.predictions_ids)
    
    def __iter__(self):
        for prediction_id in self.predictions_ids:
            output_file = self.output_dir / f"sample_{prediction_id}.npy"
            yield np.load(output_file, allow_pickle=True).item()
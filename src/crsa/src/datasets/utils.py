


from pathlib import Path
import torch


class Predictions:

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, prediction: dict):
        if prediction["idx"] in self.predictions_ids:
            raise ValueError(f"Prediction with idx {prediction['idx']} already exists.")
        output_file = self.output_dir / f"sample_{prediction['idx']}.pt"
        torch.save(prediction, output_file)

    @property
    def predictions_ids(self):
        return [int(f.stem.split("_")[1]) for f in self.output_dir.glob("sample_*.pt")]

    def __contains__(self, prediction):
        return prediction["idx"] in self.predictions_ids
    
    def __len__(self):
        return len(self.predictions_ids)
    
    def __iter__(self):
        for prediction_id in self.predictions_ids:
            output_file = self.output_dir / f"sample_{prediction_id}.pt"
            yield torch.load(output_file, weights_only=False)

    
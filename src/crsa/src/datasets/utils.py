


from pathlib import Path
import pickle


class Predictions:

    def __init__(self, predictions_ids, output_dir: Path):
        self.predictions_ids = predictions_ids
        self.output_dir = output_dir

    def add(self, prediction: dict):
        if prediction["idx"] in self.predictions_ids:
            raise ValueError(f"Prediction with idx {prediction['idx']} already exists.")
        self.predictions_ids.append(prediction["idx"])
        output_file = self.output_dir / f"sample_{prediction['idx']}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(prediction, f)

    @classmethod
    def from_directory(cls, directory: Path):
        processed_predictions = directory.glob("sample_*.pkl")
        if processed_predictions:
            predictions_ids = [int(f.stem.split("_")[1]) for f in processed_predictions]
        else:
            predictions_ids = []
        return cls(predictions_ids, directory)
    
    @property
    def ids(self):
        return self.predictions_ids
    
    def __contains__(self, prediction):
        return prediction["idx"] in self.predictions_ids
    
    def __len__(self):
        return len(self.predictions_ids)
    
    def __iter__(self):
        for prediction_id in self.predictions_ids:
            output_file = self.output_dir / f"sample_{prediction_id}.pkl"
            with open(output_file, "rb") as f:
                yield pickle.load(f)

    
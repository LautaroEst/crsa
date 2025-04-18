

from pathlib import Path


class BaseDialogModel:

    def save(self, output_dir: Path, prefix: str = ""):
        """
        Save the model to a file.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @classmethod
    def load(cls, output_dir: Path, prefix: str = ""):
        """
        Load the model from a file.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def update(self, *args, **kwargs):
        """
        Update the model with the speaker's information.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
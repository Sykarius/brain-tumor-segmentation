import os
import torch
import numpy as np

from .config import CheckpointConfig

class EarlyStoppingCheckpointer:
    """
    Tracks validation metrics, saves the best model state, and triggers early stopping 
    if the network stops improving over a set number of epochs.
    """
    def __init__(
        self, 
        config: CheckpointConfig
    ):
        self.save_dir = config.save_dir
        self.filename = config.filename
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.mode = config.mode
        
        # State tracking
        self.counter = 0
        self.best_metric = -np.inf if self.mode == "max" else np.inf
        self.early_stop = False
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, self.filename)

    def __call__(self, current_metric: float, model: torch.nn.Module):
        """
        Evaluates the current epoch's metric against the historical best.
        Expects the raw student model to be passed in for saving.
        """
        if self.mode == "max":
            # For metrics like Dice Score (higher is better)
            improvement = (current_metric - self.best_metric) > self.min_delta
        else:
            # For metrics like Validation Loss (lower is better)
            improvement = (self.best_metric - current_metric) > self.min_delta

        if improvement:
            print(f"Validation metric improved from {self.best_metric:.4f} to {current_metric:.4f}. Saving checkpoint...")
            self.best_metric = current_metric
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.save_path)
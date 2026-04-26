import os
import torch
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """
    Handles logging of scalars, gradient histograms, and image slices to TensorBoard.
    """
    def __init__(self, log_dir: str = "./logs/brats_distillation_experiment"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_losses(self, loss_dict: dict, step: int, phase: str = "Train"):
        """
        Logs individual loss components from the composite loss dictionary.
        """
        for loss_name, loss_value in loss_dict.items():
            self.writer.add_scalar(f"{phase}/{loss_name}", loss_value, step)

    def log_metrics(self, metric_name: str, metric_value: float, step: int, phase: str = "Val"):
        """
        Logs evaluation metrics like Dice Score.
        """
        self.writer.add_scalar(f"{phase}/{metric_name}", metric_value, step)

    def log_gradients(self, model: torch.nn.Module, step: int):
        """
        Iterates through the model and logs the gradient histograms.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Log the raw gradient distribution
                self.writer.add_histogram(f"Gradients/{name}", param.grad, step)
                # Log the gradient norm as a scalar for easy plotting
                grad_norm = param.grad.norm().item()
                self.writer.add_scalar(f"Gradient_Norms/{name}", grad_norm, step)

    def log_image_slices(self, inputs, labels, predictions, step: int, phase: str = "Val", num_samples: int = 1):
        """
        Takes 3D BraTS tensors (B, C, D, H, W) and extracts the middle axial slice 
        to visualize the network's structural understanding.
        """
        batch_size = inputs.shape[0]
        samples_to_log = min(batch_size, num_samples)

        for i in range(samples_to_log):
            # 1. Extract the specific patient from the batch
            img_vol = inputs[i]  # Shape: (C, D, H, W)
            lbl_vol = labels[i]  # Shape: (1, D, H, W)
            prd_vol = predictions[i] # Shape: (1, D, H, W)

            D = img_vol.shape[1]
            mid_d = D // 2

            # Shape becomes (C, H, W) or (1, H, W)
            img_slice = img_vol[:, mid_d, :, :]
            lbl_slice = lbl_vol[:, mid_d, :, :].float() 
            prd_slice = prd_vol[:, mid_d, :, :].float()

            bg_slice = img_slice[0:1, :, :] # Keep as (1, H, W)

            bg_slice = (bg_slice - bg_slice.min()) / (bg_slice.max() - bg_slice.min() + 1e-8)

            max_lbl = max(lbl_slice.max(), 1.0)
            lbl_slice = lbl_slice / max_lbl
            prd_slice = prd_slice / max_lbl

            display_grid = torch.cat([bg_slice, lbl_slice, prd_slice], dim=2) # Concat along width

            self.writer.add_image(
                f"{phase}/Patient_{i}_Middle_Slice", 
                display_grid, 
                step, 
                dataformats="CHW"
            )

    def close(self):
        self.writer.close()
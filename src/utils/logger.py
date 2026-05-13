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
        Takes 3D BraTS tensors (B, C, H, W, D) and extracts the middle axial slice 
        to visualize the network's structural understanding for TC, WT, and ET independently.
        """
        batch_size = inputs.shape[0]
        samples_to_log = min(batch_size, num_samples)
        
        # BraTS overlapping classes mapping
        region_map = {0: "TC", 1: "WT", 2: "ET"}

        for i in range(samples_to_log):
            # 1. Extract the specific patient from the batch (C, H, W, D)
            img_vol = inputs[i]  
            lbl_vol = labels[i]  
            prd_vol = predictions[i] 

            # 2. Correctly target the Depth dimension (the last axis)
            D = img_vol.shape[-1]
            mid_d = D // 2

            # Safely grab the middle axial slice: shape becomes (C, H, W)
            img_slice = img_vol[..., mid_d]
            lbl_slice = lbl_vol[..., mid_d]
            prd_slice = prd_vol[..., mid_d]

            # Grab the first background modality (e.g., FLAIR) and normalize it for visualization
            bg_slice = img_slice[0:1, :, :] 
            bg_slice = (bg_slice - bg_slice.min()) / (bg_slice.max() - bg_slice.min() + 1e-8)

            # 3. Loop through TC, WT, ET and log them as separate images
            for region_idx, region_name in region_map.items():
                if lbl_slice.shape[0] > region_idx:
                    # Extract the specific channel mask (1, H, W)
                    curr_lbl = lbl_slice[region_idx:region_idx + 1, :, :].float()
                    curr_prd = prd_slice[region_idx:region_idx + 1, :, :].float()

                    # Normalize the masks to ensure they render brightly (0 to 1)
                    curr_lbl = curr_lbl / max(curr_lbl.max().item(), 1.0)
                    curr_prd = curr_prd / max(curr_prd.max().item(), 1.0)

                    # Concatenate side-by-side: [Background MRI | Ground Truth Mask | Predicted Mask]
                    display_grid = torch.cat([bg_slice, curr_lbl, curr_prd], dim=2) 

                    # Log to TensorBoard
                    self.writer.add_image(
                        f"{phase}/Patient_{i}_{region_name}", 
                        display_grid, 
                        step, 
                        dataformats="CHW"
                    )

    def close(self):
        self.writer.close()

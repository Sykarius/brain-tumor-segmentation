import torch
import torch.optim as optim
from tqdm import tqdm
from monai.metrics import DiceMetric

from data.dataset import get_dataloaders
from models.wrapper import ContrastiveDistillationWrapper
from losses.composite import CompositeDistillationLoss
from utils.config import TrainingConfig
from utils.checkpoint import EarlyStoppingCheckpointer
from utils.logger import TensorBoardLogger

def main(config: TrainingConfig):
    device = config.device
    print(f"Accelerating training on: {device}")

    epochs = config.epochs
    
    print("Loading DataLoaders...")
    train_loader, val_loader = get_dataloaders(config.data)

    print("Initializing Dual-Network Wrapper...")
    wrapper = ContrastiveDistillationWrapper(bundle_dir=config.bundle_dir).to(device)

    print("Initializing Composite Loss...")
    criterion = CompositeDistillationLoss(config.loss).to(device)

    # Only student parameters.
    optimizer = optim.AdamW(wrapper.student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # MONAI's metric for evaluation
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Checkpointer
    checkpointer = EarlyStoppingCheckpointer(config.checkpoint)

    logger = TensorBoardLogger(config.logging_dir)
    global_step = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        wrapper.train()
        epoch_loss = 0.0
        step_metrics = {"seg": 0.0, "kl": 0.0, "crd": 0.0}

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move data to MPS
            full_inputs = batch["image_full"].to(device)
            missing_inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            s_logits, t_logits, s_embeddings, t_embeddings = wrapper(full_inputs, missing_inputs)

            loss, loss_dict = criterion(
                s_logits, t_logits, s_embeddings, t_embeddings, labels
            )

            loss.backward()
            optimizer.step()

            # Tracking
            epoch_loss += loss.item()
            step_metrics["seg"] += loss_dict["loss_seg"]
            step_metrics["kl"] += loss_dict["loss_kl"]
            step_metrics["crd"] += loss_dict["loss_crd"]

            if global_step % 10 == 0:
                logger.log_losses(loss_dict, global_step, phase="Train")
                logger.log_gradients(wrapper.student.model, global_step)
            
            global_step += 1
            
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Calculate epoch averages
        num_batches = len(train_loader)
        print(f"Train Loss: {epoch_loss/num_batches:.4f} | "
              f"Seg: {step_metrics['seg']/num_batches:.4f} | "
              f"KL: {step_metrics['kl']/num_batches:.4f} | "
              f"CRD: {step_metrics['crd']/num_batches:.4f}")

        # --- VALIDATION PHASE ---
        # We evaluate deterministically on the dropped modality (e.g., T1ce missing)
        wrapper.eval()
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="Validating")
            for batch in val_progress:
                missing_inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

                # We bypass the wrapper logic and evaluate the Student directly
                val_logits, _ = wrapper.student(missing_inputs)
                
                # Convert logits to discrete predictions (Argmax over channels)
                val_preds = torch.argmax(val_logits, dim=1, keepdim=True)
                
                # Compute Dice score for this volume
                dice_metric(y_pred=val_preds, y=labels)

            # Aggregate Dice score across the entire validation set
            current_val_dice = dice_metric.aggregate().item()
            logger.log_metrics("Dice_Score", current_val_dice, epoch, phase="Val")
            dice_metric.reset()

            # Visual Check
            logger.log_image_slices(
                inputs=missing_inputs, 
                labels=labels, 
                predictions=val_preds, 
                step=epoch, 
                phase="Val"
            )

            print(f"Validation Dice Score (Missing T1ce): {current_val_dice:.4f}")

            # Early Stop
            checkpointer(current_val_dice, wrapper.student.model)
            if checkpointer.early_stop:
                print(f"Eary stopping triggered. Halting training.")
                break
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
from utils.config import load_config


class Trainer:

    def __init__(self, config: TrainingConfig):
        self.device = config.device
        device = config.device
        print(f"Accelerating training on: {device}")

        self.epochs = config.epochs

        print("Loading DataLoaders...")
        self.train_loader, self.val_loader = get_dataloaders(config.data)

        print("Initializing Dual-Network Wrapper...")
        self.wrapper = ContrastiveDistillationWrapper(bundle_dir=config.bundle_dir).to(device)

        print("Initializing Composite Loss...")
        self.criterion = CompositeDistillationLoss(config.loss).to(device)

        # Only student parameters.
        self.optimizer = optim.AdamW(self.wrapper.student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
        # MONAI's metric for evaluation
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")

        # Checkpointer
        self.checkpointer = EarlyStoppingCheckpointer(config.checkpoint)

        self.logger = TensorBoardLogger(config.logging_dir)

    def train(self):
        global_step = 0

        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch+1}/{self.epochs} ---")
            
            self.wrapper.student.train()
            self.wrapper.teacher.eval()
            epoch_loss = 0.0
            step_metrics = {"seg": 0.0, "kd": 0.0, "crd": 0.0}

            progress_bar = tqdm(self.train_loader, desc="Training")
            for batch in progress_bar:

                full_inputs = batch["image_full"].to(self.device)
                missing_inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()

                s_logits, t_logits, s_embeddings, t_embeddings = self.wrapper(full_inputs, missing_inputs)

                loss, loss_dict = self.criterion(
                    s_logits, t_logits, s_embeddings, t_embeddings, labels
                )

                loss.backward()
                self.optimizer.step()

                # Tracking
                epoch_loss += loss.item()
                step_metrics["seg"] += loss_dict["loss_seg"]
                step_metrics["kd"] += loss_dict["loss_kd"]
                step_metrics["crd"] += loss_dict["loss_crd"]

                if global_step % 10 == 0:
                    self.logger.log_losses(loss_dict, global_step, phase="Train")
                    self.logger.log_gradients(self.wrapper.student, global_step)
                
                global_step += 1
                
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Calculate epoch averages
            num_batches = len(self.train_loader)
            print(f"Train Loss: {epoch_loss/num_batches:.4f} | "
                f"Seg: {step_metrics['seg']/num_batches:.4f} | "
                f"KD: {step_metrics['kd']/num_batches:.4f} | "
                f"CRD: {step_metrics['crd']/num_batches:.4f}")

            # Validate
            current_val_dice = self.validate(epoch)
            print(f"Validation Dice Score (Missing T1ce): {current_val_dice:.4f}")
    
            # Early Stop
            self.checkpointer(current_val_dice, self.wrapper.student.model)
            if self.checkpointer.early_stop:
                print(f"Early stopping triggered. Halting training.")
                break

    def validate(self, epoch: int):
        self.wrapper.eval()
        with torch.no_grad():
            val_progress = tqdm(self.val_loader, desc="Validating")
            for batch in val_progress:
                missing_inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # We bypass the wrapper logic and evaluate the Student directly
                val_logits = self.wrapper.student(missing_inputs)[0]
                
                # Convert multilabel logits to binary TC/WT/ET predictions
                val_probs = torch.sigmoid(val_logits)
                thresholds = torch.tensor([0.3, 0.3, 0.1], device=self.device).reshape(1, 3, 1, 1, 1)
                val_preds = (val_probs > thresholds).float()
                
                # Compute Dice score for this volume
                self.dice_metric(y_pred=val_preds, y=labels)

            # Aggregate Dice score across the entire validation set
            metric_batch = self.dice_metric.aggregate()
            
            dice_tc = metric_batch[0].item()
            dice_wt = metric_batch[1].item()
            dice_et = metric_batch[2].item()
            mean_dice = metric_batch.mean().item()

            self.logger.log_metrics("Val_Dice/TC", dice_tc, epoch, phase="Val")
            self.logger.log_metrics("Val_Dice/WT", dice_wt, epoch, phase="Val")
            self.logger.log_metrics("Val_Dice/ET", dice_et, epoch, phase="Val")
            self.logger.log_metrics("Val_Dice/Mean", mean_dice, epoch, phase="Val")
            self.dice_metric.reset()

            # Visual Check
            self.logger.log_image_slices(
                inputs=missing_inputs, 
                labels=labels, 
                predictions=val_preds, 
                step=epoch, 
                phase="Val"
            )

            return mean_dice


if __name__ == "__main__":
    config = load_config("./config/test.yaml")
    trainer = Trainer(config)
    trainer.train()

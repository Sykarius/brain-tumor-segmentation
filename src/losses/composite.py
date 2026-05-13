import torch
import torch.nn as nn
from monai.losses import DiceLoss

from .infonce import IntraPatientInfoNCE
from utils.config import LossConfig


class CompositeDistillationLoss(nn.Module):

    def __init__(
        self,
        config: LossConfig
    ):
        super().__init__()
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.kd_temperature = config.kl_temperature
        
        weights = config.class_weights
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

        # Multi-label Dice loss for TC, WT, ET channels
        self.seg_loss_fn = DiceLoss(
            include_background=True,
            sigmoid=True,
            squared_pred=True,
            reduction="none",
            batch=True,
        )

        # 2. Output-level distillation with sigmoid teacher probabilities
        self.kd_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        # 3. Intra Patient CRD
        self.crd_loss_fn = IntraPatientInfoNCE(
            temperature=config.temperature, aggregation=config.aggregation, max_samples=config.max_samples
        )

    def forward(self, s_logits, t_logits, s_embeddings, t_embeddings, labels):

        labels = labels.float()

        # Dice with weights
        loss_seg_raw = self.seg_loss_fn(s_logits, labels)
        loss_seg = (loss_seg_raw * self.class_weights).mean()

        # Output-level distillation with temperature-scaled sigmoid probabilities
        T = self.kd_temperature

        with torch.no_grad():
            teacher_probs = torch.sigmoid(t_logits / T)

        loss_kd_raw = self.kd_loss_fn(s_logits / T, teacher_probs)
        w_shape = (1, self.class_weights.size(0), 1, 1, 1)
        loss_kd_weighted = loss_kd_raw * self.class_weights.reshape(w_shape)
        loss_kd = loss_kd_weighted.mean() * (T ** 2)

        # CRD
        loss_crd = self.crd_loss_fn(s_embeddings, t_embeddings, labels)

        # Total Loss
        loss_total = (self.alpha * loss_seg) + (self.beta * loss_kd) + (self.gamma * loss_crd)

        # Returning loss dict for logging
        loss_dict = {
            "loss_total": loss_total.item(),
            "loss_seg": loss_seg.item(),
            "loss_kd": loss_kd.item(),
            "loss_crd": loss_crd.item(),
        }

        return loss_total, loss_dict

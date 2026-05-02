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

        # Multi-label Dice loss for TC, WT, ET channels
        self.seg_loss_fn = DiceLoss(
            include_background=True,
            sigmoid=True,
            squared_pred=True,
            reduction="mean",
            batch=True,
        )

        # 2. Output-level distillation with sigmoid teacher probabilities
        self.kd_loss_fn = nn.BCEWithLogitsLoss()

        # 3. Intra Patient CRD
        self.crd_loss_fn = IntraPatientInfoNCE(
            temperature=config.temperature, aggregation=config.aggregation, max_samples=config.max_samples
        )

    def forward(self, s_logits, t_logits, s_embeddings, t_embeddings, labels):

        labels = labels.float()

        # Dice
        loss_seg = self.seg_loss_fn(s_logits, labels)

        # Output-level distillation
        with torch.no_grad():
            teacher_probs = torch.sigmoid(t_logits)

        loss_kd = self.kd_loss_fn(s_logits, teacher_probs)

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

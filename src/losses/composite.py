import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss

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

        # Dice Loss
        self.seg_loss_fn = DiceCELoss(
            include_background=False,
            softmax=True,
            to_onehot_y=False,
            squared_pred=True,
            batch=True,
        )

        # 2. KL Loss
        self.kl_temperature = config.kl_temperature
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

        # 3. Intra Patient CRD
        self.crd_loss_fn = IntraPatientInfoNCE(
            temperature=config.temperature, granularity=config.granularity
        )

    def forward(self, s_logits, t_logits, s_embeddings, t_embeddings, labels):

        # Dice
        loss_seg = self.seg_loss_fn(s_logits, labels)

        # KL Loss
        s_log_probs = F.log_softmax(s_logits / self.kl_temperature, dim=1)
        t_probs = F.softmax(t_logits / self.kl_temperature, dim=1)
        loss_kl = self.kl_loss_fn(s_log_probs, t_probs) * (self.kl_temperature**2)

        # CRD
        loss_crd = self.crd_loss_fn(s_embeddings, t_embeddings, labels)

        # Total Loss
        loss_total = (self.alpha * loss_seg) + (self.beta * loss_kl) + (self.gamma * loss_crd)

        # Returning loss dict for logging
        loss_dict = {
            "loss_total": loss_total.item(),
            "loss_seg": loss_seg.item(),
            "loss_kl": loss_kl.item(),
            "loss_crd": loss_crd.item(),
        }

        return loss_total, loss_dict

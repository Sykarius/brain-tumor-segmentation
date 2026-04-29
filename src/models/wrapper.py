from torch import nn
import torch

from .model import SegResNet

class ContrastiveDistillationWrapper(nn.Module):
    def __init__(self, bundle_dir: str = "./models/brats_mri_segmentation"):
        super().__init__()
        
        self.teacher = SegResNet(bundle_dir=bundle_dir, is_teacher=True)
        self.student = SegResNet(bundle_dir=bundle_dir, is_teacher=False)

    def forward(self, full_inputs, missing_inputs):
        with torch.no_grad():
            t_logits, t_embeddings = self.teacher(full_inputs)
            
        s_logits, s_embeddings = self.student(missing_inputs)
        
        return s_logits, t_logits, s_embeddings, t_embeddings
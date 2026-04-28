import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAggregator(nn.Module):
    def __init__(self, granularity: str = "dense"):
        super().__init__()
        valid_modes = ["global", "region", "dense"]
        if granularity not in valid_modes:
            raise ValueError(f"Granularity must be one of {valid_modes}")
        self.granularity = granularity

    def forward(self, s_embeddings, t_embeddings, labels=None):
        if self.granularity == "global":
            s_pooled = F.adaptive_avg_pool3d(s_embeddings, 1).squeeze(-1).squeeze(-1).squeeze(-1)
            t_pooled = F.adaptive_avg_pool3d(t_embeddings, 1).squeeze(-1).squeeze(-1).squeeze(-1)
            return s_pooled, t_pooled

        elif self.granularity == "region":
            if labels is None:
                raise ValueError("Region-level pooling requires ground-truth labels.")
            
            _, _, D, H, W = s_embeddings.shape
            resized_labels = F.interpolate(labels.float(), size=(D, H, W), mode="nearest")

            s_regions, t_regions = [], []
            
            for class_idx in [0, 1, 2, 4]:
                class_mask = (resized_labels == class_idx).float()
                
                mask_sum = class_mask.sum(dim=(2, 3, 4)) + 1e-8
                
                s_reg = (s_embeddings * class_mask).sum(dim=(2, 3, 4)) / mask_sum
                t_reg = (t_embeddings * class_mask).sum(dim=(2, 3, 4)) / mask_sum
                
                s_regions.append(s_reg)
                t_regions.append(t_reg)
                
            return torch.stack(s_regions, dim=1), torch.stack(t_regions, dim=1)

        elif self.granularity == "dense":
            B, C, D, H, W = s_embeddings.shape
            s_flat = s_embeddings.view(B, C, -1).transpose(1, 2)
            t_flat = t_embeddings.view(B, C, -1).transpose(1, 2)
            return s_flat, t_flat
        

class IntraPatientInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.1, granularity: str = "dense"):
        super().__init__()
        self.temperature = temperature
        self.aggregator = FeatureAggregator(granularity=granularity)

    def forward(self, s_embeddings, t_embeddings, labels=None):
        s_agg, t_agg = self.aggregator(s_embeddings, t_embeddings, labels)
        
        s_norm = F.normalize(s_agg, p=2, dim=-1)
        t_norm = F.normalize(t_agg, p=2, dim=-1)

        total_loss = 0.0
        B = s_norm.size(0)

        # Iterate through the batch. Intra-Patient means we only compare within the same sample
        for i in range(B):
            s_i = s_norm[i] # (1 x Latent_Dim)
            t_i = t_norm[i]


            similarity_matrix = torch.matmul(s_i, t_i.T) / self.temperature
            # (Latent_Dim, Latent_Dim)

            # In InfoNCE, the "Positives" are the diagonal of this matrix 
            # (e.g., Student Voxel A vs Teacher Voxel A)
            # The "Negatives" are all the off-diagonal elements 
            # (e.g., Student Voxel A vs Teacher Voxel B)
            num_items = s_i.size(0)
            labels = torch.arange(num_items, device=s_i.device)

            loss = F.cross_entropy(similarity_matrix, labels)
            total_loss += loss

        return total_loss / B

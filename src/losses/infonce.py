import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAggregator(nn.Module):
    def __init__(self, granularity: str = "dense", max_samples: float = 4096):
        super().__init__()
        valid_modes = ["global", "region", "dense"]
        if granularity not in valid_modes:
            raise ValueError(f"Granularity must be one of {valid_modes}")
        self.granularity = granularity
        self.max_samples = max_samples

    def forward(self, s_embeddings, t_embeddings, labels=None):
        if self.granularity == "global":
            raise NotImplementedError(f"This does not work with Intra Sample CRD")

        elif self.granularity == "region":
            raise NotImplementedError(f"Need to implemented in the future.")

        elif self.granularity == "dense":
            B, C, D, H, W = s_embeddings.shape
            s_flat = s_embeddings.view(B, C, -1).transpose(1, 2)
            t_flat = t_embeddings.view(B, C, -1).transpose(1, 2)
            
            num_voxels = s_flat.size(1)
            
            # --- THE MEMORY SHIELD ---
            if num_voxels > self.max_samples:
                rand_indices = torch.randperm(num_voxels, device=s_embeddings.device)[:self.max_samples]
                s_flat = s_flat[:, rand_indices, :]
                t_flat = t_flat[:, rand_indices, :]
        
            return s_flat, t_flat
        

class IntraPatientInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.1, granularity: str = "dense", max_samples: float = 4096):
        super().__init__()
        self.temperature = temperature
        self.aggregator = FeatureAggregator(granularity=granularity, max_samples=max_samples)

    def forward(self, s_embeddings_list: list, t_embeddings_list: list, labels=None):

        total_loss = 0.0
        for s_emb, t_emb in zip(s_embeddings_list, t_embeddings_list):
            s_agg, t_agg = self.aggregator(s_emb, t_emb, labels)
            
            s_norm = F.normalize(s_agg, p=2, dim=-1)
            t_norm = F.normalize(t_agg, p=2, dim=-1)

            scale_loss = 0.0
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
                scale_loss += loss
            
            total_loss += scale_loss / B

        return total_loss

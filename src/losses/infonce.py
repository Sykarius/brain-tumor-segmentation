import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAggregator(nn.Module):
    def __init__(self, aggregation: str = "avg_pool", max_samples: float = 4096):
        super().__init__()
        valid_modes = ["random", "avg_pool", "class_balanced"]
        if aggregation not in valid_modes:
            raise ValueError(f"Aggregation mode must be one of {valid_modes}")
        self.aggregation = aggregation
        self.max_samples = int(max_samples)
        self.num_classes = 4
    
    def _get_target_shape(self, D: int, H: int, W: int):
        total_voxels = D * H * W
        if total_voxels <= self.max_samples:
            return (D, H, W)
        scale = (self.max_samples / total_voxels) ** (1 / 3.0)

        D_out = max(1, int(D * scale))
        H_out = max(1, int(H * scale))
        W_out = max(1, int(W * scale))
        
        return (D_out, H_out, W_out)



    def forward(self, s_embeddings, t_embeddings, labels=None):
        B, C, D, H, W = s_embeddings.shape
        if self.aggregation == "avg_pool":
            target_shape = self._get_target_shape(D, H, W)

            s_embeddings = F.adaptive_avg_pool3d(s_embeddings, target_shape)
            t_embeddings = F.adaptive_avg_pool3d(t_embeddings, target_shape)

            s_flat = s_embeddings.reshape(B, C, -1).transpose(1, 2)
            t_flat = t_embeddings.reshape(B, C, -1).transpose(1, 2)

            return s_flat, t_flat
        elif self.aggregation == "class_balanced":
            if labels is None:
                raise ValueError(f"Labels cannot be none for {self.aggregation}")
            
            s_flat = s_embeddings.reshape(B, C, -1).transpose(1, 2)
            t_flat = t_embeddings.reshape(B, C, -1).transpose(1, 2)

            resized_labels = F.interpolate(labels.float(), size=(D, H, W), mode="nearest")
            labels_flat = resized_labels.reshape(B, 3, -1)

            s_sampled_batch = []
            t_sampled_batch = []

            for i in range(B):
                s_i = s_flat[i]
                t_i = t_flat[i]
                l_i = labels_flat[i]

                tc, wt, et = l_i[0], l_i[1], l_i[2]
                masks = [
                    (wt == 0), # 0: Background
                    (wt == 1) & (tc == 0), # 1: Edema (WT)
                    (tc == 1) & (et == 0), # 2: Necrotic Core TC
                    (et == 1) # 3: Enhancind Tumor (ET)
                ]

                samples_per_class = self.max_samples // self.num_classes
                sampled_indices = []

                for mask in masks:
                    cls_indices = torch.nonzero(mask, as_tuple=True)[0]
                    num_cls_voxels = cls_indices.size(0)

                    if num_cls_voxels == 0:
                        continue

                    if num_cls_voxels > samples_per_class:
                        rand_perm = torch.randperm(num_cls_voxels, device=s_i.device)[:samples_per_class]
                        sampled_indices.append(cls_indices[rand_perm])
                    else:
                        sampled_indices.append(cls_indices)
                    
                selected_indeces = torch.cat(sampled_indices)

                if selected_indeces.size(0) < self.max_samples:
                    shortfall = self.max_samples - selected_indeces.size(0)
                    pad_indices = torch.randperm(l_i.size(1), device=s_i.device)[:shortfall]
                    selected_indeces = torch.cat([selected_indeces, pad_indices])
                
                final_perm = torch.randperm(selected_indeces.size(0), device=s_i.device)
                selected_indeces = selected_indeces[final_perm]

                s_sampled_batch.append(s_i[selected_indeces])
                t_sampled_batch.append(t_i[selected_indeces])

            return torch.stack(s_sampled_batch), torch.stack(t_sampled_batch)

        elif self.aggregation == "random":
            s_flat = s_embeddings.reshape(B, C, -1).transpose(1, 2)
            t_flat = t_embeddings.reshape(B, C, -1).transpose(1, 2)
            
            num_voxels = s_flat.size(1)
            
            # --- THE MEMORY SHIELD ---
            if num_voxels > self.max_samples:
                rand_indices = torch.randperm(num_voxels, device=s_embeddings.device)[:self.max_samples]
                s_flat = s_flat[:, rand_indices, :]
                t_flat = t_flat[:, rand_indices, :]
        
        
            return s_flat, t_flat
        
        else:
            raise ValueError(f"The aggregation mode {self.aggregation} is not supported")
        

class IntraPatientInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.1, aggregation: str = "avg_pool", max_samples: float = 4096):
        super().__init__()
        self.temperature = temperature
        self.aggregator = FeatureAggregator(aggregation=aggregation, max_samples=max_samples)

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
                targets = torch.arange(num_items, device=s_i.device)
                
                loss = F.cross_entropy(similarity_matrix, targets)
                scale_loss += loss
            
            total_loss += scale_loss / B

        return total_loss / len(s_embeddings_list)

import torch
from abc import ABC, abstractmethod

class BaseImportanceRanker(ABC):
    @abstractmethod
    def select_tokens(
        self,
        attn_weights: torch.Tensor,
        vision_token_indices: torch.Tensor,
        keep_ratio: float
    ) -> torch.Tensor:
        pass

class AttentionScoreRanker(BaseImportanceRanker):
    def select_tokens(
        self,
        attn_weights: torch.Tensor,
        vision_token_indices: torch.Tensor,
        keep_ratio: float
    ) -> torch.Tensor:
        
        avg_attn_received = attn_weights.mean(dim=(1, 2))  # [batch_size, seq_len][cite: 2]
        
        vision_scores = avg_attn_received[:, vision_token_indices]
        num_keep = int(vision_token_indices.shape[0] * keep_ratio)
        
        _, topk_local_indices = torch.topk(vision_scores, k=num_keep, dim=-1, largest=True)
        keep_indices = vision_token_indices[topk_local_indices.squeeze(0)]
        
        keep_indices, _ = torch.sort(keep_indices)
        return keep_indices
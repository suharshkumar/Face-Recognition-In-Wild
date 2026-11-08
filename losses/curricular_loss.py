import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class CurricularFaceLoss(nn.Module):
    """
    CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition (Huang et al., CVPR 2020)
    
    Dynamically adjusts the relative importance of easy and hard samples during training:
    - Early training: suppresses hard negative samples to avoid divergence from noisy/corrupted faces.
    - Late training: amplifies hard misclassified samples to enforce sharp inter-class margins.
    
    Includes PyTorch Distributed Data Parallel (DDP) all-reduce support and AMP FP16 safeguards.
    """
    def __init__(self, in_features: int, num_classes: int, s: float = 64.0, m: float = 0.50, alpha: float = 0.99):
        super(CurricularFaceLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.alpha = alpha
        
        # Linear classifier weights (C, d)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Exponential Moving Average parameter t
        self.register_buffer('t', torch.zeros(1))
        
        # Trigonometric constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, in_features) L2-normalized feature embeddings
            labels: (B,) ground truth class indices
        Returns:
            loss: scalar cross-entropy loss with adaptive curricular margin modulation
        """
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must be rank-1 and align with embeddings")
        if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= self.num_classes):
            raise ValueError("labels contain an index outside the classifier range")
        # Ensure FP32 precision for margin and cosine trigonometric calculations under AMP
        norm_embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        norm_weight = F.normalize(self.weight.float(), p=2, dim=1)
        
        # Cosine similarity matrix: (B, num_classes)
        cos_theta = F.linear(norm_embeddings, norm_weight)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        # Ground truth positive cosine similarities
        batch_size = embeddings.size(0)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        
        cos_yi = cos_theta[one_hot.bool()]
        
        # Update EMA curriculum parameter t during training across DDP nodes
        if self.training:
            with torch.no_grad():
                mean_cos = cos_yi.mean()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(mean_cos, op=dist.ReduceOp.SUM)
                    mean_cos /= dist.get_world_size()
                self.t = self.alpha * self.t + (1.0 - self.alpha) * mean_cos
        
        # Target angle margin: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        sin_yi = torch.sqrt((1.0 - cos_yi ** 2).clamp(min=1e-7))
        cos_yi_m = cos_yi * self.cos_m - sin_yi * self.sin_m
        
        # Fallback for angle overflow > pi
        cos_yi_m = torch.where(cos_yi > self.threshold, cos_yi_m, cos_yi - self.mm)
        
        # Build modulated negative similarity matrix
        target_margin = cos_yi_m.view(-1, 1)
        mask = 1.0 - one_hot
        
        # Hard sample criterion: cos(theta_yi + m) < cos(theta_j)
        is_hard = (target_margin < cos_theta) & mask.bool()
        
        # Modulation function N(t, cos_theta_j) = cos_theta_j * (t + cos_theta_j) for hard negatives
        modulated_negatives = torch.where(
            is_hard,
            cos_theta * (self.t + cos_theta),
            cos_theta
        )
        
        # Combine target positive logits and modulated negative logits in FP32 precision for AMP stability
        output_logits = torch.where(one_hot.bool(), target_margin, modulated_negatives).float()
        output_logits = output_logits * self.s
        
        loss = F.cross_entropy(output_logits, labels)
        return loss

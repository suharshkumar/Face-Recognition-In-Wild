"""CurricularFace with an explicit, DDP-safe BroadFace-style memory."""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class BroadFaceCurricularLoss(nn.Module):
    """Margin classifier plus a detached queue of past face embeddings.

    The queue is deliberately updated by :meth:`enqueue` after the optimizer
    step. This prevents the old implementation's duplicate current-batch loss
    and makes checkpoint/resume behavior explicit. Queue examples update the
    classifier only; the current batch updates both classifier and backbone.
    """

    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 100,
        scale_factor: float = 64.0,
        margin: float = 0.50,
        alpha: float = 0.99,
        queue_size: int = 10000,
        compensate: bool = True,
        feature_dim: int = None,
        sync_curriculum: bool = True,
    ):
        super().__init__()
        if feature_dim is not None:
            in_features = feature_dim
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if queue_size < 0:
            raise ValueError("queue_size cannot be negative")
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.scale_factor = float(scale_factor)
        self.margin = float(margin)
        self.alpha = float(alpha)
        self.queue_size = int(queue_size)
        self.compensate = bool(compensate)
        self.sync_curriculum = bool(sync_curriculum)

        self.weight = nn.Parameter(torch.empty(self.num_classes, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer("t", torch.zeros(1))
        self.register_buffer("feature_mb", torch.zeros(self.queue_size, self.in_features))
        self.register_buffer("label_mb", torch.zeros(self.queue_size, dtype=torch.long))
        self.register_buffer("proxy_mb", torch.zeros(self.queue_size, self.in_features))
        self.register_buffer("queue_length", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_next", torch.zeros(1, dtype=torch.long))

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    @property
    def queue_ptr(self) -> torch.Tensor:
        return self.queue_length.to(self.weight.device)

    @property
    def is_full(self) -> torch.Tensor:
        return self.queue_length.to(self.weight.device) >= self.queue_size

    def get_queue_samples(self):
        length = int(self.queue_length.item())
        return self.feature_mb[:length], self.label_mb[:length]

    def _validate_labels(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        if labels.ndim != 1:
            raise ValueError(f"labels must be rank-1, got {tuple(labels.shape)}")
        if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= self.num_classes):
            raise ValueError("labels contain an index outside the classifier range")
        return labels

    def compute_curricular(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                           update_curriculum: bool = True) -> torch.Tensor:
        y = self._validate_labels(y)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"embeddings must have shape (B,{self.in_features})")
        norm_embeddings = F.normalize(x.float(), p=2, dim=1)
        norm_weight = F.normalize(w.float(), p=2, dim=1)
        cos_theta = F.linear(norm_embeddings, norm_weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        one_hot = F.one_hot(y, num_classes=self.num_classes).to(cos_theta.dtype)
        cos_yi = (cos_theta * one_hot).sum(dim=1)

        if self.training and update_curriculum and cos_yi.numel():
            with torch.no_grad():
                mean_cos = cos_yi.mean()
                if self.sync_curriculum and dist.is_available() and dist.is_initialized():
                    dist.all_reduce(mean_cos, op=dist.ReduceOp.SUM)
                    mean_cos /= dist.get_world_size()
                self.t.mul_(self.alpha).add_((1.0 - self.alpha) * mean_cos)

        sin_yi = torch.sqrt((1.0 - cos_yi.square()).clamp_min(1e-7))
        cos_yi_m = cos_yi * self.cos_m - sin_yi * self.sin_m
        cos_yi_m = torch.where(cos_yi > self.th, cos_yi_m, cos_yi - self.mm)
        target_margin = cos_yi_m[:, None]
        is_hard = (target_margin < cos_theta) & (one_hot == 0)
        negatives = torch.where(is_hard, cos_theta * (self.t + cos_theta), cos_theta)
        logits = torch.where(one_hot.bool(), target_margin, negatives) * self.scale_factor
        return F.cross_entropy(logits.float(), y)

    def _compensated_queue(self) -> torch.Tensor:
        length = int(self.queue_length.item())
        if length == 0:
            return self.feature_mb[:0]
        features = self.feature_mb[:length]
        labels = self.label_mb[:length]
        proxies = self.proxy_mb[:length]
        current_proxy = self.weight.detach()[labels]
        if not self.compensate:
            return features
        drift = current_proxy - proxies
        scale = features.norm(dim=1, keepdim=True) / proxies.norm(dim=1, keepdim=True).clamp_min(1e-7)
        return features + scale * drift

    def forward(self, input_tensor: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        label = self._validate_labels(label)
        current_loss = self.compute_curricular(input_tensor, label, self.weight)
        if int(self.queue_length.item()) == 0:
            return current_loss

        # Queue embeddings are detached, so this term trains the classifier
        # without backpropagating stale representations into the backbone.
        queue_features = self._compensated_queue().detach()
        queue_loss = self.compute_curricular(queue_features, self.label_mb[:int(self.queue_length.item())], self.weight, update_curriculum=False)
        return current_loss + queue_loss

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Enqueue samples after an optimizer step, synchronizing DDP ranks."""
        labels = self._validate_labels(labels).detach()
        embeddings = F.normalize(embeddings.detach().float(), p=2, dim=1)
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("embeddings and labels must contain the same number of samples")
        if embeddings.numel() == 0 or self.queue_size == 0:
            return

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_features = [torch.empty_like(embeddings) for _ in range(world_size)]
            gathered_labels = [torch.empty_like(labels) for _ in range(world_size)]
            dist.all_gather(gathered_features, embeddings)
            dist.all_gather(gathered_labels, labels)
            embeddings = torch.cat(gathered_features, dim=0)
            labels = torch.cat(gathered_labels, dim=0)

        proxies = self.weight.detach()[labels]
        if embeddings.shape[0] >= self.queue_size:
            embeddings = embeddings[-self.queue_size:]
            labels = labels[-self.queue_size:]
            proxies = proxies[-self.queue_size:]
            self.feature_mb.copy_(embeddings)
            self.label_mb.copy_(labels)
            self.proxy_mb.copy_(proxies)
            self.queue_length.fill_(self.queue_size)
            self.queue_next.zero_()
            return

        pointer = int(self.queue_next.item())
        length = int(self.queue_length.item())
        for index in range(embeddings.shape[0]):
            self.feature_mb[pointer].copy_(embeddings[index])
            self.label_mb[pointer].copy_(labels[index])
            self.proxy_mb[pointer].copy_(proxies[index])
            pointer = (pointer + 1) % self.queue_size
            length = min(self.queue_size, length + 1)
        self.queue_next.fill_(pointer)
        self.queue_length.fill_(length)

    # Compatibility with existing callers. New training code should use
    # enqueue() after optimizer.step().
    update = enqueue

    @torch.no_grad()
    def compensate_weight_drift(self, old_weight: torch.Tensor, new_weight: torch.Tensor):
        length = int(self.queue_length.item())
        if length == 0:
            return
        labels = self.label_mb[:length]
        delta = new_weight.detach()[labels] - old_weight.detach()[labels]
        self.proxy_mb[:length].add_(delta)


BroadFaceMemoryQueue = BroadFaceCurricularLoss

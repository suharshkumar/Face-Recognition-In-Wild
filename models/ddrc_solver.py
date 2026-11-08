"""Dictionary-conditioned sparse open-set classification.

The previous implementation generated coefficients from an unrelated random
matrix and then classified with cosine similarity. This version makes every
LISTA/ISTA step use the enrolled dictionary and computes genuine class-specific
reconstruction residuals. The dictionary can be trained externally or filled
from an enrollment gallery.
"""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Open-set defaults. ``residual_threshold`` is a *relative* residual: the
# fraction of the (outlier-cleaned) query energy that the best class fails to
# reconstruct, so it lives in [0, 1] regardless of the embedding dimension.
# With one atom per class, 0.80 accepts roughly cosine >= 0.45 to the class
# prototype; ``margin_threshold`` is the relative gap to the runner-up class.
# Both should be re-selected on a held-out known/unknown split
# (see calibrate_open_set.py).
DEFAULT_RESIDUAL_THRESHOLD = 0.80
DEFAULT_MARGIN_THRESHOLD = 0.10


def spherical_kmeans(embeddings: torch.Tensor, k: int, iterations: int = 25) -> torch.Tensor:
    """Return ``k`` unit-norm centroids that summarise ``embeddings``.

    Deterministic: initialisation is farthest-point sampling seeded at the
    sample closest to the mean, so the same gallery always yields the same
    dictionary. ``k == 1`` reduces to the normalised mean, the standard
    face-recognition class prototype.
    """
    x = F.normalize(embeddings.float(), p=2, dim=1)
    n = x.shape[0]
    if n == 0:
        raise ValueError("spherical_kmeans needs at least one embedding")
    mean = F.normalize(x.mean(dim=0, keepdim=True), p=2, dim=1)
    if k <= 1 or n == 1:
        return mean if k <= 1 else x[:1].repeat(k, 1)
    k = min(k, n)
    chosen = [int((x @ mean.T).squeeze(1).argmax())]
    closest = x @ x[chosen[0]]
    for _ in range(1, k):
        nxt = int(closest.argmin())
        chosen.append(nxt)
        closest = torch.maximum(closest, x @ x[nxt])
    centroids = x[chosen].clone()
    for _ in range(iterations):
        assign = (x @ centroids.T).argmax(dim=1)
        updated = torch.zeros_like(centroids).index_add_(0, assign, x)
        empty = updated.norm(dim=1) < 1e-8
        updated[empty] = centroids[empty]
        updated = F.normalize(updated, p=2, dim=1)
        if torch.allclose(updated, centroids, atol=1e-6):
            centroids = updated
            break
        centroids = updated
    return centroids


def _soft_threshold(value: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * F.relu(value.abs() - threshold)


class LISTASparseSolver(nn.Module):
    """Fixed-step unrolled robust sparse coding for a supplied dictionary."""

    def __init__(self, feature_dim: int = 512, num_atoms: int = 2048,
                 num_layers: int = 5, theta: float = 0.05):
        super().__init__()
        if feature_dim <= 0 or num_atoms <= 0 or num_layers <= 0:
            raise ValueError("feature_dim, num_atoms, and num_layers must be positive")
        self.feature_dim = feature_dim
        self.num_atoms = num_atoms
        self.num_layers = num_layers
        self.log_step = nn.Parameter(torch.tensor(-1.0))
        self.threshold_x = nn.ParameterList([
            nn.Parameter(torch.full((num_atoms,), theta)) for _ in range(num_layers)
        ])
        self.threshold_e = nn.ParameterList([
            nn.Parameter(torch.tensor(theta)) for _ in range(num_layers)
        ])

    def forward(self, f: torch.Tensor, dictionary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if f.ndim != 2 or f.shape[1] != self.feature_dim:
            raise ValueError(f"f must have shape (B,{self.feature_dim})")
        if dictionary.ndim != 2 or dictionary.shape != (self.feature_dim, self.num_atoms):
            raise ValueError(
                f"dictionary must have shape ({self.feature_dim},{self.num_atoms}), "
                f"got {tuple(dictionary.shape)}"
            )
        d = F.normalize(dictionary.float(), p=2, dim=0)
        f = f.float()
        x = torch.zeros((f.shape[0], self.num_atoms), device=f.device, dtype=f.dtype)
        e = torch.zeros_like(f)
        step = self.log_step.exp().clamp(1e-4, 2.0)
        for layer in range(self.num_layers):
            residual = f - x @ d.T - e
            x = _soft_threshold(x + step * (residual @ d), self.threshold_x[layer].abs())
            e = _soft_threshold(f - x @ d.T, self.threshold_e[layer].abs())
        return x, e


class DDRCClassifier(nn.Module):
    """Gallery dictionary classifier with thresholded open-set gating.

    The thresholds become calibrated only after they are selected on a held-out
    known/unknown validation protocol.
    """

    def __init__(self, feature_dim: int = 512, num_classes: int = 0,
                 atoms_per_class: int = 20, num_layers: int = 5,
                 residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
                 margin_threshold: float = DEFAULT_MARGIN_THRESHOLD):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.atoms_per_class = max(1, int(atoms_per_class))
        self.residual_threshold = float(residual_threshold)
        self.margin_threshold = float(margin_threshold)
        self.num_classes = 0
        self.total_atoms = 0
        self.class_names: List[str] = []
        self.register_parameter("dictionary", None)
        self.lista_solver = LISTASparseSolver(
            feature_dim=self.feature_dim,
            num_atoms=1,
            num_layers=num_layers,
        )
        if num_classes:
            self._reset_dictionary(num_classes, device=torch.device("cpu"))

    def _reset_dictionary(self, num_classes: int, device: torch.device) -> None:
        if num_classes <= 0:
            self.num_classes = 0
            self.total_atoms = 0
            self.dictionary = None
            self.class_names = []
            return
        self.num_classes = int(num_classes)
        self.total_atoms = self.num_classes * self.atoms_per_class
        self.dictionary = nn.Parameter(
            torch.empty(self.feature_dim, self.total_atoms, device=device)
        )
        nn.init.xavier_uniform_(self.dictionary)
        self.lista_solver = LISTASparseSolver(
            feature_dim=self.feature_dim,
            num_atoms=self.total_atoms,
            num_layers=self.lista_solver.num_layers,
        ).to(device)
        self.class_names = [f"Person_{i}" for i in range(self.num_classes)]

    def set_class_names(self, class_names: List[str]) -> None:
        if not class_names:
            self._reset_dictionary(0, next(self.parameters()).device)
            return
        device = self.dictionary.device if self.dictionary is not None else next(self.parameters()).device
        old_atoms = self.atoms_per_class
        self._reset_dictionary(len(class_names), device)
        self.atoms_per_class = old_atoms
        self.class_names = list(class_names)

    @torch.no_grad()
    def enroll_prototypes(self, class_prototypes: torch.Tensor,
                          class_names: Optional[List[str]] = None) -> None:
        if class_prototypes.ndim != 2 or class_prototypes.shape[1] != self.feature_dim:
            raise ValueError(f"prototypes must have shape (C,{self.feature_dim})")
        device = self.dictionary.device if self.dictionary is not None else next(self.parameters()).device
        prototypes = F.normalize(class_prototypes.to(device).float(), p=2, dim=1)
        self._reset_dictionary(prototypes.shape[0], device)
        atoms = prototypes[:, None, :].expand(-1, self.atoms_per_class, -1)
        self.dictionary.copy_(atoms.reshape(-1, self.feature_dim).T)
        if class_names is not None:
            if len(class_names) != prototypes.shape[0]:
                raise ValueError("class_names must match the number of prototypes")
            self.class_names = list(class_names)

    @torch.no_grad()
    def enroll_gallery(self, embeddings: torch.Tensor, labels: torch.Tensor,
                       class_names: Optional[Sequence[str]] = None) -> None:
        """Build atoms from all gallery embeddings instead of classifier weights."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.feature_dim:
            raise ValueError(f"embeddings must have shape (N,{self.feature_dim})")
        if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
            raise ValueError("labels must be rank-1 and align with embeddings")
        if embeddings.shape[0] == 0:
            self._reset_dictionary(0, embeddings.device)
            return
        labels = labels.long()
        unique = torch.unique(labels, sorted=True)
        names = list(class_names) if class_names is not None else [f"Person_{int(i)}" for i in unique]
        if len(names) != unique.numel():
            raise ValueError("class_names must match the number of unique labels")
        device = embeddings.device
        self._reset_dictionary(unique.numel(), device)
        blocks = []
        for class_id in unique:
            class_embeddings = F.normalize(embeddings[labels == class_id].float(), p=2, dim=1)
            if class_embeddings.shape[0] > self.atoms_per_class:
                # Summarise *every* enrolled image. Taking the first N files
                # discarded the rest of the gallery, so one bad first photo
                # became the whole identity.
                block = spherical_kmeans(class_embeddings, self.atoms_per_class)
            else:
                block = class_embeddings
            if block.shape[0] < self.atoms_per_class:
                block = block.repeat((self.atoms_per_class + block.shape[0] - 1) // block.shape[0], 1)[:self.atoms_per_class]
            blocks.append(block)
        self.dictionary.copy_(torch.cat(blocks, dim=0).T)
        self.class_names = names

    def class_residuals(self, f: torch.Tensor) -> torch.Tensor:
        """Relative per-class reconstruction residuals, shape ``(B, C)``.

        Each value is ``||clean - D_c x_c||^2 / ||clean||^2``: the fraction of
        the outlier-cleaned query that class ``c`` cannot explain. It is scale
        free, so thresholds mean the same thing for any embedding size. (A
        per-dimension mean is bounded by ``4 / feature_dim`` for unit vectors,
        which left a 0.55 threshold unable to reject anything.)
        """
        if f.ndim != 2 or f.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B,{self.feature_dim})")
        if self.dictionary is None or self.num_classes == 0:
            return torch.empty(f.shape[0], 0, device=f.device)
        norm_f = F.normalize(f.float(), p=2, dim=1)
        x, error = self.lista_solver(norm_f, self.dictionary)
        clean = norm_f - error
        energy = clean.square().sum(dim=1).clamp_min(1e-12)
        dictionary = F.normalize(self.dictionary.float(), p=2, dim=0)
        residuals = []
        for class_id in range(self.num_classes):
            start = class_id * self.atoms_per_class
            end = start + self.atoms_per_class
            class_reconstruction = x[:, start:end] @ dictionary[:, start:end].T
            residuals.append((clean - class_reconstruction).square().sum(dim=1) / energy)
        return torch.stack(residuals, dim=1)

    def decide(self, class_residuals: torch.Tensor,
               residual_threshold: Optional[float] = None,
               margin_threshold: Optional[float] = None):
        """Apply the open-set gates to precomputed residuals.

        Split from :meth:`class_residuals` so threshold sweeps reuse one
        LISTA pass per query.
        """
        residual_threshold = self.residual_threshold if residual_threshold is None else float(residual_threshold)
        margin_threshold = self.margin_threshold if margin_threshold is None else float(margin_threshold)
        batch_size = class_residuals.shape[0]
        device = class_residuals.device
        if class_residuals.shape[1] == 0:
            return ["unknown"] * batch_size, torch.zeros(batch_size, device=device)
        sorted_residuals, sorted_indices = torch.sort(class_residuals, dim=1)
        best = sorted_indices[:, 0]
        best_residual = sorted_residuals[:, 0]
        if class_residuals.shape[1] > 1:
            second = sorted_residuals[:, 1]
            margin = (second - best_residual) / second.clamp_min(1e-7)
        else:
            # With one enrolled identity there is no runner-up; the residual
            # gate alone decides, so a stranger is not auto-accepted.
            margin = torch.ones_like(best_residual)

        confidence = torch.exp(-best_residual / max(residual_threshold, 1e-6))
        confidence = (confidence * margin.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        known = (best_residual <= residual_threshold) & (margin >= margin_threshold)
        predictions = [
            self.class_names[int(best[index])] if bool(known[index]) else "unknown"
            for index in range(batch_size)
        ]
        return predictions, confidence

    def forward(self, f: torch.Tensor, residual_threshold: Optional[float] = None,
                margin_threshold: Optional[float] = None):
        """Return ``(names, confidences, relative_residuals)`` for a batch."""
        residuals = self.class_residuals(f)
        predictions, confidence = self.decide(residuals, residual_threshold, margin_threshold)
        return predictions, confidence, residuals

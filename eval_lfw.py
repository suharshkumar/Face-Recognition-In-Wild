"""Face-verification accuracy on LFW-style pairs (standard 10-fold protocol).

This measures the *recognition backbone* on its own: for each image pair it
decides "same person" or "different person" from the cosine similarity of the
two embeddings. Following the LFW "unrestricted, labelled outside data"
protocol, the decision threshold is chosen on 9 folds and scored on the
held-out 10th, and the 10 held-out accuracies are averaged.

Inputs
------
``--pairs``       LFW ``pairs.txt`` (first line ``<folds> <pairs_per_class>``,
                  then ``name i j`` for matched and ``name1 i name2 j`` for
                  mismatched pairs).
``--images_dir``  directory laid out as ``<name>/<name>_<4-digit index>.jpg``.

Use a benchmark that does not overlap the training data. This project's
``name_label`` stream contains LFW, so LFW pairs would measure memorisation;
CFP-FP or AgeDB-30 pairs in the same format are the honest choice.

The backbone expects tight 112x112 face crops. Raw LFW images are 250x250
with the face in the centre, so ``--center_crop 0.56`` (default) crops the
central 140 px before resizing; pass ``--center_crop 1.0`` for pre-aligned
crops such as the insightface ``lfw`` bin export.

Example::

    python eval_lfw.py --weights_dir weights \\
        --pairs lfw/pairs.txt --images_dir lfw/lfw-deepfunneled
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_default_transform


def parse_pairs(pairs_path: str, images_dir: str) -> Tuple[List[Tuple[str, str]], List[bool]]:
    """Return ``(path_a, path_b)`` pairs and same-identity flags."""
    def image(name: str, index: str) -> str:
        return os.path.join(images_dir, name, f"{name}_{int(index):04d}.jpg")

    pairs, issame = [], []
    with open(pairs_path, encoding="utf-8") as stream:
        lines = [line.split() for line in stream if line.strip()]
    if lines and len(lines[0]) == 2 and all(token.isdigit() for token in lines[0]):
        lines = lines[1:]  # "<folds> <pairs per class>" header
    for fields in lines:
        if len(fields) == 3:
            pairs.append((image(fields[0], fields[1]), image(fields[0], fields[2])))
            issame.append(True)
        elif len(fields) == 4:
            pairs.append((image(fields[0], fields[1]), image(fields[2], fields[3])))
            issame.append(False)
        else:
            raise ValueError(f"malformed pairs line: {' '.join(fields)}")
    return pairs, issame


def kfold_verification_accuracy(similarity: torch.Tensor, issame: torch.Tensor, folds: int = 10):
    """Mean/std held-out accuracy and mean threshold over ``folds`` folds.

    Folds are contiguous blocks, matching LFW's pairs.txt layout. For each
    fold the threshold maximising accuracy on the other folds is applied to it.
    """
    similarity = similarity.float().cpu()
    issame = issame.bool().cpu()
    n = similarity.numel()
    if n != issame.numel() or n < folds:
        raise ValueError("need aligned similarity/issame with at least one pair per fold")
    candidates = torch.unique(similarity)
    # Midpoints between sorted unique scores plus both ends are the only
    # thresholds that can change a decision.
    thresholds = torch.cat([candidates[:1] - 1e-6, (candidates[1:] + candidates[:-1]) / 2, candidates[-1:] + 1e-6])
    bounds = torch.linspace(0, n, folds + 1).round().long().tolist()
    accuracies, chosen = [], []
    for k in range(folds):
        test = torch.zeros(n, dtype=torch.bool)
        test[bounds[k]:bounds[k + 1]] = True
        train = ~test
        predictions = similarity[train][None, :] > thresholds[:, None]
        train_acc = (predictions == issame[train][None, :]).float().mean(dim=1)
        threshold = thresholds[int(train_acc.argmax())]
        accuracies.append(((similarity[test] > threshold) == issame[test]).float().mean().item())
        chosen.append(float(threshold))
    acc = torch.tensor(accuracies)
    return float(acc.mean()), float(acc.std(unbiased=False)), sum(chosen) / len(chosen)


@torch.no_grad()
def embed_images(backbone, paths: Sequence[str], device, center_crop: float, flip: bool, batch_size: int = 64):
    transform = get_default_transform((112, 112), is_train=False)
    backbone.eval()
    out = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as source:
                image = source.convert("RGB")
            if center_crop < 1.0:
                w, h = image.size
                cw, ch = w * center_crop, h * center_crop
                image = image.crop((round((w - cw) / 2), round((h - ch) / 2), round((w + cw) / 2), round((h + ch) / 2)))
            batch.append(transform(image))
        x = torch.stack(batch).to(device)
        emb = backbone(x)
        if flip:  # standard test-time flip: sum original and mirrored features
            emb = emb + backbone(torch.flip(x, dims=[3]))
        out.append(F.normalize(emb.float(), dim=1).cpu())
    return torch.cat(out)


def run(args: argparse.Namespace):
    from inference import load_pipeline, resolve_device

    device = resolve_device(args.device)
    pipeline = load_pipeline(args.weights_dir, gallery_dir=None, device=device)
    if not pipeline.has_trained_backbone:
        raise RuntimeError(f"no trained recognition backbone found in {args.weights_dir}")
    pairs, issame = parse_pairs(args.pairs, args.images_dir)
    unique = sorted({p for pair in pairs for p in pair})
    missing = [p for p in unique if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} pair images are missing, e.g. {missing[0]}")
    print(f"Embedding {len(unique)} images for {len(pairs)} pairs")
    embeddings = embed_images(pipeline.backbone, unique, device, args.center_crop, not args.no_flip)
    index = {p: i for i, p in enumerate(unique)}
    a = embeddings[[index[p] for p, _ in pairs]]
    b = embeddings[[index[q] for _, q in pairs]]
    similarity = (a * b).sum(dim=1)
    mean, std, threshold = kfold_verification_accuracy(similarity, torch.tensor(issame), args.folds)
    print(f"Verification accuracy: {mean:.2%} +/- {std:.2%} ({args.folds}-fold, cosine threshold ~{threshold:.3f})")
    return mean, std, threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LFW-style 10-fold face verification accuracy")
    parser.add_argument("--weights_dir", default="./weights")
    parser.add_argument("--pairs", required=True, help="LFW pairs.txt")
    parser.add_argument("--images_dir", required=True, help="<name>/<name>_0001.jpg image tree")
    parser.add_argument("--center_crop", type=float, default=0.56,
                        help="fraction of each image kept around the centre (1.0 = already cropped)")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--no_flip", action="store_true", help="disable flip test-time augmentation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

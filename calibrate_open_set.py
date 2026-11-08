"""Choose open-set thresholds on a held-out known/unknown split.

The DDRC gallery classifier answers "who is this?" *or* "unknown". Two gates
control that trade-off (see ``models/ddrc_solver.py``):

* ``residual_threshold`` - max fraction of the query embedding the best
  identity may leave unexplained (0-1, higher accepts more);
* ``margin_threshold``   - min relative gap between the best and runner-up
  identity.

This script enrolls ``--gallery_dir`` exactly like ``inference.py``, embeds
two held-out probe sets of *face crops*, and sweeps both gates:

* ``--known_dir``   - other photos of gallery identities, one folder per
  person using the same folder names as the gallery;
* ``--unknown_dir`` - photos of people who are NOT in the gallery.

For every pair it reports the open-set identification rate (known probes given
the correct name), the misidentification rate, and the false-accept rate
(unknown probes given any name). It then recommends the pair with the highest
identification rate whose false-accept rate stays within ``--max_false_accept``.

Example::

    python calibrate_open_set.py \\
        --weights_dir weights \\
        --gallery_dir Face_Dataset/name_label \\
        --known_dir   Face_Dataset/heldout_known \\
        --unknown_dir Face_Dataset/heldout_unknown \\
        --max_false_accept 0.01

then pass the printed values to ``inference.py --residual_threshold ...
--margin_threshold ...``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

from dataset import NameLabeledFaceDataset, get_default_transform
from models.ddrc_solver import DDRCClassifier

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_RESIDUAL_GRID = [round(0.40 + 0.025 * i, 3) for i in range(23)]  # 0.40 .. 0.95
DEFAULT_MARGIN_GRID = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]


def sweep_open_set(
    classifier: DDRCClassifier,
    known_residuals: torch.Tensor,
    known_names: Sequence[str],
    unknown_residuals: torch.Tensor,
    residual_grid: Sequence[float] = DEFAULT_RESIDUAL_GRID,
    margin_grid: Sequence[float] = DEFAULT_MARGIN_GRID,
) -> List[Dict[str, float]]:
    """Evaluate every threshold pair on precomputed class residuals."""
    if known_residuals.shape[0] != len(known_names):
        raise ValueError("known_residuals and known_names must align")
    if known_residuals.shape[0] == 0 and unknown_residuals.shape[0] == 0:
        raise ValueError("need at least one known or unknown probe")
    rows = []
    for residual_threshold in residual_grid:
        for margin_threshold in margin_grid:
            row = {"residual_threshold": float(residual_threshold), "margin_threshold": float(margin_threshold)}
            if known_residuals.shape[0]:
                predicted, _ = classifier.decide(known_residuals, residual_threshold, margin_threshold)
                correct = sum(p == t for p, t in zip(predicted, known_names))
                rejected = sum(p == "unknown" for p in predicted)
                n_known = len(known_names)
                row["identification_rate"] = correct / n_known
                row["misidentification_rate"] = (n_known - correct - rejected) / n_known
                row["known_rejection_rate"] = rejected / n_known
            if unknown_residuals.shape[0]:
                predicted, _ = classifier.decide(unknown_residuals, residual_threshold, margin_threshold)
                row["false_accept_rate"] = sum(p != "unknown" for p in predicted) / len(predicted)
            rows.append(row)
    return rows


def recommend(rows: List[Dict[str, float]], max_false_accept: float) -> Optional[Dict[str, float]]:
    """Highest identification rate within the false-accept budget.

    Ties go to the lower false-accept rate, then to the stricter gates.
    Returns ``None`` when no pair meets the budget.
    """
    feasible = [r for r in rows if r.get("false_accept_rate", 0.0) <= max_false_accept]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda r: (
            r.get("identification_rate", 0.0),
            -r.get("false_accept_rate", 0.0),
            -r["residual_threshold"],
            r["margin_threshold"],
        ),
    )


@torch.no_grad()
def embed_face_crops(pipeline, paths: Sequence[str], device: torch.device, batch_size: int = 64) -> torch.Tensor:
    """Embed face crops with the same transform and path used for enrollment."""
    transform = get_default_transform((112, 112), is_train=False)
    pipeline.eval()
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                batch.append(transform(image.convert("RGB")))
        embedding, _, _, _, _ = pipeline._embed_crop(torch.stack(batch).to(device), None)
        chunks.append(embedding.cpu())
    if not chunks:
        return torch.empty(0, pipeline.feature_dim)
    return torch.cat(chunks)


def _image_files(root: str) -> List[str]:
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"probe directory was not found: {root}")
    return [str(p) for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]


def run(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    from inference import load_pipeline, resolve_device

    device = resolve_device(args.device)
    pipeline = load_pipeline(
        weights_dir=args.weights_dir,
        gallery_dir=args.gallery_dir,
        device=device,
        detector_checkpoint=None,
        atoms_per_class=args.atoms_per_class,
    )
    classifier = pipeline.ddrc_classifier
    if classifier.num_classes == 0:
        raise RuntimeError("the gallery is empty; check --gallery_dir")
    if not pipeline.has_trained_backbone:
        print("Warning: no trained recognition backbone was loaded; thresholds will be meaningless")

    known = NameLabeledFaceDataset(name_label_dir=args.known_dir, is_train=False)
    idx_to_name = {index: pipeline.display_name(name) for name, index in known.class_to_idx.items()}
    known_paths = [path for path, _ in known.samples]
    known_names = [idx_to_name[label] for _, label in known.samples]
    missing = sorted(set(known_names) - set(classifier.class_names))
    if missing:
        raise ValueError(
            f"{len(missing)} known-probe identities are not enrolled in the gallery, e.g. {missing[:5]}; "
            "known probes must use the gallery's folder names"
        )
    unknown_paths = _image_files(args.unknown_dir)
    leaked = sorted(
        {pipeline.display_name(Path(p).parent.name) for p in unknown_paths} & set(classifier.class_names)
    )
    if leaked:
        print(f"Warning: unknown-probe folders share names with enrolled identities: {leaked[:5]}")

    print(f"Embedding {len(known_paths)} known and {len(unknown_paths)} unknown probe crops")
    known_res = classifier.class_residuals(embed_face_crops(pipeline, known_paths, device).to(device))
    unknown_res = classifier.class_residuals(embed_face_crops(pipeline, unknown_paths, device).to(device))
    rows = sweep_open_set(classifier, known_res, known_names, unknown_res)

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} threshold pairs to {output}")

    best = recommend(rows, args.max_false_accept)
    if best is None:
        print(f"No threshold pair keeps false accepts <= {args.max_false_accept:.3f}; "
              "enroll more/better gallery photos or relax --max_false_accept")
        return None
    print(
        "Recommended: --residual_threshold {residual_threshold:.3f} --margin_threshold {margin_threshold:.2f}\n"
        "  identification {identification_rate:.1%}, misidentified {misidentification_rate:.1%}, "
        "false accepts {false_accept_rate:.1%}".format(**best)
    )
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate open-set recognition thresholds")
    parser.add_argument("--weights_dir", default="./weights")
    parser.add_argument("--gallery_dir", required=True, help="enrollment crops, one folder per identity")
    parser.add_argument("--known_dir", required=True, help="held-out crops of enrolled identities")
    parser.add_argument("--unknown_dir", required=True, help="crops of people who are not enrolled")
    parser.add_argument("--max_false_accept", type=float, default=0.01)
    parser.add_argument("--atoms_per_class", type=int, default=1)
    parser.add_argument("--output_csv", default="./inference_results/open_set_sweep.csv")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

"""Canonical inference entry point for the multi-phase face pipeline.

The training phases do not produce one monolithic neural network.  This
module loads their compatible artifacts into ``OccuPoseBroadDictPipeline``:

* Phase 0 supplies localization regions and detector confidence.
* Phase 1, or the Phase 3 adapted replacement, supplies 512-D embeddings.
* Phase 2 supplies attribute and occlusion predictions when its checkpoint
  is present.
* Gallery enrollment supplies the names used by the open-set dictionary
  classifier.

The Phase 3 graph predictor is a training-time link model.  Its main
inference effect is the adapted backbone checkpoint; the final image query is
classified against the enrolled gallery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont

from dataset import pil_to_float_tensor
from models.ddrc_solver import DEFAULT_MARGIN_THRESHOLD, DEFAULT_RESIDUAL_THRESHOLD
from pipeline.wild_face_pipeline import OccuPoseBroadDictPipeline


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``, ``cpu``, or ``cuda`` without silently faking CUDA."""
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_detector_checkpoint(weights_dir: Path, explicit: Optional[str]) -> Optional[Path]:
    """Find the Phase 0 artifact, including the organized local subfolder."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            weights_dir / "face_detector.pt",
            weights_dir / "phase0_detector" / "face_detector.pt",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def image_paths(image: Optional[str], input_dir: Optional[str]) -> List[Path]:
    """Return deterministic image inputs from either one file or a directory."""
    if bool(image) == bool(input_dir):
        raise ValueError("provide exactly one of --image or --input_dir")
    if image:
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"image was not found: {path}")
        return [path]

    root = Path(input_dir or "")
    if not root.is_dir():
        raise FileNotFoundError(f"input directory was not found: {root}")
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"no supported images were found under: {root}")
    return paths


def load_pipeline(
    weights_dir: str,
    gallery_dir: Optional[str],
    device: torch.device,
    detector_checkpoint: Optional[str] = None,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    atoms_per_class: int = 1,
) -> OccuPoseBroadDictPipeline:
    """Load detector, recognition, attributes, adaptation, and gallery state."""
    weights_root = Path(weights_dir)
    detector_path = resolve_detector_checkpoint(weights_root, detector_checkpoint)
    pipeline = OccuPoseBroadDictPipeline(
        feature_dim=512,
        detector_weights_dir=str(weights_root),
        detector_checkpoint=str(detector_path) if detector_path else None,
        residual_threshold=residual_threshold,
        margin_threshold=margin_threshold,
        atoms_per_class=atoms_per_class,
    ).to(device)
    pipeline.load_pretrained_weights(checkpoint_dir=str(weights_root), device=device)

    if gallery_dir:
        gallery_path = Path(gallery_dir)
        if gallery_path.is_dir():
            enrolled = pipeline.enroll_dataset(
                str(gallery_path),
                device=device,
                cache_path=str(weights_root / "gallery_embeddings.pt"),
            )
            print(f"Enrolled {enrolled} gallery identities from {gallery_path}")
        else:
            print(f"Warning: gallery directory was not found; identities remain unknown: {gallery_path}")
    return pipeline.eval()


def render_predictions(image_path: Path, predictions: List[Dict[str, Any]], output_path: Path) -> None:
    """Save an annotated copy of an image with detected face boxes."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for prediction in predictions:
        box = prediction.get("box_xyxy", prediction.get("box"))
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        identity = str(prediction.get("identity", "unknown"))
        detector_score = float(prediction.get("detector_score", 0.0))
        label = f"{identity} ({detector_score:.2f})"
        color = "lime" if identity != "unknown" else "red"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_bbox = draw.textbbox((x1, max(0, y1 - 18)), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((text_bbox[0] + 2, text_bbox[1] + 1), label, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def json_safe(value: Any) -> Any:
    """Convert tensors and nested prediction values into JSON-compatible data."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


@torch.inference_mode()
def predict_image(
    pipeline: OccuPoseBroadDictPipeline,
    path: Path,
    device: torch.device,
    score_threshold: float,
) -> Dict[str, Any]:
    """Run one image through localization, embedding, attributes, and identity."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    original_width, original_height = rgb.size
    # Keep the scene at its native resolution.  Faster R-CNN owns the
    # detector resize policy (min_size=800, max_size=1333); shrinking a full
    # scene to 224x224 first removes small faces before the detector sees it.
    tensor = pil_to_float_tensor(rgb).mul(2.0).sub(1.0).unsqueeze(0).to(device)
    predictions = pipeline(tensor, score_threshold=score_threshold)
    scale_x = original_width / float(tensor.shape[3])
    scale_y = original_height / float(tensor.shape[2])
    for prediction in predictions:
        box = prediction.get("box_xyxy", prediction.get("box"))
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        original_box = [
            max(0, min(original_width, int(round(x1 * scale_x)))),
            max(0, min(original_height, int(round(y1 * scale_y)))),
            max(0, min(original_width, int(round(x2 * scale_x)))),
            max(0, min(original_height, int(round(y2 * scale_y)))),
        ]
        prediction["box"] = original_box
        prediction["box_xyxy"] = original_box
    return {"image": str(path), "predictions": json_safe(predictions)}


def run_inference(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    paths = image_paths(args.image, args.input_dir)
    pipeline = load_pipeline(
        weights_dir=args.weights_dir,
        gallery_dir=args.gallery_dir,
        device=device,
        detector_checkpoint=args.detector_checkpoint,
        residual_threshold=args.residual_threshold,
        margin_threshold=args.margin_threshold,
        atoms_per_class=args.atoms_per_class,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_json
    with output_path.open("w", encoding="utf-8") as stream:
        for path in paths:
            result = predict_image(pipeline, path, device, args.score_threshold)
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            annotated_path = output_dir / "annotated" / path.name
            render_predictions(path, result["predictions"], annotated_path)
            print(f"Processed {path}: {len(result['predictions'])} localized faces; saved {annotated_path}")
    print(f"Saved predictions to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run final multi-phase face inference")
    parser.add_argument("--image", help="single input image")
    parser.add_argument("--input_dir", help="directory of input images")
    parser.add_argument("--weights_dir", default="./weights", help="checkpoint bundle directory")
    parser.add_argument(
        "--detector_checkpoint",
        default=None,
        help="optional explicit Phase 0 checkpoint; nested phase0_detector/ is auto-discovered",
    )
    parser.add_argument(
        "--gallery_dir",
        default="./Face_Dataset/name_label",
        help="labelled face-crop directory used to build the identity gallery",
    )
    parser.add_argument("--output_dir", default="./inference_results")
    parser.add_argument("--output_json", default="predictions.jsonl")
    # Low-quality, blurred, and occluded faces can receive modest detector
    # scores even when the box is useful. Keep this lower than the training
    # pseudo-label cutoff; callers can raise it for precision-oriented runs.
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--residual_threshold",
        type=float,
        default=DEFAULT_RESIDUAL_THRESHOLD,
        help="open-set gate: max fraction of the query the best identity may leave unexplained "
        "(0-1; higher accepts more weak matches). Calibrate with calibrate_open_set.py",
    )
    parser.add_argument(
        "--margin_threshold",
        type=float,
        default=DEFAULT_MARGIN_THRESHOLD,
        help="open-set gate: min relative residual gap between the best and second-best identity",
    )
    parser.add_argument(
        "--atoms_per_class",
        type=int,
        default=1,
        help="gallery dictionary atoms per identity; >1 clusters each person's photos "
        "(e.g. with/without mask) into that many prototypes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

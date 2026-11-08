"""Phase 0 face-detector-only inference.

This script deliberately does not load the recognition backbone, CelebA
attributes, gallery, or Phase 3 adaptation. It loads only the trained Phase 0
``face_detector.pt`` checkpoint and returns face boxes and confidence scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image, ImageDraw

from dataset import pil_to_float_tensor
from models.detector import TorchvisionFaceDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint(weights_dir: str, explicit: Optional[str]) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    root = Path(weights_dir)
    candidates.extend((root / "face_detector.pt", root / "phase0_detector" / "face_detector.pt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Phase 0 checkpoint was not found. Searched: {searched}")


def resolve_images(image: Optional[str], input_dir: Optional[str]) -> List[Path]:
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
        raise FileNotFoundError(f"no supported images found under: {root}")
    return paths


def detector_input(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to the detector's expected BCHW range [-1, 1]."""
    return (pil_to_float_tensor(image).mul(2.0).sub(1.0)).unsqueeze(0)


def json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


@torch.inference_mode()
def predict(detector: TorchvisionFaceDetector, path: Path, device: torch.device, threshold: float) -> Dict[str, Any]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        tensor = detector_input(image).to(device)
    boxes, scores, landmarks = detector(tensor, score_threshold=threshold)
    return {
        "image": str(path),
        "width": width,
        "height": height,
        "boxes_xyxy": json_value(boxes),
        "scores": json_value(scores),
        "landmarks": json_value(landmarks),
    }


def save_overlay(path: Path, result: Dict[str, Any], output_dir: Path) -> Path:
    with Image.open(path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = result["boxes_xyxy"]
    scores = result["scores"]
    for box, score in zip(boxes, scores):
        left, top, right, bottom = (int(round(value)) for value in box)
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=3)
        draw.text((left, max(0, top - 16)), f"face {float(score):.3f}", fill=(255, 0, 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{path.stem}_detected{path.suffix.lower()}"
    image.save(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference using only the Phase 0 face detector")
    parser.add_argument("--image", help="single input image")
    parser.add_argument("--input_dir", help="directory of input images")
    parser.add_argument("--weights_dir", default="./weights", help="weights directory")
    parser.add_argument("--checkpoint", default=None, help="explicit face_detector.pt path")
    parser.add_argument("--output_json", default="detector_predictions.jsonl")
    parser.add_argument("--overlay_dir", default=None, help="optional directory for boxed output images")
    parser.add_argument("--score_threshold", type=float, default=0.40)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    checkpoint = resolve_checkpoint(args.weights_dir, args.checkpoint)
    paths = resolve_images(args.image, args.input_dir)
    detector = TorchvisionFaceDetector(
        checkpoint_path=str(checkpoint),
        pretrained_backbone=False,
    ).to(device).eval()
    print(f"Loaded Phase 0 detector: {checkpoint}")
    print(f"Device: {device}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for path in paths:
            result = predict(detector, path, device, args.score_threshold)
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            if args.overlay_dir:
                overlay = save_overlay(path, result, Path(args.overlay_dir))
                print(f"{path}: {len(result['boxes_xyxy'])} face(s), overlay={overlay}")
            else:
                print(f"{path}: {len(result['boxes_xyxy'])} face(s)")
    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()

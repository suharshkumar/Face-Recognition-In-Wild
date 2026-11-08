"""Face-crop alignment and landmark-based pose utilities.

The recognition backbone is trained on a fixed canonical face coordinate system.
This module is deliberately independent from the detector so that training,
enrollment, validation, and inference can share exactly the same transform.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ArcFace-style five-point template for a 112x112 crop:
# left eye, right eye, nose, left mouth corner, right mouth corner.
CANONICAL_LANDMARKS_112 = torch.tensor(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=torch.float32,
)


def _scaled_template(size: Tuple[int, int], device, dtype) -> torch.Tensor:
    height, width = size
    template = CANONICAL_LANDMARKS_112.to(device=device, dtype=dtype).clone()
    template[:, 0] *= float(width) / 112.0
    template[:, 1] *= float(height) / 112.0
    return template


def _similarity_transform(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a 2x3 affine map from target pixels to source pixels.

    ``grid_sample`` needs the inverse map. Solving the map in this direction
    avoids an extra matrix inverse and is numerically stable for the five-point
    configuration used here.
    """

    ones = torch.ones((source.shape[0], 1), device=source.device, dtype=source.dtype)
    design = torch.cat([target, ones], dim=1)
    solution = torch.linalg.lstsq(design, source).solution
    # ``solution`` maps [x_t, y_t, 1] to [x_s, y_s], so the grid needs the
    # full 2x3 affine matrix including translation.
    return solution.T


def _pixel_grid_to_normalized(grid: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
    height, width = input_size
    x = 2.0 * (grid[..., 0] + 0.5) / float(width) - 1.0
    y = 2.0 * (grid[..., 1] + 0.5) / float(height) - 1.0
    return torch.stack([x, y], dim=-1)


def resize_face_crop(crop: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    """Resize a crop the way the training and gallery transforms do.

    Training crops and gallery images go through PIL/torchvision ``Resize``,
    which low-pass filters when shrinking. Plain bilinear interpolation does
    not, so a large detected face (say 400 px) was aliased at inference but
    smooth at training time. ``antialias=True`` closes that train/serve gap;
    it is a no-op when enlarging.
    """
    return F.interpolate(crop, size=output_size, mode="bilinear", align_corners=False, antialias=True)


_resize_crop = resize_face_crop  # backwards-compatible private alias


def align_face_crop(
    crop: torch.Tensor,
    landmarks: Optional[torch.Tensor] = None,
    output_size: Tuple[int, int] = (112, 112),
    min_landmark_span: float = 2.0,
) -> Tuple[torch.Tensor, bool]:
    """Align a crop using five landmarks, or resize it as a safe fallback.

    Args:
        crop: ``(B, C, H, W)`` tensor in any value range.
        landmarks: optional ``(B, 5, 2)`` coordinates in crop pixel space.
        output_size: output ``(height, width)``.
        min_landmark_span: reject degenerate landmark configurations.

    Returns:
        ``(aligned_crop, used_landmarks)``. The boolean is false whenever any
        sample required the box-only fallback. A batch is processed row by row
        because each face has a different transform.
    """

    if crop.ndim != 4:
        raise ValueError(f"crop must have shape (B,C,H,W), got {tuple(crop.shape)}")
    batch, _, input_h, input_w = crop.shape
    target = _scaled_template(output_size, crop.device, crop.dtype)
    outputs = []
    used_all_landmarks = True

    for index in range(batch):
        sample = crop[index : index + 1]
        points = None if landmarks is None else landmarks[index]
        valid = (
            points is not None
            and points.shape == (5, 2)
            and torch.isfinite(points).all()
            and (points[:, 0].max() - points[:, 0].min()) >= min_landmark_span
            and (points[:, 1].max() - points[:, 1].min()) >= min_landmark_span
        )
        if not bool(valid):
            outputs.append(_resize_crop(sample, output_size)[0])
            used_all_landmarks = False
            continue

        inverse_map = _similarity_transform(points.to(crop.dtype), target)
        out_h, out_w = output_size
        y, x = torch.meshgrid(
            torch.arange(out_h, device=crop.device, dtype=crop.dtype),
            torch.arange(out_w, device=crop.device, dtype=crop.dtype),
            indexing="ij",
        )
        target_grid = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        source_grid = target_grid @ inverse_map.T
        normalized_grid = _pixel_grid_to_normalized(source_grid, (input_h, input_w))
        aligned = F.grid_sample(
            sample,
            normalized_grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        outputs.append(aligned[0])

    return torch.stack(outputs, dim=0), used_all_landmarks


def estimate_pose_from_landmarks(landmarks: Optional[torch.Tensor]) -> Dict[str, float]:
    """Estimate a conservative pose signal from five facial landmarks.

    This is intentionally a geometry-based quality signal, not a replacement
    for a calibrated 3-D pose model. Unlike the old aspect-ratio heuristic, it
    depends on the actual facial geometry and returns an explicit validity flag.
    """

    if landmarks is None or tuple(landmarks.shape) != (5, 2) or not torch.isfinite(landmarks).all():
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "valid": 0.0}

    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks.float()
    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (left_mouth + right_mouth) * 0.5
    eye_distance = torch.linalg.vector_norm(right_eye - left_eye).clamp_min(1e-4)
    face_height = torch.linalg.vector_norm(mouth_mid - eye_mid).clamp_min(1e-4)
    yaw = ((nose[0] - eye_mid[0]) / eye_distance) * 70.0
    expected_nose_y = eye_mid[1] + 0.55 * (mouth_mid[1] - eye_mid[1])
    pitch = ((nose[1] - expected_nose_y) / face_height) * 60.0
    roll = math.degrees(math.atan2(float(right_eye[1] - left_eye[1]), float(right_eye[0] - left_eye[0])))
    return {
        "yaw": float(yaw.clamp(-90.0, 90.0).item()),
        "pitch": float(pitch.clamp(-90.0, 90.0).item()),
        "roll": float(max(-90.0, min(90.0, roll))),
        "valid": 1.0,
    }

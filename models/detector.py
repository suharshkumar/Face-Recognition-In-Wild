"""Face detection backends and index-preserving Soft-NMS.

The detector facade intentionally returns an empty result when no trained or
external detector is available. It never fabricates a full-image face and it
never returns landmarks whose row no longer matches the corresponding box.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


def _empty_detection(device: torch.device):
    return (
        torch.empty((0, 4), device=device, dtype=torch.float32),
        torch.empty((0,), device=device, dtype=torch.float32),
        torch.empty((0, 5, 2), device=device, dtype=torch.float32),
    )


def soft_nms_pytorch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
    sigma: float = 0.5,
    score_threshold: float = 0.3,
    return_indices: bool = False,
):
    """Gaussian Soft-NMS that preserves the source row for every detection."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape (N,4)")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("scores must have shape (N,) matching boxes")
    if boxes.numel() == 0:
        empty_boxes = boxes.new_empty((0, 4))
        empty_scores = scores.new_empty((0,))
        empty_indices = torch.empty((0,), dtype=torch.long, device=boxes.device)
        return (empty_boxes, empty_scores, empty_indices) if return_indices else (empty_boxes, empty_scores)

    work_boxes = boxes.detach().clone()
    work_scores = scores.detach().clone()
    source_indices = torch.arange(boxes.shape[0], device=boxes.device)
    n = work_boxes.shape[0]
    for i in range(n):
        best_offset = torch.argmax(work_scores[i:])
        best = i + int(best_offset.item())
        if best != i:
            work_boxes[[i, best]] = work_boxes[[best, i]]
            work_scores[[i, best]] = work_scores[[best, i]]
            source_indices[[i, best]] = source_indices[[best, i]]
        if i + 1 >= n:
            continue
        current = work_boxes[i]
        x1 = torch.maximum(current[0], work_boxes[i + 1 :, 0])
        y1 = torch.maximum(current[1], work_boxes[i + 1 :, 1])
        x2 = torch.minimum(current[2], work_boxes[i + 1 :, 2])
        y2 = torch.minimum(current[3], work_boxes[i + 1 :, 3])
        inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
        area_current = ((current[2] - current[0]).clamp_min(0) * (current[3] - current[1]).clamp_min(0))
        area_other = ((work_boxes[i + 1 :, 2] - work_boxes[i + 1 :, 0]).clamp_min(0) *
                      (work_boxes[i + 1 :, 3] - work_boxes[i + 1 :, 1]).clamp_min(0))
        iou = inter / (area_current + area_other - inter).clamp_min(1e-7)
        decay = torch.exp(-(iou.square()) / max(float(sigma), 1e-7))
        # The threshold is retained as an API parameter for compatibility; the
        # final filter below controls which decayed boxes survive.
        work_scores[i + 1 :] *= decay

    keep = work_scores >= float(score_threshold)
    filtered = (work_boxes[keep], work_scores[keep], source_indices[keep])
    return filtered if return_indices else filtered[:2]


class LNetAnchorFaceLocalizer(nn.Module):
    """Compatibility stub for the old untrained fallback.

    A randomly initialized detector is worse than an explicit empty result, so
    this class only becomes usable when a trained state dictionary is supplied.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.ready = False
        if checkpoint_path:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"LNet checkpoint not found: {checkpoint_path}")
            raise NotImplementedError(
                "The legacy LNet checkpoint format is not supported. Use YuNet, MTCNN, or a face-trained detector."
            )

    def forward(self, img_tensor: torch.Tensor, score_threshold: float = 0.3):
        return _empty_detection(img_tensor.device)


class SOTAFaceDetector(nn.Module):
    """External face detector facade using MTCNN and/or YuNet.

    Both available backends contribute detections; Soft-NMS then merges them.
    TTA is opt-in because it multiplies detector latency and is not a substitute
    for detector training.
    """

    def __init__(self, weights_dir: str = "./weights", enable_tta: bool = False,
                 backend: str = "auto"):
        super().__init__()
        self.enable_tta = bool(enable_tta)
        self.backend = backend
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if backend not in {"auto", "yunet", "mtcnn", "none"}:
            raise ValueError(f"Unknown detector backend: {backend}")

        self.mtcnn = None
        if backend in {"auto", "mtcnn"}:
            try:
                from facenet_pytorch import MTCNN
                self.mtcnn = MTCNN(
                    keep_all=True,
                    post_process=False,
                    min_face_size=12,
                    thresholds=[0.5, 0.6, 0.6],
                    device=self.device,
                )
            except Exception:
                self.mtcnn = None

        self.yunet_path = None
        candidate = os.path.join(weights_dir, "face_detection_yunet_2023mar.onnx")
        if backend in {"auto", "yunet"} and cv2 is not None and np is not None and os.path.isfile(candidate):
            self.yunet_path = candidate

    def _detect_mtcnn(self, image_pil, scale: float, threshold: float, device):
        if self.mtcnn is None:
            return [], [], []
        try:
            boxes, probabilities, landmarks = self.mtcnn.detect(image_pil, landmarks=True)
        except Exception:
            return [], [], []
        result_boxes, result_scores, result_landmarks = [], [], []
        if boxes is None:
            return result_boxes, result_scores, result_landmarks
        for index, box in enumerate(boxes):
            score = float(probabilities[index]) if probabilities is not None else 0.0
            if score < threshold or not np.isfinite(box).all():
                continue
            result_boxes.append(np.asarray(box, dtype=np.float32) / float(scale))
            result_scores.append(score)
            if landmarks is not None and index < len(landmarks):
                result_landmarks.append(np.asarray(landmarks[index], dtype=np.float32) / float(scale))
            else:
                result_landmarks.append(np.zeros((5, 2), dtype=np.float32))
        return result_boxes, result_scores, result_landmarks

    def _detect_yunet(self, image_np, scale: float, threshold: float):
        if self.yunet_path is None:
            return [], [], []
        try:
            height, width = image_np.shape[:2]
            detector = cv2.FaceDetectorYN.create(
                self.yunet_path, "", (width, height),
                score_threshold=threshold, nms_threshold=0.3,
            )
            _, faces = detector.detect(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        except Exception:
            return [], [], []
        boxes, scores, landmarks = [], [], []
        if faces is None:
            return boxes, scores, landmarks
        for face in faces:
            score = float(face[14])
            if score < threshold:
                continue
            x, y, width, height = [float(value) / float(scale) for value in face[:4]]
            boxes.append(np.asarray([x, y, x + width, y + height], dtype=np.float32))
            scores.append(score)
            landmarks.append(np.asarray(face[4:14], dtype=np.float32).reshape(5, 2) / float(scale))
        return boxes, scores, landmarks

    def forward(self, img_tensor: torch.Tensor, score_threshold: float = 0.4,
                enable_tta: Optional[bool] = None):
        if img_tensor.ndim != 4 or img_tensor.shape[0] != 1 or img_tensor.shape[1] != 3:
            raise ValueError("detector input must have shape (1,3,H,W)")
        device = img_tensor.device
        if self.backend == "none":
            return _empty_detection(device)
        if np is None:
            return _empty_detection(device)

        _, _, height, width = img_tensor.shape
        scales = [0.75, 1.0, 1.25] if (self.enable_tta if enable_tta is None else enable_tta) else [1.0]
        boxes_all, scores_all, landmarks_all = [], [], []
        for scale in scales:
            scaled_h = max(32, int(round(height * scale)))
            scaled_w = max(32, int(round(width * scale)))
            current = torch.nn.functional.interpolate(
                img_tensor, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False
            )
            image_np = ((current[0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            try:
                from PIL import Image
                image_pil = Image.fromarray(image_np)
            except Exception:
                image_pil = None

            if image_pil is not None and self.backend in {"auto", "mtcnn"}:
                b, s, l = self._detect_mtcnn(image_pil, scale, score_threshold, device)
                boxes_all.extend(b); scores_all.extend(s); landmarks_all.extend(l)
            if self.backend in {"auto", "yunet"}:
                b, s, l = self._detect_yunet(image_np, scale, score_threshold)
                boxes_all.extend(b); scores_all.extend(s); landmarks_all.extend(l)

        if not boxes_all:
            return _empty_detection(device)
        boxes = torch.tensor(np.asarray(boxes_all), device=device, dtype=torch.float32)
        scores = torch.tensor(np.asarray(scores_all), device=device, dtype=torch.float32).clamp(0.0, 1.0)
        landmarks = torch.tensor(np.asarray(landmarks_all), device=device, dtype=torch.float32)
        boxes[:, 0::2].clamp_(0.0, float(width))
        boxes[:, 1::2].clamp_(0.0, float(height))
        valid = (boxes[:, 2] - boxes[:, 0] >= 4.0) & (boxes[:, 3] - boxes[:, 1] >= 4.0)
        if not valid.any():
            return _empty_detection(device)
        boxes, scores, landmarks = boxes[valid], scores[valid], landmarks[valid]
        boxes, scores, keep = soft_nms_pytorch(
            boxes, scores, score_threshold=score_threshold, return_indices=True
        )
        return boxes, scores, landmarks[keep]


class AnchorFreeFaceDetector(nn.Module):
    """Small pure-PyTorch detector used when torchvision is unavailable.

    It predicts one face box per feature-map cell. This keeps AQUA training
    reproducible on nodes where the optional torchvision C++/CUDA extension is
    not importable; it uses only torch, PIL-loaded tensors, and YOLO targets.
    """

    def __init__(self, channels: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.box_regression = nn.Conv2d(channels, 4, 1)

    def _heads(self, image):
        features = self.backbone(image.unsqueeze(0))
        return self.objectness(features), self.box_regression(features)

    @staticmethod
    def _target_grid(target, height, width, grid_height, grid_width, device):
        objectness = torch.zeros((1, 1, grid_height, grid_width), device=device)
        boxes = torch.zeros((1, 4, grid_height, grid_width), device=device)
        areas = torch.zeros((grid_height, grid_width), device=device)
        for box in target.get("boxes", torch.empty((0, 4), device=device)):
            x1, y1, x2, y2 = box.float()
            if x2 <= x1 or y2 <= y1:
                continue
            center_x = ((x1 + x2) * 0.5 / max(float(width), 1.0)).clamp(0.0, 0.999999)
            center_y = ((y1 + y2) * 0.5 / max(float(height), 1.0)).clamp(0.0, 0.999999)
            grid_x = min(grid_width - 1, int((center_x * grid_width).item()))
            grid_y = min(grid_height - 1, int((center_y * grid_height).item()))
            area = ((x2 - x1) * (y2 - y1)).clamp_min(0.0)
            if area <= areas[grid_y, grid_x]:
                continue
            areas[grid_y, grid_x] = area
            objectness[0, 0, grid_y, grid_x] = 1.0
            boxes[0, :, grid_y, grid_x] = torch.stack([
                center_x,
                center_y,
                ((x2 - x1) / max(float(width), 1.0)).clamp(0.0, 1.0),
                ((y2 - y1) / max(float(height), 1.0)).clamp(0.0, 1.0),
            ])
        return objectness, boxes

    def loss(self, images, targets):
        losses = {"loss_objectness": [], "loss_box_reg": []}
        for image, target in zip(images, targets):
            objectness_logits, box_logits = self._heads(image)
            _, height, width = image.shape
            grid_height, grid_width = objectness_logits.shape[-2:]
            objectness_target, box_target = self._target_grid(
                target, height, width, grid_height, grid_width, image.device
            )
            positive = objectness_target.bool().expand_as(box_target)
            # The face cells are sparse, so a modest positive weighting keeps
            # the detector from learning an all-background solution.
            objectness_loss = F.binary_cross_entropy_with_logits(
                objectness_logits,
                objectness_target,
                pos_weight=torch.tensor(4.0, device=image.device),
            )
            predicted_boxes = torch.sigmoid(box_logits)
            if positive.any():
                box_loss = F.smooth_l1_loss(predicted_boxes[positive], box_target[positive])
            else:
                box_loss = box_logits.sum() * 0.0
            losses["loss_objectness"].append(objectness_loss)
            losses["loss_box_reg"].append(box_loss)
        return {name: torch.stack(values).mean() for name, values in losses.items()}

    def forward(self, images, targets=None):
        """Support both ordinary inference and DDP's module forward path."""
        if isinstance(images, (list, tuple)):
            if targets is not None:
                return self.loss(images, targets)
            return [self.predict(image) for image in images]
        if targets is not None:
            return self.loss([images], [targets])
        return self.predict(images)

    @torch.no_grad()
    def predict(self, image, score_threshold=0.4, max_detections=200):
        objectness_logits, box_logits = self._heads(image)
        _, height, width = image.shape
        grid_height, grid_width = objectness_logits.shape[-2:]
        scores = torch.sigmoid(objectness_logits[0, 0]).flatten()
        count = min(max_detections, scores.numel())
        scores, indices = torch.topk(scores, count)
        keep = scores >= float(score_threshold)
        if not keep.any():
            return _empty_detection(image.device)
        scores = scores[keep]
        indices = indices[keep]
        grid_y = torch.div(indices, grid_width, rounding_mode="floor")
        grid_x = indices % grid_width
        values = torch.sigmoid(box_logits[0, :, grid_y, grid_x]).T
        center_x, center_y, box_width, box_height = values.unbind(dim=1)
        center_x = center_x * float(width)
        center_y = center_y * float(height)
        box_width = box_width * float(width)
        box_height = box_height * float(height)
        boxes = torch.stack([
            center_x - box_width * 0.5,
            center_y - box_height * 0.5,
            center_x + box_width * 0.5,
            center_y + box_height * 0.5,
        ], dim=1)
        boxes[:, 0::2].clamp_(0.0, float(width))
        boxes[:, 1::2].clamp_(0.0, float(height))
        valid = (boxes[:, 2] - boxes[:, 0] >= 4.0) & (boxes[:, 3] - boxes[:, 1] >= 4.0)
        if not valid.any():
            return _empty_detection(image.device)
        boxes, scores = boxes[valid], scores[valid]
        boxes, scores, keep_indices = soft_nms_pytorch(
            boxes, scores, score_threshold=score_threshold, return_indices=True
        )
        landmarks = torch.zeros((boxes.shape[0], 5, 2), device=image.device, dtype=boxes.dtype)
        return boxes, scores, landmarks


class TorchvisionFaceDetector(nn.Module):
    """Trainable class-agnostic Faster R-CNN backed by AQUA YOLO boxes.

    Every foreground YOLO row is treated as a face (class 1); this avoids
    assuming that the mask-condition class IDs are identical across AQUA
    subsets. Landmark coordinates are unavailable from this detector and are
    represented by zeros, so the pipeline uses its box-resize alignment
    fallback unless an external landmark backend is also enabled.
    """

    def __init__(self, num_classes: int = 2, checkpoint_path: Optional[str] = None,
                 pretrained_backbone: bool = True, backend: Optional[str] = None,
                 detector_min_size: int = 800, detector_max_size: int = 1333):
        super().__init__()
        checkpoint = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        requested_backend = backend or os.environ.get("AQUA_TRAIN_DETECTOR_BACKEND", "auto")
        if checkpoint and checkpoint.get("backend"):
            requested_backend = checkpoint["backend"]
        self.backend = requested_backend
        self.architecture = "aqua_anchor_free"
        self.model = None
        if requested_backend not in {"auto", "torchvision", "anchor_free", "aqua_anchor_free"}:
            raise ValueError(f"Unknown training detector backend: {requested_backend}")
        if requested_backend in {"anchor_free", "aqua_anchor_free"}:
            self.backend = "anchor_free"
            self.architecture = "aqua_anchor_free"
            self.model = AnchorFreeFaceDetector()
        else:
            try:
                from torchvision.models.detection import (
                    FasterRCNN_ResNet50_FPN_Weights,
                    fasterrcnn_resnet50_fpn,
                )
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                from torchvision.models.detection.rpn import AnchorGenerator
                if detector_min_size <= 0 or detector_max_size < detector_min_size:
                    raise ValueError("detector_max_size must be >= detector_min_size > 0")
                # WIDER-style face boxes are often smaller than generic COCO
                # objects, so use smaller RPN anchors across the five FPN levels.
                anchor_generator = AnchorGenerator(
                    sizes=((16,), (32,), (64,), (128,), (256,)),
                    aspect_ratios=((0.5, 1.0, 2.0),) * 5,
                )
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone and not checkpoint_path else None
                try:
                    model = fasterrcnn_resnet50_fpn(
                        weights=weights,
                        weights_backbone=None,
                        min_size=detector_min_size,
                        max_size=detector_max_size,
                        rpn_anchor_generator=anchor_generator,
                        rpn_pre_nms_top_n_train=2000,
                        rpn_post_nms_top_n_train=1000,
                        rpn_pre_nms_top_n_test=1000,
                        rpn_post_nms_top_n_test=300,
                    )
                except Exception:
                    # Keep the requested architecture if the COCO weights are
                    # unavailable on an offline node.
                    model = fasterrcnn_resnet50_fpn(
                        weights=None,
                        weights_backbone=None,
                        min_size=detector_min_size,
                        max_size=detector_max_size,
                        rpn_anchor_generator=anchor_generator,
                        rpn_pre_nms_top_n_train=2000,
                        rpn_post_nms_top_n_train=1000,
                        rpn_pre_nms_top_n_test=1000,
                        rpn_post_nms_top_n_test=300,
                    )
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                self.model = model
                self.backend = "torchvision"
                self.architecture = "fasterrcnn_resnet50_fpn"
            except Exception as exc:
                if requested_backend == "torchvision":
                    raise ImportError("Requested torchvision detector is unavailable") from exc
                self.backend = "anchor_free"
                self.architecture = "aqua_anchor_free"
                self.model = AnchorFreeFaceDetector()
        self.num_classes = num_classes
        self.loaded_checkpoint = False
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Face detector checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("schema_version") != 2:
            raise ValueError(f"Unsupported face detector checkpoint schema: {checkpoint_path}")
        if checkpoint.get("kind") != "face_detector":
            raise ValueError(f"Not a face detector checkpoint: {checkpoint_path}")
        saved_classes = int(checkpoint.get("num_classes", self.num_classes))
        if saved_classes != self.num_classes:
            raise ValueError(f"Detector class mismatch: checkpoint={saved_classes}, model={self.num_classes}")
        state = checkpoint.get("model_state", checkpoint.get("state_dict"))
        if state is None:
            raise ValueError(f"Detector checkpoint has no model state: {checkpoint_path}")
        self.model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
        self.loaded_checkpoint = True

    def detection_loss(self, images, targets):
        """Return torchvision's training loss dictionary."""
        self.model.train()
        if self.backend == "anchor_free":
            # Call through DDP rather than reaching into ``.module``. DDP
            # registers gradient hooks around its forward call; bypassing it
            # would run the loss but silently skip cross-GPU synchronization.
            return self.model(images, targets)
        return self.model(images, targets)

    @torch.no_grad()
    def forward(self, img_tensor: torch.Tensor, score_threshold: float = 0.4):
        if img_tensor.ndim != 4 or img_tensor.shape[0] != 1 or img_tensor.shape[1] != 3:
            raise ValueError("detector input must have shape (1,3,H,W)")
        device = img_tensor.device
        if self.backend == "anchor_free":
            self.model.eval()
            return self.model.predict(img_tensor[0], score_threshold=score_threshold)
        _, _, height, width = img_tensor.shape
        image = ((img_tensor[0] + 1.0) / 2.0).clamp(0.0, 1.0)
        self.model.eval()
        prediction = self.model([image])[0]
        labels = prediction["labels"]
        keep = (labels >= 1) & (labels < self.num_classes) & (prediction["scores"] >= score_threshold)
        boxes = prediction["boxes"][keep].to(device)
        scores = prediction["scores"][keep].to(device)
        landmarks = torch.zeros((boxes.shape[0], 5, 2), device=device, dtype=boxes.dtype)
        if boxes.numel() == 0:
            return _empty_detection(device)
        boxes[:, 0::2].clamp_(0.0, float(width))
        boxes[:, 1::2].clamp_(0.0, float(height))
        boxes, scores, kept = soft_nms_pytorch(
            boxes, scores, score_threshold=score_threshold, return_indices=True
        )
        return boxes, scores, landmarks[kept]


class FasterRCNNFaceDetector(nn.Module):
    """Backward-compatible facade for the configured face detector.

    Uses the trained AQUA ``face_detector.pt`` artifact when present. If no
    trained artifact is available, it falls back to the external YuNet/MTCNN
    facade rather than treating generic COCO ``person`` detections as faces.
    """

    def __init__(self, weights_dir: str = "./weights", backend: str = "auto", enable_tta: bool = False, **kwargs):
        super().__init__()
        detector_checkpoint = kwargs.get("detector_checkpoint") or os.path.join(weights_dir, "face_detector.pt")
        self.trained_detector = None
        if detector_checkpoint and os.path.isfile(detector_checkpoint):
            self.trained_detector = TorchvisionFaceDetector(checkpoint_path=detector_checkpoint)
        self.sota_refiner = SOTAFaceDetector(
            weights_dir=weights_dir,
            backend=backend,
            enable_tta=enable_tta,
        )

    def load_trained_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None) -> None:
        """Replace the external fallback with a trained AQUA detector."""
        self.trained_detector = TorchvisionFaceDetector(checkpoint_path=checkpoint_path)
        if device is not None:
            self.trained_detector.to(device)

    def forward(self, img_tensor: torch.Tensor, score_threshold: float = 0.4):
        if self.trained_detector is not None:
            return self.trained_detector.to(img_tensor.device)(img_tensor, score_threshold=score_threshold)
        return self.sota_refiner(img_tensor, score_threshold=score_threshold)


LNetFaceLocalizer = LNetAnchorFaceLocalizer
WildFaceDetector = FasterRCNNFaceDetector

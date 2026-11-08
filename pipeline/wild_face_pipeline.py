"""End-to-end detection, alignment, embedding, enrollment, and open-set flow."""

import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image

from dataset import NameLabeledFaceDataset, get_default_transform
from models.anet_attribute import ANetAttributeParser
from models.backbone import FasterRCNNBackbone, ResNet100Backbone, vit_face_base
from models.ddrc_solver import DEFAULT_MARGIN_THRESHOLD, DEFAULT_RESIDUAL_THRESHOLD, DDRCClassifier
from models.detector import WildFaceDetector
from models.pim_frontalizer import PIMFrontalizationGAN
from models.sam_segmentor import SAMFaceSegmentor
from pipeline.alignment import align_face_crop, estimate_pose_from_landmarks, resize_face_crop


class OccuPoseBroadDictPipeline(nn.Module):
    """Validated inference pipeline for wild face recognition.

    Optional mask, attribute, and frontalization modules are disabled unless a
    compatible checkpoint is loaded. This prevents random preprocessing layers
    from corrupting otherwise useful face embeddings.
    """

    def __init__(
        self,
        num_enrolled_classes: int = 0,
        feature_dim: int = 512,
        backbone_type: str = "iresnet100",
        detector_backend: str = "auto",
        detector_weights_dir: str = "./weights",
        detector_checkpoint: Optional[str] = None,
        recognition_alignment: str = "resize",
        pose_gate_degrees: float = 35.0,
        use_frontalizer: bool = False,
        use_attributes: bool = False,
        use_foreground_mask: bool = False,
        atoms_per_class: int = 1,
        residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        if recognition_alignment not in {"resize", "landmarks"}:
            raise ValueError("recognition_alignment must be 'resize' or 'landmarks'")
        self.recognition_alignment = recognition_alignment
        self.pose_gate_degrees = float(pose_gate_degrees)
        self.use_frontalizer = bool(use_frontalizer)
        self.use_attributes = bool(use_attributes)
        self.use_foreground_mask = bool(use_foreground_mask)
        self._attributes_requested = bool(use_attributes)
        self._foreground_requested = bool(use_foreground_mask)
        self._frontalizer_requested = bool(use_frontalizer)
        self.detector = WildFaceDetector(
            weights_dir=detector_weights_dir,
            backend=detector_backend,
            detector_checkpoint=detector_checkpoint,
        )
        if backbone_type == "vit_face_base":
            self.backbone = vit_face_base(embedding_dim=feature_dim)
        elif backbone_type == "iresnet100":
            self.backbone = ResNet100Backbone(embedding_dim=feature_dim)
        elif backbone_type == "fasterrcnn_backbone":
            self.backbone = FasterRCNNBackbone(embedding_dim=feature_dim)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        self.backbone_type = backbone_type
        # Optional modules are instantiated only when a compatible checkpoint
        # is loaded. Constructing them here would silently enable random
        # preprocessing weights.
        self.attribute_parser = None
        self.sam_segmentor = None
        self.frontalizer = None
        self.ddrc_classifier = DDRCClassifier(
            feature_dim=feature_dim,
            # Class count is created only by a real enrollment artifact. Never
            # allocate random identity atoms merely because a caller supplied a
            # nominal class count.
            num_classes=0,
            atoms_per_class=atoms_per_class,
            residual_threshold=residual_threshold,
            margin_threshold=margin_threshold,
        )
        self.gcn_predictor = None
        self.has_trained_backbone = False
        self.has_calibrated_gallery = False

    @staticmethod
    def _load_state(module: nn.Module, state: Dict[str, torch.Tensor], name: str) -> None:
        if not state:
            raise ValueError(f"Checkpoint contains no state for {name}")
        cleaned = {key.removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = module.load_state_dict(cleaned, strict=False)
        if missing:
            raise ValueError(f"Incompatible {name} checkpoint; missing keys include {missing[:3]}")
        if unexpected:
            print(f"Warning: ignored unexpected {name} keys: {unexpected[:3]}")

    def load_pretrained_weights(self, checkpoint_dir: str = "./weights", device: torch.device = None):
        """Load the available checkpoint bundle without silently using random modules."""
        if device is None:
            device = next(self.parameters()).device
        self.to(device)
        loaded = False
        # Phase 0 artifacts may live directly in checkpoint_dir or nested in
        # the organized phase0_detector/ subfolder.  Search both locations so
        # inference works regardless of how the checkpoint bundle was saved.
        detector_search_dirs = [checkpoint_dir, os.path.join(checkpoint_dir, "phase0_detector")]
        detector_path = next(
            (path for path in (os.path.join(d, "face_detector.pt") for d in detector_search_dirs)
             if os.path.isfile(path)),
            None,
        )
        if detector_path and hasattr(self.detector, "load_trained_checkpoint"):
            self.detector.load_trained_checkpoint(detector_path, device=device)
            print(f"Loaded AQUA face detector from {detector_path}")
        p1_candidates = [
            os.path.join(checkpoint_dir, "phase3_adapted_backbone.pt"),
            os.path.join(checkpoint_dir, "recognition_backbone.pt"),
            os.path.join(checkpoint_dir, "latest_backbone.pt"),
            os.path.join(checkpoint_dir, "phase1_backbone_curricular.pt"),
            os.path.join(checkpoint_dir, "phase1_epoch_100.pt"),
            os.path.join(os.path.join(checkpoint_dir, "phase0_detector"), "recognition_backbone.pt"),
        ]
        for path in p1_candidates:
            if not os.path.isfile(path):
                continue
            checkpoint = torch.load(path, map_location=device)
            if checkpoint.get("schema_version") != 2:
                raise ValueError(f"Unsupported checkpoint schema in {path}: {checkpoint.get('schema_version')}")
            if checkpoint.get("kind") != "recognition_backbone":
                raise ValueError(f"Checkpoint is not a recognition backbone artifact: {path}")
            if checkpoint.get("input_size") not in (None, [112, 112], (112, 112)):
                raise ValueError(f"Recognition checkpoint requires input_size=(112,112): {path}")
            saved_config = checkpoint.get("model_config", {})
            if not saved_config:
                raise ValueError(f"Recognition checkpoint has no model_config metadata: {path}")
            saved_alignment = checkpoint.get("recognition_alignment", saved_config.get("recognition_alignment", self.recognition_alignment))
            if saved_alignment not in {"resize", "landmarks"}:
                raise ValueError(f"Unsupported recognition alignment mode in {path}: {saved_alignment}")
            self.recognition_alignment = saved_alignment
            saved_backbone_type = saved_config.get("backbone_type", self.backbone_type)
            saved_embedding_dim = int(saved_config.get("embedding_dim", self.feature_dim))
            if saved_backbone_type != self.backbone_type or saved_embedding_dim != self.feature_dim:
                if saved_backbone_type == "vit_face_base":
                    self.backbone = vit_face_base(embedding_dim=saved_embedding_dim).to(device)
                elif saved_backbone_type == "fasterrcnn_backbone":
                    self.backbone = FasterRCNNBackbone(embedding_dim=saved_embedding_dim).to(device)
                elif saved_backbone_type == "iresnet100":
                    self.backbone = ResNet100Backbone(embedding_dim=saved_embedding_dim).to(device)
                else:
                    raise ValueError(f"Unsupported backbone in checkpoint: {saved_backbone_type}")
                self.backbone_type = saved_backbone_type
                self.feature_dim = saved_embedding_dim
                # Rebuild for the checkpoint's embedding size but keep the
                # caller's gallery/open-set configuration.
                previous = self.ddrc_classifier
                self.ddrc_classifier = DDRCClassifier(
                    feature_dim=saved_embedding_dim,
                    num_classes=0,
                    atoms_per_class=previous.atoms_per_class,
                    residual_threshold=previous.residual_threshold,
                    margin_threshold=previous.margin_threshold,
                ).to(device)
            state = checkpoint.get("backbone_state", checkpoint.get("backbone"))
            self._load_state(self.backbone, state, "backbone")
            self.has_trained_backbone = True
            loaded = True
            print(f"Loaded recognition backbone from {path}")
            break

        p2_search_dirs = [checkpoint_dir, os.path.join(checkpoint_dir, "phase0_detector")]
        p2_path = next(
            (path for path in (os.path.join(d, "phase2_anet_attributes.pt") for d in p2_search_dirs)
             if os.path.isfile(path)),
            None,
        )
        if p2_path:
            checkpoint = torch.load(p2_path, map_location=device)
            if self.attribute_parser is None:
                self.attribute_parser = ANetAttributeParser().to(device)
            self._load_state(self.attribute_parser, checkpoint.get("anet"), "attribute parser")
            print(f"Loaded attribute parser from {p2_path}")

            sam_state = checkpoint.get("sam")
            if sam_state is not None:
                if self.sam_segmentor is None:
                    self.sam_segmentor = SAMFaceSegmentor().to(device)
                self._load_state(self.sam_segmentor, sam_state, "foreground segmentor")

            frontalizer_state = checkpoint.get("frontalizer")
            if frontalizer_state is not None:
                if self.frontalizer is None:
                    self.frontalizer = PIMFrontalizationGAN().to(device)
                self._load_state(self.frontalizer, frontalizer_state, "frontalizer")

        p3_path = os.path.join(checkpoint_dir, "phase3_semi_gcn_predictor.pt")
        if os.path.isfile(p3_path):
            from models.gcn_cluster import GCNLinkPredictor
            checkpoint = torch.load(p3_path, map_location=device)
            if checkpoint.get("schema_version") != 2 or checkpoint.get("kind") != "gcn_adapter":
                raise ValueError(f"Unsupported GCN adapter checkpoint: {p3_path}")
            if checkpoint.get("input_size") not in ([112, 112], (112, 112)):
                raise ValueError(f"GCN adapter requires input_size=(112,112): {p3_path}")
            saved_config = checkpoint.get("model_config", {})
            saved_dim = int(saved_config.get("embedding_dim", self.feature_dim))
            if saved_dim != self.feature_dim:
                raise ValueError(
                    f"GCN adapter embedding dimension {saved_dim} does not match backbone {self.feature_dim}"
                )
            gcn_state = checkpoint.get("gcn_predictor", checkpoint.get("state_dict"))
            if gcn_state:
                self.gcn_predictor = GCNLinkPredictor(feature_dim=self.feature_dim).to(device)
                self._load_state(self.gcn_predictor, gcn_state, "GCN predictor")
                print(f"Loaded GCN adapter from {p3_path}")
            else:
                # The cached Phase 3 fast path stores pseudo-label metadata
                # without a serialised predictor.  The adapted backbone is
                # still a valid inference artifact; leave the optional graph
                # module disabled instead of rejecting the whole checkpoint.
                print(f"Loaded Phase 3 adaptation metadata from {p3_path}; no graph predictor state was saved")
        return loaded

    def _prepare_crop(self, crop: torch.Tensor, landmarks: Optional[torch.Tensor] = None):
        if self._foreground_requested and self.sam_segmentor is None:
            raise RuntimeError("Foreground preprocessing was requested but no compatible checkpoint is loaded")
        alignment_landmarks = None
        if self.recognition_alignment == "landmarks":
            alignment_landmarks = landmarks.unsqueeze(0) if landmarks is not None and landmarks.ndim == 2 else landmarks
        aligned, used_landmarks = align_face_crop(crop, alignment_landmarks, output_size=(112, 112))
        if self.sam_segmentor is not None:
            box_prompt = torch.tensor([[0.0, 0.0, 112.0, 112.0]], device=aligned.device).expand(aligned.shape[0], -1)
            mask = self.sam_segmentor(aligned, box_prompt=box_prompt)
            # A soft residual keeps the foreground model from erasing the face
            # when a mask prediction is uncertain.
            aligned = aligned * (0.35 + 0.65 * mask)
        return aligned, used_landmarks

    def _embed_crop(self, crop: torch.Tensor, landmarks: Optional[torch.Tensor] = None):
        processed, aligned = self._prepare_crop(crop, landmarks)
        pose = estimate_pose_from_landmarks(landmarks)
        used_frontalizer = False
        if self._frontalizer_requested and self.frontalizer is None:
            raise RuntimeError("Frontalization was requested but no compatible checkpoint is loaded")
        if self.frontalizer is not None and pose["valid"] and abs(pose["yaw"]) >= self.pose_gate_degrees:
            processed = self.frontalizer(processed, yaw_angle=pose["yaw"])
            used_frontalizer = True
        embedding = self.backbone(processed)
        return embedding, pose, aligned, used_frontalizer, processed

    @staticmethod
    def display_name(folder_identity: str) -> str:
        """Gallery folder identity -> the name reported in predictions."""
        return folder_identity.replace("_", " ").title()

    @staticmethod
    def _gallery_cache_signature(samples):
        signature = []
        for path, label in samples:
            try:
                stat = os.stat(path)
                signature.append((path, int(label), stat.st_size, stat.st_mtime_ns))
            except OSError:
                signature.append((path, int(label), None, None))
        return signature

    def _enroll_embeddings(self, embedding_tensor, label_tensor, names):
        self.ddrc_classifier.enroll_gallery(embedding_tensor, label_tensor, names)
        self.has_calibrated_gallery = False
        return len(names)

    @torch.no_grad()
    def enroll_dataset(self, name_label_dir: str, device: torch.device = None, cache_path: str = None):
        """Enroll face crops, reusing a cache when the gallery is unchanged."""
        if device is None:
            device = next(self.parameters()).device
        self.to(device)
        dataset = NameLabeledFaceDataset(
            name_label_dir=name_label_dir,
            transform=get_default_transform((112, 112), is_train=False),
            is_train=False,
        )
        signature = self._gallery_cache_signature(dataset.samples)
        if cache_path and os.path.isfile(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if cached.get("schema_version") == 1 and cached.get("gallery_signature") == signature:
                    embedding_tensor = cached["embeddings"].to(device)
                    label_tensor = cached["labels"].to(device)
                    cached_names = cached.get("names")
                    if cached_names:
                        print(f"Loaded cached gallery embeddings from {cache_path} ({embedding_tensor.shape[0]} samples)", flush=True)
                        return self._enroll_embeddings(embedding_tensor, label_tensor, cached_names)
            except Exception as exc:
                print(f"Warning: could not load gallery embedding cache; recomputing: {exc}", flush=True)

        embeddings, labels = [], []
        skipped = 0
        self.eval()
        total_samples = len(dataset.samples)
        for sample_index, (path, label) in enumerate(dataset.samples, start=1):
            try:
                image = Image.open(path).convert("RGB")
                tensor = get_default_transform((112, 112), is_train=False)(image).unsqueeze(0).to(device)
                embedding, _, _, _, _ = self._embed_crop(tensor, None)
                embeddings.append(embedding.squeeze(0).cpu())
                labels.append(label)
                if sample_index == 1 or sample_index % 25 == 0 or sample_index == total_samples:
                    print(f"Gallery embedding progress: {sample_index}/{total_samples}", flush=True)
            except Exception as exc:
                skipped += 1
                print(f"Warning: skipping unreadable gallery image {path}: {exc}")
        if not embeddings:
            raise RuntimeError(f"No readable gallery images found under {name_label_dir}")
        if skipped:
            print(f"Skipped {skipped} unreadable gallery images during enrollment")
        embedding_tensor = torch.stack(embeddings)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        label_to_name = {
            int(index): self.display_name(name)
            for name, index in dataset.class_to_idx.items()
        }
        unique_labels = sorted(set(labels))
        names = [label_to_name[label] for label in unique_labels]
        label_remap = {label: index for index, label in enumerate(unique_labels)}
        label_tensor = torch.tensor([label_remap[int(label)] for label in labels], dtype=torch.long)
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save({
                    "schema_version": 1,
                    "gallery_signature": signature,
                    "names": names,
                    "embeddings": embedding_tensor.cpu(),
                    "labels": label_tensor.cpu(),
                }, cache_path)
                print(f"Saved gallery embedding cache to {cache_path}")
            except Exception as exc:
                print(f"Warning: could not save gallery embedding cache: {exc}")
        return self._enroll_embeddings(embedding_tensor.to(device), label_tensor.to(device), names)

    def forward(self, img_tensor: torch.Tensor, score_threshold: float = 0.4) -> List[Dict[str, Any]]:
        if img_tensor.ndim != 4 or img_tensor.shape[0] != 1:
            raise ValueError("pipeline input must have shape (1,3,H,W)")
        self.eval()
        with torch.no_grad():
            boxes, det_scores, landmarks = self.detector(img_tensor, score_threshold=score_threshold)
            if boxes.shape[0] == 0:
                return []
            img_h, img_w = img_tensor.shape[2:]
            results = []
            for index in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[index].round().long().tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                crop = img_tensor[:, :, y1:y2, x1:x2]
                local_landmarks = None
                if landmarks.shape[0] > index:
                    candidate = landmarks[index].clone()
                    if torch.isfinite(candidate).all() and candidate.abs().sum() > 0:
                        candidate[:, 0] -= float(x1)
                        candidate[:, 1] -= float(y1)
                        local_landmarks = candidate
                embedding, pose, aligned, used_frontalizer, _ = self._embed_crop(crop, local_landmarks)
                is_occluded = False
                attributes = {}
                if self._attributes_requested and self.attribute_parser is None:
                    raise RuntimeError("Attribute preprocessing was requested but no compatible checkpoint is loaded")
                if self.attribute_parser is not None:
                    attribute_landmarks = None
                    if local_landmarks is not None:
                        attribute_landmarks = local_landmarks.clone()
                        attribute_landmarks[:, 0] *= 112.0 / max(float(x2 - x1), 1.0)
                        attribute_landmarks[:, 1] *= 112.0 / max(float(y2 - y1), 1.0)
                    attr_logits, occ_mask, occluded = self.attribute_parser(
                        resize_face_crop(crop, (112, 112)),
                        attribute_landmarks.unsqueeze(0) if attribute_landmarks is not None else None,
                    )
                    is_occluded = bool(occluded[0].item())
                    probabilities = torch.sigmoid(attr_logits[0])
                    attributes = {
                        ANetAttributeParser.ATTRIBUTE_NAMES[i]: bool(probabilities[i] >= 0.5)
                        for i in range(min(len(ANetAttributeParser.ATTRIBUTE_NAMES), probabilities.numel()))
                    }
                predictions, recognition_scores, residuals = self.ddrc_classifier(embedding)
                identity = predictions[0]
                recognition_score = float(recognition_scores[0].item())
                detector_score = float(det_scores[index].item())
                results.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "identity": identity,
                    "is_unknown": identity == "unknown",
                    "detector_score": detector_score,
                    "recognition_score": recognition_score,
                    "confidence": round(detector_score * recognition_score, 6),
                    "is_occluded": is_occluded,
                    "attributes": attributes,
                    "pose": pose,
                    "aligned": bool(aligned),
                    "frontalized": bool(used_frontalizer),
                    "residuals": residuals[0].detach().cpu().tolist(),
                })
            return sorted(results, key=lambda result: result["confidence"], reverse=True)

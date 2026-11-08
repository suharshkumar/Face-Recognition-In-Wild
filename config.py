from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class DatasetConfig:
    train_data_dir: str = "" # Identity-labelled, face-crop dataset
    bb_data_dir: str = "" # YOLO/Darknet face detection dataset (AQUA bb_label)
    unlabeled_data_dir: str = "" # Optional unlabeled crops/scenes; special WIDER auto-discovered
    val_data_dir: str = ""
    celeba_dir: str = ""
    image_size: Tuple[int, int] = (112, 112)
    num_classes: int = 0 # Derived from the labelled manifest; never silently guessed
    num_workers: int = 4
    allow_synthetic_fixtures: bool = False
    unlabeled_mode: str = "detected_scenes" # detected_scenes or face_crops
    unlabeled_score_threshold: float = 0.75
    wider_identity_data_dir: str = "" # Optional nested WIDER folder-label crops

@dataclass
class ModelConfig:
    backbone_type: str = "iresnet100" # "iresnet100" or "vit_face_base"
    embedding_dim: int = 512
    fp16: bool = True
    detector_backend: str = "auto" # auto, yunet, mtcnn, none
    detector_score_threshold: float = 0.40
    recognition_alignment: str = "resize" # resize or landmarks
    use_foreground_mask: bool = False
    use_pim_frontalizer: bool = False
    use_anet_attributes: bool = False
    use_gcn_clustering: bool = False
    pose_gate_degrees: float = 35.0
    lista_num_layers: int = 5
    atoms_per_class: int = 20

@dataclass
class LossConfig:
    scale: float = 64.0
    margin: float = 0.50
    ema_alpha: float = 0.99
    queue_size: int = 32768
    broadface_momentum: float = 0.99
    lambda_semi: float = 0.5 # Weight for semi-supervised pseudo-labeled loss
    pseudo_label_threshold: float = 0.75 # Confidence threshold for GCN graph clustering
    unknown_similarity_threshold: float = 0.45
    unknown_margin_threshold: float = 0.05

@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 2
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 50
    device: str = "cuda"
    seed: int = 42
    detector_epochs: int = 10
    detector_batch_size: int = 2
    detector_learning_rate: float = 0.005
    detector_pretrained_backbone: bool = True
    detector_min_size: int = 800
    detector_max_size: int = 1333
    detector_split: str = "all" # all, or train for a held-out evaluation split

@dataclass
class SystemConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

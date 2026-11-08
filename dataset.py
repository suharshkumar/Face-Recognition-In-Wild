import os
import re
import glob
import hashlib
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torch
import torch.utils.data as data

# Torchvision's optional C++ extension is not available on every AQUA node.
# The pure-PyTorch/PIL transform is deterministic and sufficient for this
# project, so do not import torchvision unless explicitly requested.
if os.environ.get("FACE_USE_TORCHVISION_TRANSFORMS", "0") == "1":
    try:
        import torchvision.transforms as transforms
    except Exception:
        transforms = None
else:
    transforms = None

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
DETECTION_SPLIT_NAMES = frozenset({'train', 'test', 'val', 'validate', 'validation'})
TASK_DIRECTORY_NAMES = frozenset({'bb_label', 'name_label', 'unlabeled', 'celeba'})
WIDER_SPECIAL_DIRECTORY_NAME = 'masked-face-detection-wider-dataset'


class DatasetConfigurationError(ValueError):
    """Raised when a production dataset cannot provide valid samples."""


def _require_dataset_root(root_dir: str, name: str) -> str:
    if not root_dir:
        raise FileNotFoundError(
            f"{name} is not configured. Pass an explicit dataset directory; "
            "synthetic fixtures are disabled in production."
        )
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {root_dir}")
    return root_dir


def resolve_task_root(root_dir: str, task_directory: str, task_name: str) -> str:
    """Resolve an AQUA task root without mixing sibling datasets."""
    if not root_dir:
        return root_dir
    normalized = os.path.normpath(root_dir)
    basename = os.path.basename(normalized).lower()
    expected = task_directory.lower()
    if basename in TASK_DIRECTORY_NAMES and basename != expected:
        raise DatasetConfigurationError(
            f"{task_name} received the {basename}/ directory; expected {expected}/"
        )
    child = os.path.join(root_dir, task_directory)
    if os.path.isdir(child):
        return child
    if os.path.isdir(root_dir):
        sibling_tasks = [
            name for name in TASK_DIRECTORY_NAMES
            if os.path.isdir(os.path.join(root_dir, name))
        ]
        if sibling_tasks and basename != expected:
            raise DatasetConfigurationError(
                f"{task_name} root {root_dir} contains task directories {sorted(sibling_tasks)} "
                f"but no {task_directory}/ child. Pass the task-specific directory explicitly."
            )
    return root_dir


def find_wider_special_root(root_dir: str):
    """Find AQUA's nested WIDER mask dataset without scanning other sources."""
    if not root_dir or not os.path.isdir(root_dir):
        return None
    normalized = os.path.normpath(root_dir)
    if os.path.basename(normalized).lower() == WIDER_SPECIAL_DIRECTORY_NAME:
        return normalized
    direct = os.path.join(normalized, WIDER_SPECIAL_DIRECTORY_NAME)
    if os.path.isdir(direct):
        return direct
    for current_root, directories, _ in os.walk(normalized):
        directories.sort()
        for directory in directories:
            if directory.lower() == WIDER_SPECIAL_DIRECTORY_NAME:
                return os.path.join(current_root, directory)
    return None


def wider_folder_identity_name(folder_name: str) -> str:
    """Convert ``0--Parade`` into the requested generic label ``Parade``."""
    cleaned = re.sub(r'[^A-Za-z]+', ' ', folder_name)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        raise DatasetConfigurationError(
            f"WIDER identity folder has no alphabetic label: {folder_name!r}"
        )
    return cleaned


def canonical_identity_name(name: str) -> str:
    """Normalize dataset condition prefixes without making conditions identities."""
    normalized = name.strip()
    normalized = re.sub(r'^(ROF_|WILD_|MASK_)', '', normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r'(?:_wearing_(?:mask|sunglasses|hat)|_(?:mask|sunglasses|hat)|'
        r'_(?:with_mask|masked))$',
        '', normalized, flags=re.IGNORECASE,
    )
    return normalized or name.strip()


def is_readable_image(path: str) -> bool:
    """Return whether an uploaded image file has a non-empty payload.

    Dataset uploads can leave zero-byte files behind. Do this cheap check
    during indexing instead of opening every image over AQUA's high-latency
    filesystem; PIL remains the authoritative decoder in ``__getitem__``.
    """
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False

class PurePyTorchImageTransform:
    """
    Fallback Image Transform using pure PyTorch & PIL.
    Requires ZERO external C-libraries.
    """
    def __init__(self, image_size=(112, 112), is_train=True):
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, img: Image.Image) -> torch.Tensor:
        height, width = self.image_size
        if img.size != (width, height):
            img = img.resize((width, height), Image.BILINEAR)
            
        if self.is_train and torch.rand(1).item() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
        img_bytes = bytearray(img.tobytes())
        tensor = torch.frombuffer(img_bytes, dtype=torch.uint8)
        tensor = tensor.view(height, width, 3).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor

def get_default_transform(image_size=(112, 112), is_train=True):
    if transforms is not None:
        try:
            if is_train:
                return transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        except Exception:
            pass
    return PurePyTorchImageTransform(image_size=image_size, is_train=is_train)


def pil_to_float_tensor(img: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image to a detector tensor in [0, 1] without resizing."""
    img = img.convert("RGB")
    width, height = img.size
    raw = bytearray(img.tobytes())
    return torch.frombuffer(raw, dtype=torch.uint8).view(height, width, 3).permute(2, 0, 1).float() / 255.0

def parse_yolo_txt(txt_path: str):
    """
    Parses YOLO format bounding box annotation file (.txt).
    """
    bboxes = []
    try:
        # Open directly instead of doing an existence check first. AQUA's
        # dataset is on a high-latency filesystem, where the extra stat call
        # for every image makes a 30k-image scan take many minutes.
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(float(parts[0]))
                        vals = [class_id] + [float(x) for x in parts[1:5]]
                        bboxes.append(vals)
                    except ValueError:
                        pass
    except (OSError, TypeError):
        pass
    return bboxes

class UnlabeledWildDataset(data.Dataset):
    """
    Dynamic Dataset Loader for 'unlabeled/' directory.
    Supports dynamic rescanning of constantly growing datasets uploaded in parallel.
    """
    def __init__(self, root_dir: str = None, unlabeled_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112), allow_synthetic: bool = False):
        super(UnlabeledWildDataset, self).__init__()
        configured_root = root_dir if root_dir is not None else unlabeled_dir
        self.unlabeled_dir = resolve_task_root(configured_root, "unlabeled", "unlabeled dataset")
        self.special_wider_root = find_wider_special_root(self.unlabeled_dir)
        self.image_size = image_size
        self.allow_synthetic = allow_synthetic
        self.transform = transform if transform is not None else get_default_transform(image_size, is_train)
        self.samples = []
        self.rescan()

    def rescan(self):
        """
        Rescans the explicitly configured unlabeled image stream.

        This loader returns image crops as unlabeled recognition samples. Full
        scenes must be routed through DetectorCropDataset before Phase 3; this
        loader intentionally does not invent face boxes.
        """
        prev_count = len(self.samples)
        self.samples = []
        if self.unlabeled_dir and os.path.exists(self.unlabeled_dir):
            for root, dirs, files in os.walk(self.unlabeled_dir):
                dirs.sort()  # os.walk order is filesystem-dependent
                for f in sorted(files):
                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        path = os.path.join(root, f)
                        if self.special_wider_root and os.path.commonpath(
                                [os.path.abspath(path), os.path.abspath(self.special_wider_root)]
                        ) == os.path.abspath(self.special_wider_root):
                            continue
                        self.samples.append(path)
                        
        if len(self.samples) == 0 and self.allow_synthetic:
            for _ in range(500):
                self.samples.append("dummy_unlabeled")
        elif len(self.samples) == 0:
            raise DatasetConfigurationError(
                f"No valid images were found in unlabeled dataset: {self.unlabeled_dir}"
            )
        if len(self.samples) != prev_count:
            print(f"[Dynamic Unlabeled Rescan] Indexed {len(self.samples)} images (New: +{len(self.samples) - prev_count})")
        return len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Unlabeled dataset index out of range: {idx}")
        path = self.samples[idx]
        if path == "dummy_unlabeled":
            img_tensor = torch.randn(3, *self.image_size)
        else:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
            except Exception as exc:
                raise RuntimeError(f"Could not decode unlabeled image {path}: {exc}") from exc
                
        return img_tensor, -1, False


class DetectorCropDataset(data.Dataset):
    """Materialize detector-approved face crops from unlabeled scene images.

    This is the explicit bridge between full wild images and recognition
    adaptation. Detector boxes are preprocessing metadata only; they are never
    treated as identity labels or ground-truth localization targets.
    """
    def __init__(self, root_dir: str, detector, device: torch.device,
                 score_threshold: float = 0.75, transform=None,
                 image_size=(112, 112), max_detections_per_image: int = 20,
                 cache_path: str = None, use_cache: bool = True):
        super().__init__()
        self.root_dir = resolve_task_root(root_dir, "unlabeled", "detector-crop dataset")
        self.special_wider_root = find_wider_special_root(self.root_dir)
        if not self.root_dir or not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Unlabeled scene dataset does not exist: {self.root_dir}")
        self.device = device
        self.detector = detector.to(device).eval()
        self.score_threshold = float(score_threshold)
        self.transform = transform if transform is not None else get_default_transform(image_size, is_train=False)
        image_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs.sort()  # os.walk order is filesystem-dependent
            for filename in sorted(files):
                if filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    path = os.path.join(root, filename)
                    if self.special_wider_root and os.path.commonpath(
                            [os.path.abspath(path), os.path.abspath(self.special_wider_root)]
                    ) == os.path.abspath(self.special_wider_root):
                        continue
                    image_paths.append(path)
        if not image_paths:
            raise DatasetConfigurationError(f"No scene images were found in: {self.root_dir}")

        # Try to load the cache BEFORE computing the expensive manifest hash.
        # On AQUA's high-latency filesystem, stat()-ing hundreds of thousands of
        # files can take hours, so the cache must be checked first.
        if cache_path and use_cache and os.path.isfile(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                if cached.get("schema_version") == 1 and "samples" in cached:
                    self.samples = cached["samples"]
                    print(f"[DetectorCropDataset] Loaded cached detector crops: {cache_path} ({len(self.samples)} samples)")
                    self.detector = None
                    return
                print("[DetectorCropDataset] Detector crop cache is stale; rebuilding.")
            except Exception as exc:
                print(f"[DetectorCropDataset] Could not load cache; rebuilding: {exc}")

        manifest_hash = hashlib.sha256()
        for path in image_paths:
            try:
                stat = os.stat(path)
                manifest_hash.update(path.encode("utf-8", errors="surrogatepass"))
                manifest_hash.update(str(stat.st_size).encode())
                manifest_hash.update(str(stat.st_mtime_ns).encode())
            except OSError:
                continue
        cache_key = manifest_hash.hexdigest()
        cache_metadata = {
            "schema_version": 1,
            "root_dir": os.path.abspath(self.root_dir),
            "manifest_hash": cache_key,
            "score_threshold": self.score_threshold,
            "image_size": tuple(image_size),
            "max_detections_per_image": int(max_detections_per_image),
        }

        self.samples = []
        unreadable_paths = []
        with torch.no_grad():
            for path in image_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    scene = (pil_to_float_tensor(image).unsqueeze(0) * 2.0 - 1.0).to(device)
                    boxes, scores, _ = self.detector(scene, score_threshold=self.score_threshold)
                    limit = min(int(boxes.shape[0]), max_detections_per_image)
                    for index in range(limit):
                        box = boxes[index].detach().cpu().tolist()
                        score = float(scores[index].detach().cpu().item())
                        if box[2] - box[0] >= 8.0 and box[3] - box[1] >= 8.0:
                            self.samples.append((path, box, score))
                except (OSError, ValueError, RuntimeError) as exc:
                    unreadable_paths.append((path, str(exc)))
                    continue
        if unreadable_paths:
            print(
                f"[DetectorCropDataset] Skipped {len(unreadable_paths)} unreadable or invalid "
                f"unlabeled images; first={unreadable_paths[0][0]}: {unreadable_paths[0][1]}"
            )
        if not self.samples:
            raise DatasetConfigurationError(
                f"The detector found no faces above score {self.score_threshold:.3f} in: {self.root_dir}"
            )
        if cache_path and use_cache:
            cache_dir = os.path.dirname(os.path.abspath(cache_path))
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            torch.save({**cache_metadata, "samples": self.samples}, cache_path)
            print(f"[DetectorCropDataset] Saved detector crop cache: {cache_path} ({len(self.samples)} samples)")
        # Detection is complete; do not pickle a full GPU model into DataLoader
        # workers when the crops are later read for adaptation.
        self.detector = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Detector crop index out of range: {idx}")
        path, box, score = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            width, height = image.size
            x1, y1, x2, y2 = box
            crop = image.crop((
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                min(width, int(round(x2))),
                min(height, int(round(y2))),
            ))
            return self.transform(crop), -1, False, score
        except Exception as exc:
            raise RuntimeError(f"Could not decode detector crop {path}: {exc}") from exc


class NameLabeledFaceDataset(data.Dataset):
    """
    Dynamic Dataset Loader for 'name_label/' directory.
    Supports dynamic rescanning of constantly growing labeled datasets uploaded in parallel.
    """
    def __init__(self, root_dir: str = None, name_label_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112), allow_synthetic: bool = False):
        super(NameLabeledFaceDataset, self).__init__()
        configured_root = root_dir if root_dir is not None else name_label_dir
        self.name_label_dir = resolve_task_root(configured_root, "name_label", "identity dataset")
        self.image_size = image_size
        self.allow_synthetic = allow_synthetic
        self.transform = transform if transform is not None else get_default_transform(image_size, is_train)
        self.samples = []
        self.class_to_idx = {}
        self.rescan()

    def rescan(self):
        """
        Dynamically rescans name_label/ directory to index newly arrived identity images.
        """
        prev_count = len(self.samples)
        self.samples = []
        self.class_to_idx = {}
        if self.name_label_dir and os.path.exists(self.name_label_dir):
            self._parse_name_label_directory()
        elif self.allow_synthetic:
            self.class_to_idx = {f"Subject_{i}": i for i in range(50)}
            for i in range(500):
                self.samples.append(("dummy_labeled", i % 50))
        else:
            raise FileNotFoundError(
                f"Labeled dataset does not exist or is not configured: {self.name_label_dir}"
            )

        if not self.samples:
            raise DatasetConfigurationError(
                f"No valid labelled images were found in: {self.name_label_dir}"
            )

        if len(self.samples) != prev_count:
            print(f"[Dynamic NameLabeled Rescan] Indexed {len(self.samples)} images across {len(self.class_to_idx)} identities (New: +{len(self.samples) - prev_count})")
        return len(self.samples)

    def _parse_name_label_directory(self):
        rof_dir = os.path.join(self.name_label_dir, "ROF")
        if os.path.exists(rof_dir):
            self._parse_folder_identities(rof_dir, prefix="ROF_")
            
        wild_dir = os.path.join(self.name_label_dir, "face_detection_in_wild_dataset")
        if os.path.exists(wild_dir):
            self._parse_folder_identities(wild_dir, prefix="WILD_")
            
        mask_dir = os.path.join(self.name_label_dir, "face_with_mask")
        if os.path.exists(mask_dir):
            self._parse_filename_identities(mask_dir)
            
        if len(self.samples) == 0:
            self._parse_folder_identities(self.name_label_dir, prefix="")

    def _parse_folder_identities(self, base_dir: str, prefix: str = ""):
        for root, dirs, files in os.walk(base_dir):
            dirs.sort()
            image_files = sorted(f for f in files if f.lower().endswith(VALID_IMAGE_EXTENSIONS))
            if len(image_files) > 0:
                folder_name = canonical_identity_name(prefix + os.path.basename(root))
                if folder_name not in self.class_to_idx:
                    self.class_to_idx[folder_name] = len(self.class_to_idx)
                label_idx = self.class_to_idx[folder_name]
                
                for f in image_files:
                    self.samples.append((os.path.join(root, f), label_idx))

    def _parse_filename_identities(self, base_dir: str):
        pattern = re.compile(r'^(.*?)(?:_\d+)?\.(?:jpg|jpeg|png|bmp|webp)$', re.IGNORECASE)
        for root, dirs, files in os.walk(base_dir):
            # Sorted traversal keeps MASK_* identity indices identical across
            # machines; unsorted order silently permuted class labels.
            dirs.sort()
            for f in sorted(files):
                if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    match = pattern.match(f)
                    raw_name = match.group(1) if match else "unknown_mask"
                    identity_name = canonical_identity_name("MASK_" + raw_name)
                    
                    if identity_name not in self.class_to_idx:
                        self.class_to_idx[identity_name] = len(self.class_to_idx)
                    label_idx = self.class_to_idx[identity_name]
                    self.samples.append((os.path.join(root, f), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Labeled dataset index out of range: {idx}")
        last_error = None
        for offset in range(len(self.samples)):
            candidate = (idx + offset) % len(self.samples)
            path, label_idx = self.samples[candidate]
            try:
                if path == "dummy_labeled":
                    return torch.randn(3, *self.image_size), label_idx, True
                with Image.open(path) as image:
                    img_tensor = self.transform(image.convert('RGB'))
                return img_tensor, label_idx, True
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Could not decode any labelled image while serving index {idx}: {last_error}"
        ) from last_error


class WiderFolderLabeledFaceDataset(data.Dataset):
    """Use WIDER YOLO boxes as crops and the folder text as a label.

    AQUA's special dataset stores images and labels in parallel trees:

    ``.../images/{train,val}/0--Parade/image.jpg``
    ``.../labels/{train,val}/0--Parade/image.txt``

    Each valid YOLO box becomes one recognition crop. The numeric prefix and
    punctuation in the folder name are removed, so ``0--Parade`` becomes the
    label ``Parade``. This is the requested generic folder-label contract; it
    is not treated as a verified person identity.
    """
    def __init__(self, root_dir: str, transform=None, is_train: bool = True,
                 image_size=(112, 112), split: str = None):
        super().__init__()
        self.wider_root = find_wider_special_root(root_dir) or root_dir
        if not self.wider_root or not os.path.isdir(self.wider_root):
            raise FileNotFoundError(f"WIDER folder-label dataset does not exist: {root_dir}")
        self.image_size = image_size
        self.is_train = is_train
        self.split = split.lower() if split else None
        self.transform = transform if transform is not None else get_default_transform(image_size, is_train)
        self.samples = []
        self.class_to_idx = {}
        self.rescan()

    def _register_class(self, name):
        if name not in self.class_to_idx:
            self.class_to_idx[name] = len(self.class_to_idx)
        return self.class_to_idx[name]

    def _images_tree_roots(self):
        roots = []
        for current_root, directories, _ in os.walk(self.wider_root):
            directories.sort()
            if os.path.basename(current_root).lower() == "images":
                labels_root = os.path.join(os.path.dirname(current_root), "labels")
                if os.path.isdir(labels_root):
                    roots.append((current_root, labels_root))
        return roots

    @staticmethod
    def _folder_name_for_image(image_root, image_path):
        relative = os.path.relpath(image_path, image_root)
        parts = relative.split(os.sep)
        if len(parts) < 2:
            raise DatasetConfigurationError(
                f"WIDER image is not inside a folder label: {image_path}"
            )
        if parts[0].lower() in DETECTION_SPLIT_NAMES:
            if len(parts) < 3:
                raise DatasetConfigurationError(
                    f"WIDER image is missing a folder label below split: {image_path}"
                )
            return parts[1]
        return parts[0]

    def rescan(self):
        self.samples = []
        self.class_to_idx = {}
        for images_root, labels_root in self._images_tree_roots():
            for current_root, directories, files in os.walk(images_root):
                directories.sort()
                for filename in sorted(files):
                    if not filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        continue
                    image_path = os.path.join(current_root, filename)
                    relative = os.path.relpath(image_path, images_root)
                    parts = relative.split(os.sep)
                    explicit_split = parts[0].lower() if parts and parts[0].lower() in DETECTION_SPLIT_NAMES else None
                    if self.split and explicit_split and explicit_split != self.split:
                        continue
                    label_path = os.path.join(labels_root, os.path.splitext(relative)[0] + ".txt")
                    boxes = parse_yolo_txt(label_path)
                    if not boxes:
                        continue
                    folder_name = wider_folder_identity_name(
                        self._folder_name_for_image(images_root, image_path)
                    )
                    label_idx = self._register_class(folder_name)
                    for _, xc, yc, width, height in self._iter_boxes(boxes):
                        self.samples.append((image_path, (xc, yc, width, height), label_idx))
        if not self.samples:
            raise DatasetConfigurationError(
                f"No valid WIDER folder-labelled boxes were found in: {self.wider_root}"
            )
        print(
            "[WIDER folder labels] Indexed %d face crops across %d cleaned folder labels from %s" %
            (len(self.samples), len(self.class_to_idx), self.wider_root)
        )
        return len(self.samples)

    @staticmethod
    def _iter_boxes(boxes):
        for row in boxes:
            if len(row) == 4:
                yield 0, *row
            else:
                yield int(row[0]), *row[1:5]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"WIDER labelled dataset index out of range: {idx}")
        path, (xc, yc, width, height), label_idx = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            image_width, image_height = image.size
            x1 = max(0, int((xc - width / 2.0) * image_width))
            y1 = max(0, int((yc - height / 2.0) * image_height))
            x2 = min(image_width, int((xc + width / 2.0) * image_width))
            y2 = min(image_height, int((yc + height / 2.0) * image_height))
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                raise ValueError(f"invalid WIDER crop box {(x1, y1, x2, y2)}")
            crop = image.crop((x1, y1, x2, y2))
            return self.transform(crop), label_idx, True
        except Exception as exc:
            raise RuntimeError(f"Could not decode WIDER labelled crop {path}: {exc}") from exc


class CombinedIdentityDataset(data.Dataset):
    """Combine identity crops and WIDER folder-labelled face crops."""
    def __init__(self, datasets, image_size=(112, 112), is_train: bool = True):
        super().__init__()
        self.datasets = list(datasets)
        self.image_size = image_size
        self.is_train = is_train
        self.transform = get_default_transform(image_size, is_train)
        self.class_to_idx = {}
        self.samples = []
        for dataset in self.datasets:
            reverse = {index: name for name, index in dataset.class_to_idx.items()}
            local_to_global = {}
            for local_index, name in sorted(reverse.items()):
                if name not in self.class_to_idx:
                    self.class_to_idx[name] = len(self.class_to_idx)
                local_to_global[local_index] = self.class_to_idx[name]
            for sample in dataset.samples:
                if len(sample) == 2:
                    path, local_index = sample
                    self.samples.append((path, None, local_to_global[local_index]))
                else:
                    path, box, local_index = sample
                    self.samples.append((path, box, local_to_global[local_index]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Combined identity dataset index out of range: {idx}")
        last_error = None
        for offset in range(len(self.samples)):
            candidate = (idx + offset) % len(self.samples)
            path, box, label_idx = self.samples[candidate]
            try:
                if path == "dummy_labeled":
                    return torch.randn(3, *self.image_size), label_idx, True
                with Image.open(path) as source:
                    image = source.convert("RGB")
                if box is not None:
                    xc, yc, width, height = box
                    image_width, image_height = image.size
                    x1 = max(0, int((xc - width / 2.0) * image_width))
                    y1 = max(0, int((yc - height / 2.0) * image_height))
                    x2 = min(image_width, int((xc + width / 2.0) * image_width))
                    y2 = min(image_height, int((yc + height / 2.0) * image_height))
                    image = image.crop((x1, y1, x2, y2))
                return self.transform(image), label_idx, True
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Could not decode any combined identity image while serving index {idx}: {last_error}"
        ) from last_error


class BoundingBoxFaceDataset(data.Dataset):
    """
    Dynamic Dataset Loader for 'bb_label/' directory.
    Supports dynamic rescanning of bounding box datasets uploaded in parallel.
    """
    def __init__(self, root_dir: str = None, bb_label_dir: str = None, transform=None, is_train: bool = True, image_size=(112, 112), allow_synthetic: bool = False, detection_mode: bool = False, split: str = None, include_wider: bool = True):
        super(BoundingBoxFaceDataset, self).__init__()
        configured_root = root_dir if root_dir is not None else bb_label_dir
        self.bb_label_dir = resolve_task_root(configured_root, "bb_label", "bounding-box dataset")
        self.image_size = image_size
        self.allow_synthetic = allow_synthetic
        self.detection_mode = detection_mode
        self.is_train = is_train
        self.split = split.lower() if split else None
        self.include_wider = bool(include_wider)
        self.transform = transform if transform is not None else get_default_transform(image_size, is_train)
        self.samples = []
        self.rescan()

    def rescan(self):
        """
        Dynamically rescans bb_label/ directory for newly uploaded YOLO / Darknet bounding box images.
        """
        prev_count = len(self.samples)
        self.samples = []
        if self.bb_label_dir and os.path.exists(self.bb_label_dir):
            self._parse_bb_label_directory()
        elif self.allow_synthetic:
            for i in range(200):
                self.samples.append(("dummy_bb", None, 0))
        else:
            raise FileNotFoundError(
                f"Bounding-box dataset does not exist or is not configured: {self.bb_label_dir}"
            )

        if not self.samples:
            raise DatasetConfigurationError(
                f"No valid bounding-box images were found in: {self.bb_label_dir}"
            )
                
        if len(self.samples) != prev_count:
            print(f"[Dynamic BoundingBox Rescan] Indexed {len(self.samples)} bounding box images (New: +{len(self.samples) - prev_count})")
        return len(self.samples)

    def _parse_bb_label_directory(self):
        search_dirs = []
        if self.bb_label_dir and os.path.exists(self.bb_label_dir):
            search_dirs.append(self.bb_label_dir)
            parent_dir = os.path.dirname(self.bb_label_dir.rstrip('/'))
            if self.include_wider:
                wider_cand = os.path.join(parent_dir, "unlabeled", "masked-face-detection-wider-dataset")
                if os.path.exists(wider_cand) and wider_cand not in search_dirs:
                    search_dirs.append(wider_cand)

        indexed_images = 0
        for sdir in search_dirs:
            if sdir and os.path.exists(sdir):
                for root, dirs, files in os.walk(sdir):
                    dirs.sort()  # deterministic sample order / image_id across machines
                    label_root = root
                    normalized_root = os.path.normpath(root)
                    image_marker = os.sep + "images"
                    marker_index = normalized_root.lower().rfind(image_marker)
                    if marker_index >= 0:
                        parallel_labels = (
                            normalized_root[:marker_index]
                            + os.sep
                            + "labels"
                            + normalized_root[marker_index + len(image_marker):]
                        )
                        if os.path.isdir(parallel_labels):
                            label_root = parallel_labels
                    for f in sorted(files):
                        if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                            indexed_images += 1
                            img_path = os.path.join(root, f)
                            path_components = {
                                component.lower() for component in os.path.normpath(root).split(os.sep)
                            }
                            explicit_splits = path_components & DETECTION_SPLIT_NAMES
                            # Keep unsplit AQUA subsets (for example
                            # facemaskyolo/data/images) alongside explicit
                            # splits. When a split is requested, filter only
                            # directories that explicitly name a split.
                            if self.split and explicit_splits and self.split not in explicit_splits:
                                continue
                            base_name = os.path.splitext(f)[0]

                            txt_path = os.path.join(label_root, base_name + ".txt")
                                
                            bboxes = parse_yolo_txt(txt_path)
                            # Detection training needs positive targets. Images
                            # without a readable annotation are not silently
                            # converted into a fake class-0 sample.
                            if bboxes:
                                self.samples.append((img_path, bboxes, 0))
                    if indexed_images and indexed_images % 5000 == 0:
                        print(
                            f"[BoundingBox Rescan] inspected {indexed_images} images",
                            flush=True,
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Bounding-box dataset index out of range: {idx}")
        path, bboxes, class_id = self.samples[idx]
        if path == "dummy_bb":
            img_tensor = torch.randn(3, *self.image_size)
            return img_tensor, class_id, True
            
        try:
            img = Image.open(path).convert('RGB')
            w_img, h_img = img.size

            if self.detection_mode:
                boxes = []
                labels = []
                for class_id_yolo, xc, yc, w_box, h_box in self._iter_yolo_boxes(bboxes):
                    x1 = max(0.0, (xc - w_box / 2.0) * w_img)
                    y1 = max(0.0, (yc - h_box / 2.0) * h_img)
                    x2 = min(float(w_img), (xc + w_box / 2.0) * w_img)
                    y2 = min(float(h_img), (yc + h_box / 2.0) * h_img)
                    if x2 > x1 + 1.0 and y2 > y1 + 1.0:
                        boxes.append([x1, y1, x2, y2])
                        # Detection is intentionally class-agnostic: all
                        # foreground YOLO rows describe a face, while the
                        # torchvision background class remains 0. This keeps
                        # the detector compatible with AQUA subsets whose mask
                        # class IDs differ.
                        labels.append(1)

                if self.is_train and torch.rand(1).item() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    boxes = [[w_img - x2, y1, w_img - x1, y2] for x1, y1, x2, y2 in boxes]
                image_tensor = pil_to_float_tensor(img)
                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    "labels": torch.tensor(labels, dtype=torch.int64),
                    "image_id": torch.tensor([idx], dtype=torch.int64),
                    "area": torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=torch.float32),
                    "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
                }
                return image_tensor, target
            
            if bboxes and len(bboxes) > 0:
                _, xc, yc, w_box, h_box = next(self._iter_yolo_boxes(bboxes))
                x1 = max(0, int((xc - w_box / 2.0) * w_img))
                y1 = max(0, int((yc - h_box / 2.0) * h_img))
                x2 = min(w_img, int((xc + w_box / 2.0) * w_img))
                y2 = min(h_img, int((yc + h_box / 2.0) * h_img))
                
                if x2 > x1 + 5 and y2 > y1 + 5:
                    img = img.crop((x1, y1, x2, y2))
                    
            img_tensor = self.transform(img)
        except Exception as exc:
            raise RuntimeError(f"Could not decode bounding-box image {path}: {exc}") from exc
            
        return img_tensor, class_id, True

    @staticmethod
    def _iter_yolo_boxes(bboxes):
        """Yield ``class_id, xc, yc, width, height`` for parsed YOLO rows."""
        for row in bboxes:
            if len(row) == 4:
                yield 0, *row
            else:
                yield int(row[0]), *row[1:5]


class UnifiedWildFaceDataset(data.Dataset):
    """
    Unified Dynamic Dataset Loader combining all datastreams with automatic epoch-wise rescanning.
    """
    def __init__(self, root_dir: str = None, meta_file: str = None, is_labeled: bool = True, transform=None, is_train: bool = True, image_size=(112, 112), allow_synthetic: bool = False):
        super(UnifiedWildFaceDataset, self).__init__()
        self.root_dir = _require_dataset_root(root_dir, "unified face dataset")
        self.is_labeled = is_labeled
        self.is_train = is_train
        if is_labeled:
            labelled_root = os.path.join(root_dir, "name_label")
            labelled_root = labelled_root if os.path.isdir(labelled_root) else root_dir
            self.primary_ds = NameLabeledFaceDataset(
                name_label_dir=labelled_root,
                transform=transform, is_train=is_train, image_size=image_size,
                allow_synthetic=allow_synthetic,
            )
            self.class_to_idx = self.primary_ds.class_to_idx
        else:
            unlabeled_root = os.path.join(root_dir, "unlabeled")
            unlabeled_root = unlabeled_root if os.path.isdir(unlabeled_root) else root_dir
            self.primary_ds = UnlabeledWildDataset(
                unlabeled_dir=unlabeled_root,
                transform=transform, is_train=is_train, image_size=image_size,
                allow_synthetic=allow_synthetic,
            )
            self.class_to_idx = {}

    def rescan(self):
        """
        Rescans all sub-datasets to pick up newly arrived parallel file uploads.
        """
        count = self.primary_ds.rescan()
        self.class_to_idx = getattr(self.primary_ds, "class_to_idx", {})
        return count

    def __len__(self):
        return len(self.primary_ds)

    def __getitem__(self, idx):
        return self.primary_ds[idx]


RobustUniversalFaceDataset = UnifiedWildFaceDataset
WildFaceDataset = NameLabeledFaceDataset
UnlabeledFaceDataset = UnlabeledWildDataset

class Phase2UnifiedAttributeDataset(data.Dataset):
    """CelebA-only attribute dataset with real labels and no fake wild labels.

    Attribute supervision is a separate task from identity recognition. Wild
    images without annotation are intentionally excluded instead of receiving
    random targets. Occlusion masks are not fabricated here; a future mask
    dataset can be added as a separate task with its own annotation contract.
    """

    def __init__(self, root_dir: str = None, celeba_dir: str = None, attr_file: str = None,
                 is_train: bool = True, image_size=(112, 112)):
        super().__init__()
        self.root_dir = _require_dataset_root(root_dir or celeba_dir, "CelebA attribute dataset")
        self.image_size = image_size
        self.transform = get_default_transform(image_size, is_train)
        self.samples = []

        parent_dir = os.path.dirname(self.root_dir.rstrip('/'))
        if attr_file is None:
            candidates = [
                os.path.join(self.root_dir, "list_attr_celeba.csv"),
                os.path.join(self.root_dir, "list_attr_celeba.txt"),
                os.path.join(self.root_dir, "Anno", "list_attr_celeba.txt"),
                os.path.join(parent_dir, "list_attr_celeba.csv"),
                os.path.join(parent_dir, "list_attr_celeba.txt"),
            ]
            attr_file = next((path for path in candidates if os.path.isfile(path)), None)
        if not attr_file or not os.path.isfile(attr_file):
            raise FileNotFoundError(
                "CelebA attribute annotations were not found. Expected list_attr_celeba.csv or .txt."
            )

        image_roots = [
            self.root_dir,
            os.path.join(self.root_dir, "img_align_celeba"),
            os.path.join(self.root_dir, "img_align_celeba", "img_align_celeba"),
        ]
        raw_entries = []
        if attr_file.lower().endswith('.csv'):
            import csv
            with open(attr_file, 'r', encoding='utf-8', errors='strict') as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for row in reader:
                    if len(row) < 41:
                        continue
                    raw_entries.append((row[0], torch.tensor([1.0 if int(x) == 1 else 0.0 for x in row[1:41]], dtype=torch.float32)))
        else:
            with open(attr_file, 'r', encoding='utf-8', errors='strict') as handle:
                lines = handle.readlines()
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) >= 41:
                    raw_entries.append((parts[0], torch.tensor([1.0 if int(x) == 1 else 0.0 for x in parts[1:41]], dtype=torch.float32)))

        for image_name, target in raw_entries:
            image_path = next((os.path.join(root, image_name) for root in image_roots if os.path.isfile(os.path.join(root, image_name))), None)
            if image_path:
                self.samples.append((image_path, target))
        if not self.samples:
            raise DatasetConfigurationError(f"No CelebA images matched annotations in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"CelebA dataset index out of range: {idx}")
        path, target = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            return self.transform(image), target
        except Exception as exc:
            raise RuntimeError(f"Could not decode CelebA image {path}: {exc}") from exc


CelebAAttributeDataset = Phase2UnifiedAttributeDataset

import os
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import (
    BoundingBoxFaceDataset,
    CombinedIdentityDataset,
    DatasetConfigurationError,
    DetectorCropDataset,
    NameLabeledFaceDataset,
    UnlabeledWildDataset,
    WiderFolderLabeledFaceDataset,
)
from losses.broadface_queue import BroadFaceMemoryQueue
from losses.curricular_loss import CurricularFaceLoss
from models.ddrc_solver import DDRCClassifier
from models.detector import TorchvisionFaceDetector, soft_nms_pytorch
from models.gcn_cluster import GCNLinkPredictor
from pipeline.alignment import align_face_crop, estimate_pose_from_landmarks


class TestCorrectedContracts(unittest.TestCase):
    def test_dataset_requires_real_data_and_indexes_valid_samples(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                NameLabeledFaceDataset(root_dir=os.path.join(root, "missing"))
            image_dir = os.path.join(root, "alice_wearing_mask")
            os.makedirs(image_dir)
            Image.new("RGB", (32, 32), (120, 80, 40)).save(os.path.join(image_dir, "face.png"))
            dataset = NameLabeledFaceDataset(root_dir=root, is_train=False)
            self.assertEqual(len(dataset), 1)
            image, label, is_labeled = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 112, 112))
            self.assertEqual(label, 0)
            self.assertTrue(is_labeled)
            with self.assertRaises(IndexError):
                dataset[1]

    def test_name_dataset_skips_unreadable_uploads(self):
        with tempfile.TemporaryDirectory() as root:
            image_dir = os.path.join(root, "alice")
            os.makedirs(image_dir)
            Image.new("RGB", (32, 32), (120, 80, 40)).save(os.path.join(image_dir, "valid.jpg"))
            with open(os.path.join(image_dir, "partial.jpg"), "wb"):
                pass

            dataset = NameLabeledFaceDataset(root_dir=root, is_train=False)
            self.assertEqual(len(dataset), 2)
            image, label, is_labeled = dataset[1]
            self.assertEqual(tuple(image.shape), (3, 112, 112))
            self.assertEqual(label, 0)
            self.assertTrue(is_labeled)

    def test_unlabeled_dataset_does_not_fabricate_samples(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises((FileNotFoundError, DatasetConfigurationError)):
                UnlabeledWildDataset(root_dir=root)

    def test_task_parent_roots_resolve_without_mixing_datasets(self):
        with tempfile.TemporaryDirectory() as root:
            name_root = os.path.join(root, "name_label", "alice")
            unlabeled_root = os.path.join(root, "unlabeled")
            os.makedirs(name_root)
            os.makedirs(unlabeled_root)
            Image.new("RGB", (32, 32), (120, 80, 40)).save(os.path.join(name_root, "face.png"))
            Image.new("RGB", (32, 32), (20, 40, 80)).save(os.path.join(unlabeled_root, "scene.png"))

            labelled = NameLabeledFaceDataset(root_dir=root, is_train=False)
            unlabeled = UnlabeledWildDataset(root_dir=root, is_train=False)
            self.assertEqual(len(labelled), 1)
            self.assertEqual(len(unlabeled), 1)
            with self.assertRaises(DatasetConfigurationError):
                NameLabeledFaceDataset(root_dir=os.path.join(root, "unlabeled"), is_train=False)

    def test_detector_crop_dataset_uses_boxes_as_preprocessing_only(self):
        class FakeDetector:
            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, image, score_threshold):
                device = image.device
                return (
                    torch.tensor([[8.0, 8.0, 24.0, 24.0]], device=device),
                    torch.tensor([0.95], device=device),
                    torch.zeros((1, 5, 2), device=device),
                )

        with tempfile.TemporaryDirectory() as root:
            unlabeled_root = os.path.join(root, "unlabeled")
            os.makedirs(unlabeled_root)
            Image.new("RGB", (32, 32), (20, 40, 80)).save(os.path.join(unlabeled_root, "scene.png"))
            dataset = DetectorCropDataset(
                root_dir=root,
                detector=FakeDetector(),
                device=torch.device("cpu"),
                score_threshold=0.9,
            )
            image, label, is_labeled, score = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 112, 112))
            self.assertEqual(label, -1)
            self.assertFalse(is_labeled)
            self.assertAlmostEqual(score, 0.95, places=5)

    def test_wider_folder_dataset_uses_parallel_yolo_boxes_and_cleaned_folder_labels(self):
        with tempfile.TemporaryDirectory() as root:
            wider = os.path.join(
                root,
                "masked-face-detection-wider-dataset",
                "COVID-mask-detection_WIDER",
                "Wider-mask-detection",
            )
            train_images = os.path.join(wider, "images", "train", "0--Parade")
            train_labels = os.path.join(wider, "labels", "train", "0--Parade")
            val_images = os.path.join(wider, "images", "val", "1--Handshaking")
            val_labels = os.path.join(wider, "labels", "val", "1--Handshaking")
            for directory in (train_images, train_labels, val_images, val_labels):
                os.makedirs(directory)
            Image.new("RGB", (100, 50), (120, 80, 40)).save(os.path.join(train_images, "parade.jpg"))
            with open(os.path.join(train_labels, "parade.txt"), "w", encoding="utf-8") as handle:
                handle.write("1 0.25 0.50 0.20 0.60\n0 0.75 0.50 0.20 0.60\n")
            Image.new("RGB", (100, 50), (20, 40, 80)).save(os.path.join(val_images, "handshake.jpg"))
            with open(os.path.join(val_labels, "handshake.txt"), "w", encoding="utf-8") as handle:
                handle.write("0 0.50 0.50 0.40 0.80\n")

            dataset = WiderFolderLabeledFaceDataset(root_dir=root, is_train=False)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(set(dataset.class_to_idx), {"Parade", "Handshaking"})
            image, label, is_labeled = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 112, 112))
            self.assertTrue(is_labeled)
            self.assertEqual(label, dataset.class_to_idx["Parade"])

            detector_dataset = BoundingBoxFaceDataset(
                root_dir=os.path.join(root, "masked-face-detection-wider-dataset"),
                is_train=False,
                detection_mode=True,
            )
            self.assertEqual(len(detector_dataset), 2)

            name_root = os.path.join(root, "name_label", "Alice")
            os.makedirs(name_root)
            Image.new("RGB", (32, 32), (200, 100, 60)).save(os.path.join(name_root, "alice.png"))
            name_dataset = NameLabeledFaceDataset(
                root_dir=os.path.join(root, "name_label"),
                is_train=False,
            )
            combined = CombinedIdentityDataset(
                [name_dataset, dataset],
                image_size=(112, 112),
                is_train=False,
            )
            self.assertEqual(len(combined), 4)
            self.assertEqual(set(combined.class_to_idx), {"Alice", "Parade", "Handshaking"})
            combined_image, combined_label, combined_is_labeled = combined[1]
            self.assertEqual(tuple(combined_image.shape), (3, 112, 112))
            self.assertTrue(combined_is_labeled)
            self.assertEqual(combined_label, combined.class_to_idx["Parade"])

    def test_aqua_yolo_detection_targets_support_parallel_labels_and_multiple_boxes(self):
        with tempfile.TemporaryDirectory() as root:
            images = os.path.join(root, "darknet", "mask-dataset", "images", "train")
            labels = os.path.join(root, "darknet", "mask-dataset", "labels", "train")
            os.makedirs(images)
            os.makedirs(labels)
            image_path = os.path.join(images, "sample.jpg")
            Image.new("RGB", (100, 50), (120, 80, 40)).save(image_path)
            with open(os.path.join(labels, "sample.txt"), "w", encoding="utf-8") as handle:
                handle.write("7 0.50 0.50 0.40 0.80\n3 0.20 0.20 0.10 0.20\n")
            for split_name in ("test", "validate"):
                split_images = os.path.join(root, "darknet", "mask-dataset", "images", split_name)
                split_labels = os.path.join(root, "darknet", "mask-dataset", "labels", split_name)
                os.makedirs(split_images)
                os.makedirs(split_labels)
                Image.new("RGB", (100, 50), (80, 60, 40)).save(os.path.join(split_images, f"{split_name}.jpg"))
                with open(os.path.join(split_labels, f"{split_name}.txt"), "w", encoding="utf-8") as handle:
                    handle.write("0 0.50 0.50 0.40 0.80\n")
            unsplit_images = os.path.join(root, "facemaskyolo", "data", "images")
            unsplit_labels = os.path.join(root, "facemaskyolo", "data", "labels")
            os.makedirs(unsplit_images)
            os.makedirs(unsplit_labels)
            Image.new("RGB", (40, 40), (20, 40, 80)).save(os.path.join(unsplit_images, "other.jpg"))
            with open(os.path.join(unsplit_labels, "other.txt"), "w", encoding="utf-8") as handle:
                handle.write("0 0.50 0.50 0.50 0.50\n")

            dataset = BoundingBoxFaceDataset(
                root_dir=root,
                is_train=False,
                detection_mode=True,
            )
            # Look the sample up by name: dataset order is now sorted and
            # deterministic, so index 0 is the "test" split, not "train".
            sample_index = next(
                i for i, (path, _, _) in enumerate(dataset.samples) if path.endswith("sample.jpg")
            )
            image, target = dataset[sample_index]
            self.assertEqual(tuple(image.shape), (3, 50, 100))
            self.assertEqual(len(dataset), 4)
            self.assertEqual(tuple(target["boxes"].shape), (2, 4))
            self.assertEqual(target["labels"].tolist(), [1, 1])
            self.assertEqual(target["boxes"][0].tolist(), [30.0, 5.0, 70.0, 45.0])
            self.assertEqual(target["boxes"][1].tolist(), [15.0, 5.0, 25.0, 15.0])
            train_only = BoundingBoxFaceDataset(root_dir=root, is_train=False, detection_mode=True, split="train")
            self.assertEqual(len(train_only), 2)

    def test_bb_label_detector_includes_nested_wider_source(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_root = os.path.join(root, "Face_Dataset")
            bb_images = os.path.join(dataset_root, "bb_label", "images")
            bb_labels = os.path.join(dataset_root, "bb_label", "labels")
            wider = os.path.join(
                dataset_root,
                "unlabeled",
                "masked-face-detection-wider-dataset",
                "COVID-mask-detection_WIDER",
                "Wider-mask-detection",
            )
            wider_images = os.path.join(wider, "images", "train", "0--Parade")
            wider_labels = os.path.join(wider, "labels", "train", "0--Parade")
            for directory in (bb_images, bb_labels, wider_images, wider_labels):
                os.makedirs(directory)

            Image.new("RGB", (40, 40), (100, 80, 60)).save(os.path.join(bb_images, "bb.jpg"))
            with open(os.path.join(bb_labels, "bb.txt"), "w", encoding="utf-8") as handle:
                handle.write("0 0.50 0.50 0.50 0.50\n")
            Image.new("RGB", (40, 40), (60, 80, 100)).save(os.path.join(wider_images, "wider.jpg"))
            with open(os.path.join(wider_labels, "wider.txt"), "w", encoding="utf-8") as handle:
                handle.write("1 0.50 0.50 0.40 0.40\n")

            dataset = BoundingBoxFaceDataset(
                root_dir=os.path.join(dataset_root, "bb_label"),
                is_train=False,
                detection_mode=True,
            )
            self.assertEqual(len(dataset), 2)

    def test_anchor_free_detector_backend_is_explicit_and_trainable(self):
        detector = TorchvisionFaceDetector(
            num_classes=2,
            pretrained_backbone=False,
            backend="anchor_free",
        )
        self.assertEqual(detector.backend, "anchor_free")
        image = torch.zeros(3, 64, 64)
        targets = [{
            "boxes": torch.tensor([[12.0, 12.0, 40.0, 44.0]]),
            "labels": torch.ones(1, dtype=torch.int64),
        }]
        losses = detector.detection_loss([image], targets)
        self.assertTrue(torch.isfinite(sum(losses.values())))

    def test_torchvision_faster_rcnn_backend_is_explicit_and_trainable(self):
        try:
            detector = TorchvisionFaceDetector(
                num_classes=2,
                pretrained_backbone=False,
                backend="torchvision",
                detector_min_size=64,
                detector_max_size=128,
            )
        except ImportError:
            self.skipTest("torchvision detection operators are unavailable")
        self.assertEqual(detector.backend, "torchvision")
        self.assertEqual(detector.architecture, "fasterrcnn_resnet50_fpn")
        image = torch.zeros(3, 96, 96)
        targets = [{
            "boxes": torch.tensor([[20.0, 20.0, 60.0, 70.0]]),
            "labels": torch.ones(1, dtype=torch.int64),
        }]
        losses = detector.detection_loss([image], targets)
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_alignment_and_landmark_pose(self):
        crop = torch.zeros(1, 3, 112, 112)
        landmarks = torch.tensor([[[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]]])
        aligned, used = align_face_crop(crop, landmarks)
        self.assertEqual(tuple(aligned.shape), (1, 3, 112, 112))
        self.assertTrue(used)
        pose = estimate_pose_from_landmarks(landmarks[0])
        self.assertEqual(pose["valid"], 1.0)
        self.assertTrue(-90.0 <= pose["yaw"] <= 90.0)

    def test_soft_nms_preserves_landmark_source_indices(self):
        boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 10.0, 80.0, 30.0]])
        scores = torch.tensor([0.4, 0.9])
        _, _, indices = soft_nms_pytorch(boxes, scores, score_threshold=0.1, return_indices=True)
        self.assertEqual(indices.tolist(), [1, 0])

    def test_curricular_loss_and_queue(self):
        loss_fn = CurricularFaceLoss(in_features=8, num_classes=4)
        embeddings = F.normalize(torch.randn(3, 8), dim=1)
        labels = torch.tensor([0, 1, 2])
        loss = loss_fn(embeddings, labels)
        self.assertTrue(torch.isfinite(loss))
        queue = BroadFaceMemoryQueue(in_features=8, num_classes=4, queue_size=4)
        queue.enqueue(embeddings, labels)
        self.assertEqual(int(queue.queue_ptr.item()), 3)
        with self.assertRaises(ValueError):
            queue.enqueue(embeddings, torch.tensor([0, 1, 9]))

    def test_dictionary_classifier_known_unknown_and_one_class(self):
        classifier = DDRCClassifier(feature_dim=2, atoms_per_class=1)
        classifier.enroll_prototypes(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ["alice", "bob"])
        predictions, confidences, residuals = classifier(torch.tensor([[1.0, 0.0], [0.7071, 0.7071]]), residual_threshold=0.55, margin_threshold=0.10)
        self.assertEqual(predictions[0], "alice")
        self.assertEqual(predictions[1], "unknown")
        self.assertEqual(tuple(residuals.shape), (2, 2))
        self.assertTrue(torch.isfinite(confidences).all())
        one_class = DDRCClassifier(feature_dim=2, atoms_per_class=1)
        one_class.enroll_prototypes(torch.tensor([[1.0, 0.0]]), ["alice"])
        self.assertEqual(one_class(torch.tensor([[1.0, 0.0]]))[0][0], "alice")

    def test_gcn_global_labels_are_valid(self):
        predictor = GCNLinkPredictor(feature_dim=4, hidden_dim=8, k_neighbors=2)
        predictor.eval()
        embeddings = F.normalize(torch.randn(8, 4), dim=1)
        labels, active = predictor.generate_global_pseudo_labels(embeddings, confidence_threshold=0.99, start_class_idx=10)
        self.assertEqual(tuple(labels.shape), (8,))
        self.assertEqual(tuple(active.shape), (8,))
        self.assertTrue(((labels == -1) | (labels >= 10)).all())

    def test_checkpoint_round_trip_for_queue(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "queue.pt")
            original = BroadFaceMemoryQueue(in_features=4, num_classes=2, queue_size=3)
            original.enqueue(F.normalize(torch.randn(2, 4), dim=1), torch.tensor([0, 1]))
            torch.save(original.state_dict(), path)
            restored = BroadFaceMemoryQueue(in_features=4, num_classes=2, queue_size=3)
            restored.load_state_dict(torch.load(path, map_location="cpu"))
            self.assertTrue(torch.equal(original.feature_mb, restored.feature_mb))
            self.assertTrue(torch.equal(original.label_mb, restored.label_mb))

    def test_aqua_detector_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            original = TorchvisionFaceDetector(num_classes=2, pretrained_backbone=False)
            path = os.path.join(root, "face_detector.pt")
            torch.save({
                "schema_version": 2,
                "kind": "face_detector",
                "num_classes": 2,
                "model_state": original.model.state_dict(),
            }, path)
            restored = TorchvisionFaceDetector(num_classes=2, checkpoint_path=path, pretrained_backbone=False)
            self.assertTrue(restored.loaded_checkpoint)
            first_key = next(iter(original.model.state_dict()))
            self.assertTrue(torch.equal(original.model.state_dict()[first_key], restored.model.state_dict()[first_key]))


if __name__ == "__main__":
    unittest.main()

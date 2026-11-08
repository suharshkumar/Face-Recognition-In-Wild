"""Regression tests for the open-set gallery classifier and crop preprocessing."""

import os
import sys
import unittest

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calibrate_open_set import recommend, sweep_open_set
from dataset import get_default_transform
from eval_lfw import kfold_verification_accuracy, parse_pairs
from models.ddrc_solver import DDRCClassifier, spherical_kmeans
from pipeline.alignment import align_face_crop

DIM = 512


def unit(n, dim=DIM, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, dim, generator=generator), dim=1)


def with_cosine(target, cosine, n=200, seed=1):
    """Unit vectors at an exact cosine to ``target``."""
    noise = unit(n, target.numel(), seed)
    noise = F.normalize(noise - (noise @ target)[:, None] * target, dim=1)
    return cosine * target + (1.0 - cosine ** 2) ** 0.5 * noise


def gallery(num_classes=10, seed=0):
    prototypes = unit(num_classes, seed=seed)
    classifier = DDRCClassifier(feature_dim=DIM, atoms_per_class=1)
    classifier.enroll_gallery(prototypes, torch.arange(num_classes), [f"id{i}" for i in range(num_classes)])
    return classifier, prototypes


class OpenSetGateTests(unittest.TestCase):
    def test_residuals_are_relative_and_bounded(self):
        classifier, prototypes = gallery()
        residuals = classifier.class_residuals(prototypes)
        self.assertTrue(((residuals >= 0) & (residuals <= 1.0 + 1e-5)).all())
        # Strangers leave nearly all of their energy unexplained.
        stranger = classifier.class_residuals(unit(50, seed=7)).min(dim=1).values
        self.assertGreater(float(stranger.min()), 0.9)

    def test_residual_threshold_actually_controls_acceptance(self):
        # The old per-dimension residual never exceeded 4/512, so this knob was dead.
        classifier, prototypes = gallery()
        probes = with_cosine(prototypes[3], 0.6)
        strict, _, _ = classifier(probes, residual_threshold=0.30, margin_threshold=0.0)
        loose, _, _ = classifier(probes, residual_threshold=0.95, margin_threshold=0.0)
        self.assertTrue(all(p == "unknown" for p in strict))
        self.assertTrue(all(p == "id3" for p in loose))

    def test_single_identity_gallery_rejects_strangers(self):
        classifier = DDRCClassifier(feature_dim=DIM, atoms_per_class=1)
        alice = unit(1, seed=3)
        classifier.enroll_gallery(alice, torch.tensor([0]), ["Alice"])
        names, confidence, _ = classifier(unit(100, seed=4))
        self.assertTrue(all(name == "unknown" for name in names))
        self.assertLess(float(confidence.max()), 0.5)
        self.assertEqual(classifier(alice)[0], ["Alice"])

    def test_clear_matches_still_accepted_with_default_gates(self):
        classifier, prototypes = gallery()
        names, confidence, _ = classifier(with_cosine(prototypes[5], 0.7))
        self.assertTrue(all(name == "id5" for name in names))
        self.assertTrue(((confidence > 0) & (confidence <= 1)).all())

    def test_decide_matches_forward_and_instance_defaults(self):
        classifier, prototypes = gallery()
        probes = torch.cat([with_cosine(prototypes[1], 0.55, n=20), unit(20, seed=9)])
        names, confidence, residuals = classifier(probes)
        names2, confidence2 = classifier.decide(residuals)
        self.assertEqual(names, names2)
        self.assertTrue(torch.allclose(confidence, confidence2))
        classifier.residual_threshold = 0.2
        self.assertTrue(all(n == "unknown" for n in classifier(probes)[0]))

    def test_empty_gallery_is_unknown(self):
        classifier = DDRCClassifier(feature_dim=DIM)
        names, confidence, residuals = classifier(unit(3))
        self.assertEqual(names, ["unknown"] * 3)
        self.assertEqual(tuple(residuals.shape), (3, 0))


class GalleryEnrollmentTests(unittest.TestCase):
    def test_single_atom_uses_every_image_not_the_first(self):
        typical = unit(1, seed=11)[0]
        bad_first = with_cosine(typical, 0.1, n=1, seed=12)
        good = with_cosine(typical, 0.9, n=8, seed=13)
        classifier = DDRCClassifier(feature_dim=DIM, atoms_per_class=1)
        classifier.enroll_gallery(torch.cat([bad_first, good]), torch.zeros(9, dtype=torch.long), ["Bob"])
        atom = F.normalize(classifier.dictionary[:, 0].detach(), dim=0)
        self.assertGreater(float(atom @ typical), 0.9)
        # A new photo of Bob is recognised even though the first file was bad.
        self.assertEqual(classifier(with_cosine(typical, 0.8, n=1, seed=14))[0], ["Bob"])

    def test_multi_atom_clusters_capture_distinct_appearances(self):
        mode_a, mode_b = unit(2, seed=21)
        photos = torch.cat([with_cosine(mode_a, 0.95, n=6, seed=22), with_cosine(mode_b, 0.95, n=6, seed=23)])
        centroids = spherical_kmeans(photos, 2)
        best = (centroids @ torch.stack([mode_a, mode_b]).T).max(dim=0).values
        self.assertTrue((best > 0.95).all(), best)
        # Deterministic: identical gallery gives identical atoms.
        self.assertTrue(torch.equal(centroids, spherical_kmeans(photos, 2)))

    def test_fewer_images_than_atoms_repeats(self):
        classifier = DDRCClassifier(feature_dim=DIM, atoms_per_class=4)
        classifier.enroll_gallery(unit(2), torch.tensor([0, 0]), ["Carol"])
        self.assertEqual(tuple(classifier.dictionary.shape), (DIM, 4))


class CalibrationTests(unittest.TestCase):
    def test_sweep_recommends_pair_within_false_accept_budget(self):
        classifier, prototypes = gallery()
        known = torch.cat([with_cosine(prototypes[i], 0.65, n=10, seed=30 + i) for i in range(10)])
        known_names = [f"id{i}" for i in range(10) for _ in range(10)]
        # "Lookalike" strangers resemble one enrolled person weakly.
        unknown = torch.cat([with_cosine(prototypes[i], 0.35, n=10, seed=50 + i) for i in range(10)])
        rows = sweep_open_set(
            classifier,
            classifier.class_residuals(known),
            known_names,
            classifier.class_residuals(unknown),
        )
        best = recommend(rows, max_false_accept=0.0)
        self.assertIsNotNone(best)
        self.assertEqual(best["false_accept_rate"], 0.0)
        self.assertGreater(best["identification_rate"], 0.95)
        self.assertIsNone(recommend([{"residual_threshold": 0.5, "margin_threshold": 0.1,
                                      "false_accept_rate": 0.5}], 0.01))


class CropPreprocessingTests(unittest.TestCase):
    def test_box_resize_matches_gallery_transform_on_large_faces(self):
        # High-frequency texture exposes aliasing when a large face is shrunk.
        generator = torch.Generator().manual_seed(0)
        pixels = (torch.rand(400, 400, 3, generator=generator) * 255).to(torch.uint8).numpy()
        image = Image.fromarray(pixels, "RGB")
        gallery_tensor = get_default_transform((112, 112), is_train=False)(image)
        query = torch.from_numpy(pixels).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)
        aligned, used_landmarks = align_face_crop(query, None)
        aliased = F.interpolate(query, size=(112, 112), mode="bilinear", align_corners=False)
        self.assertFalse(used_landmarks)
        new_gap = (aligned[0] - gallery_tensor).abs().mean()
        old_gap = (aliased[0] - gallery_tensor).abs().mean()
        self.assertLess(float(new_gap), 0.5 * float(old_gap))


class VerificationEvalTests(unittest.TestCase):
    def test_parse_lfw_pairs_formats(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            pairs = os.path.join(root, "pairs.txt")
            with open(pairs, "w", encoding="utf-8") as handle:
                handle.write("1 1\nAlice 1 2\nAlice 1 Bob 3\n")
            paths, issame = parse_pairs(pairs, "lfw")
        self.assertEqual(issame, [True, False])
        self.assertEqual(paths[0], (os.path.join("lfw", "Alice", "Alice_0001.jpg"), os.path.join("lfw", "Alice", "Alice_0002.jpg")))
        self.assertEqual(paths[1][1], os.path.join("lfw", "Bob", "Bob_0003.jpg"))

    def test_kfold_accuracy_separable_and_chance(self):
        issame = torch.tensor([True, False] * 300)
        perfect = torch.where(issame, torch.tensor(0.7), torch.tensor(0.1))
        mean, std, threshold = kfold_verification_accuracy(perfect, issame)
        self.assertEqual(mean, 1.0)
        self.assertTrue(0.1 < threshold < 0.7)
        generator = torch.Generator().manual_seed(0)
        noise = torch.rand(600, generator=generator)
        chance, _, _ = kfold_verification_accuracy(noise, issame)
        self.assertLess(chance, 0.6)


if __name__ == "__main__":
    unittest.main()

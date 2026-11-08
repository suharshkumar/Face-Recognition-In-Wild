import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure workspace root is importable when this file is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import eval_widerface


class FakeDetector:
    """Return deterministic boxes without constructing or running a model."""

    def __init__(self, box=None):
        self.box = box or [0.0, 0.0, 10.0, 40.0]
        self.calls = 0

    def __call__(self, image_tensor, score_threshold):
        self.calls += 1
        return None


def make_evaluator(detector):
    """Build an evaluator shell without invoking detector/model initialization."""
    evaluator = object.__new__(eval_widerface.WiderFaceEvaluator)
    evaluator.device = 'cpu'
    evaluator.detector = detector
    return evaluator


class WiderFaceEvaluationRegressions(unittest.TestCase):
    def write_gt(self, path, entries):
        with open(path, 'w', encoding='utf-8') as gt_file:
            for image_key, box_line in entries:
                gt_file.write(f'{image_key}\n1\n{box_line}\n')

    def make_image(self, image_dir, image_key):
        image_path = os.path.join(image_dir, *image_key.split('/'))
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, 'wb') as image_file:
            image_file.write(b'fake image contents')
        return image_path

    def test_repository_gt_lookup_and_event_relative_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository_gt = os.path.join(
                temp_dir, 'WIDER_val', 'wider_face_split',
                'wider_face_val_bbx_gt.txt'
            )
            os.makedirs(os.path.dirname(repository_gt), exist_ok=True)
            self.write_gt(repository_gt, [
                ('event_a/shared.jpg', '0 0 10 60 0 0 0 0'),
                ('event_b/shared.jpg', '0 0 10 20 0 0 0 0'),
            ])

            with patch.object(eval_widerface, '__file__',
                              os.path.join(temp_dir, 'eval_widerface.py')):
                found = eval_widerface.find_widerface_gt_file()
            self.assertEqual(found, repository_gt)

            gt_dict = eval_widerface.load_widerface_gt(repository_gt)
            self.assertEqual(
                set(gt_dict), {'event_a/shared.jpg', 'event_b/shared.jpg'}
            )
            self.assertNotIn('shared.jpg', gt_dict)

    def test_missing_malformed_and_partial_ground_truth_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = os.path.join(temp_dir, 'images')
            first_image = self.make_image(image_dir, 'event_a/one.jpg')
            second_image = self.make_image(image_dir, 'event_b/two.jpg')

            with self.assertRaises(FileNotFoundError):
                eval_widerface.load_widerface_gt(
                    os.path.join(temp_dir, 'missing_gt.txt')
                )

            malformed_gt = os.path.join(temp_dir, 'malformed_gt.txt')
            with open(malformed_gt, 'w', encoding='utf-8') as gt_file:
                gt_file.write('event_a/one.jpg\n1\nnot-a-box\n')
            with self.assertRaises(eval_widerface.WiderFaceGroundTruthError):
                eval_widerface.load_widerface_gt(malformed_gt)

            partial_gt = os.path.join(temp_dir, 'partial_gt.txt')
            self.write_gt(partial_gt, [('event_a/one.jpg', '0 0 10 60')])
            gt_dict = eval_widerface.load_widerface_gt(partial_gt)
            with self.assertRaises(eval_widerface.WiderFaceGroundTruthError):
                eval_widerface.validate_widerface_gt_coverage(
                    gt_dict, [first_image, second_image], image_dir
                )

    def test_missing_ground_truth_stops_before_fake_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = os.path.join(temp_dir, 'images')
            self.make_image(image_dir, 'event/face.jpg')
            detector = FakeDetector()
            evaluator = make_evaluator(detector)

            with self.assertRaises(FileNotFoundError):
                evaluator.evaluate_and_benchmark(
                    image_dir=image_dir,
                    gt_file=os.path.join(temp_dir, 'missing_gt.txt'),
                    output_dir=os.path.join(temp_dir, 'results'),
                )
            self.assertEqual(detector.calls, 0)
            self.assertFalse(os.path.exists(
                os.path.join(temp_dir, 'results',
                             'widerface_industry_metrics.csv')
            ))

    def test_fake_predictions_use_gt_height_and_keep_event_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = os.path.join(temp_dir, 'images')
            self.make_image(image_dir, 'event_a/shared.jpg')
            self.make_image(image_dir, 'event_b/shared.jpg')
            gt_path = os.path.join(temp_dir, 'wider_face_val_bbx_gt.txt')
            self.write_gt(gt_path, [
                ('event_a/shared.jpg', '0 0 10 60 0 0 0 0'),
                ('event_b/shared.jpg', '0 0 10 60 0 0 0 0'),
            ])

            # The fake box is only 40 pixels tall but overlaps each 60-pixel
            # GT box. Difficulty must come from GT height.
            image_paths = [
                os.path.join(image_dir, 'event_a', 'shared.jpg'),
                os.path.join(image_dir, 'event_b', 'shared.jpg'),
            ]
            self.assertEqual(
                [eval_widerface.image_key_for_path(path, image_dir)
                 for path in image_paths],
                ['event_a/shared.jpg', 'event_b/shared.jpg'],
            )
            metrics = eval_widerface.compute_widerface_metrics(
                eval_widerface.load_widerface_gt(gt_path),
                [
                    {
                        'image': 'event_a/shared.jpg',
                        'box': [0.0, 0.0, 10.0, 40.0],
                        'score': 0.9,
                        'height': 40.0,
                    },
                    {
                        'image': 'event_b/shared.jpg',
                        'box': [0.0, 0.0, 10.0, 40.0],
                        'score': 0.8,
                        'height': 40.0,
                    },
                ],
            )
            self.assertGreater(metrics['easy_ap'], 0.99)
            self.assertEqual(metrics['medium_ap'], 0.0)
            self.assertEqual(metrics['hard_ap'], 0.0)


if __name__ == '__main__':
    unittest.main()

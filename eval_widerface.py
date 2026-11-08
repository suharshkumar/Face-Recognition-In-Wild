import os
import sys
import time
import math
import csv
import posixpath

try:
    import torch
except ImportError:  # Keep GT/AP regression helpers usable without model deps.
    torch = None

try:
    import numpy as np
except ImportError:  # Keep GT/AP regression helpers usable without model deps.
    np = None

try:
    from PIL import Image
except ImportError:  # Actual image evaluation reports the missing dependency.
    Image = None

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def calculate_iou(box1, box2):
    """
    Computes Intersection-over-Union (IoU) between box1 [x1, y1, x2, y2] and box2 [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area + 1e-7

    return float(inter_area / union_area)

def compute_ap_from_pr(recalls, precisions):
    """
    Computes exact Average Precision (AP) using continuous interpolated precision-recall curve.
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ap = sum(
        (mrec[index + 1] - mrec[index]) * mpre[index + 1]
        for index in range(len(mrec) - 1)
        if mrec[index + 1] != mrec[index]
    )
    return float(ap)


class WiderFaceGroundTruthError(ValueError):
    """Raised when WIDER FACE ground truth cannot be trusted for evaluation."""


def normalize_widerface_image_key(image_path: str) -> str:
    """Return a normalized event-relative WIDER FACE image key."""
    normalized = str(image_path).strip().replace('\\', '/')
    normalized = posixpath.normpath(normalized)
    while normalized.startswith('./'):
        normalized = normalized[2:]

    if (not normalized or normalized == '.' or normalized == '..' or
            normalized.startswith('../') or normalized.startswith('/')):
        raise ValueError(f"Image path must be event-relative: {image_path!r}")
    return normalized


def image_key_for_path(image_path: str, image_dir: str) -> str:
    """Map an image path under ``image_dir`` to its event-relative key."""
    relative_path = os.path.relpath(
        os.path.abspath(image_path), os.path.abspath(image_dir)
    )
    return normalize_widerface_image_key(relative_path)


def find_widerface_gt_file(gt_file: str = '', image_dir: str = '') -> str:
    """Find the supplied or repository-local WIDER FACE validation GT file."""
    if gt_file:
        resolved_gt_file = os.path.abspath(gt_file)
        if not os.path.isfile(resolved_gt_file):
            raise FileNotFoundError(
                f"WIDER FACE ground truth file does not exist: {gt_file}"
            )
        return resolved_gt_file

    repository_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(repository_root, 'WIDER_val', 'wider_face_split',
                     'wider_face_val_bbx_gt.txt'),
        os.path.join(repository_root, 'WIDER_val',
                     'wider_face_val_bbx_gt.txt'),
        os.path.join(repository_root, 'wider_face_split',
                     'wider_face_val_bbx_gt.txt'),
        os.path.join(repository_root, 'wider_face_val_bbx_gt.txt'),
    ]
    if image_dir:
        candidates.extend([
            os.path.join(os.path.abspath(image_dir),
                         'wider_face_val_bbx_gt.txt'),
            os.path.join(os.path.abspath(image_dir), 'wider_face_split',
                         'wider_face_val_bbx_gt.txt'),
        ])

    checked = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in checked:
            continue
        checked.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        'WIDER FACE ground truth file was not found. Checked: '
        + ', '.join(checked)
    )


def load_widerface_gt(gt_txt_path: str):
    """
    Parses official WIDER FACE ground truth annotation file (wider_face_val_bbx_gt.txt).
    Returns a dict keyed by event-relative image path. Every annotation record is
    validated so a partial file cannot silently turn into an evaluation result.
    """
    if not os.path.isfile(gt_txt_path):
        raise FileNotFoundError(
            f"WIDER FACE ground truth file does not exist: {gt_txt_path}"
        )

    with open(gt_txt_path, 'r', encoding='utf-8') as f:
        lines = [
            (line_number, line.strip())
            for line_number, line in enumerate(f, 1)
            if line.strip()
        ]

    if not lines:
        raise WiderFaceGroundTruthError(
            f"WIDER FACE ground truth file is empty: {gt_txt_path}"
        )

    gt_dict = {}
    idx = 0
    while idx < len(lines):
        image_line_number, image_line = lines[idx]
        try:
            image_key = normalize_widerface_image_key(image_line)
        except ValueError as exc:
            raise WiderFaceGroundTruthError(
                f"Invalid image path on line {image_line_number}: {image_line!r}"
            ) from exc
        if not image_key.lower().endswith(VALID_IMAGE_EXTENSIONS):
            raise WiderFaceGroundTruthError(
                f"Invalid WIDER FACE image path on line {image_line_number}: "
                f"{image_line!r}"
            )
        if image_key in gt_dict:
            raise WiderFaceGroundTruthError(
                f"Duplicate WIDER FACE image entry: {image_key!r}"
            )
        idx += 1

        if idx >= len(lines):
            raise WiderFaceGroundTruthError(
                f"Missing face count after {image_key!r}"
            )
        count_line_number, count_line = lines[idx]
        if not count_line.isdigit():
            raise WiderFaceGroundTruthError(
                f"Invalid face count on line {count_line_number} for "
                f"{image_key!r}: {count_line!r}"
            )
        num_boxes = int(count_line)
        idx += 1

        box_list = []
        for box_number in range(num_boxes):
            if idx >= len(lines):
                raise WiderFaceGroundTruthError(
                    f"Partial annotations for {image_key!r}: expected "
                    f"{num_boxes} boxes, found {box_number}"
                )

            box_line_number, box_line = lines[idx]
            parts = box_line.split()
            if len(parts) < 4:
                raise WiderFaceGroundTruthError(
                    f"Malformed box on line {box_line_number} for "
                    f"{image_key!r}: expected at least x, y, w, h"
                )
            try:
                x, y, w, h = (float(parts[position]) for position in range(4))
            except ValueError as exc:
                raise WiderFaceGroundTruthError(
                    f"Malformed box on line {box_line_number} for "
                    f"{image_key!r}: {box_line!r}"
                ) from exc
            try:
                optional_values = [float(value) for value in parts[4:]]
            except ValueError as exc:
                raise WiderFaceGroundTruthError(
                    f"Malformed annotation fields on line {box_line_number} "
                    f"for {image_key!r}: {box_line!r}"
                ) from exc
            if not all(math.isfinite(value) for value in optional_values):
                raise WiderFaceGroundTruthError(
                    f"Invalid annotation fields on line {box_line_number} "
                    f"for {image_key!r}: {box_line!r}"
                )

            # Official WIDER annotations can contain an invalid zero-size
            # placeholder. Preserve the image entry but exclude that box.
            try:
                invalid_flag = int(float(parts[7])) if len(parts) >= 8 else 0
            except ValueError as exc:
                raise WiderFaceGroundTruthError(
                    f"Malformed invalid flag on line {box_line_number} for "
                    f"{image_key!r}: {parts[7]!r}"
                ) from exc
            if not all(math.isfinite(value) for value in (x, y, w, h)):
                raise WiderFaceGroundTruthError(
                    f"Invalid box dimensions on line {box_line_number} for "
                    f"{image_key!r}: {box_line!r}"
                )
            if w <= 0 or h <= 0:
                # The released WIDER annotations contain zero-size invalid
                # placeholders with inconsistent invalid flags. They cannot
                # match a prediction, so ignore the row after validating it.
                idx += 1
                continue

            occ = 0
            # WIDER fields after x,y,w,h are blur, expression, illumination,
            # invalid, occlusion, pose; occlusion is therefore zero-based 8.
            if len(parts) >= 9:
                try:
                    occ = int(parts[8])
                except ValueError as exc:
                    raise WiderFaceGroundTruthError(
                        f"Malformed occlusion value on line {box_line_number} "
                        f"for {image_key!r}: {parts[8]!r}"
                    ) from exc
            box_list.append({
                'box': [x, y, x + w, y + h],
                'h': h,
                'occlusion': occ,
            })
            idx += 1

        gt_dict[image_key] = box_list

    return gt_dict


def validate_widerface_gt_coverage(gt_dict, image_files, image_dir):
    """Reject a GT file that does not cover every image being evaluated."""
    image_keys = {
        image_key_for_path(image_path, image_dir) for image_path in image_files
    }
    missing_keys = sorted(image_keys - set(gt_dict))
    if missing_keys:
        preview = ', '.join(repr(key) for key in missing_keys[:5])
        if len(missing_keys) > 5:
            preview += ', ...'
        raise WiderFaceGroundTruthError(
            'WIDER FACE ground truth is partial; missing annotations for: '
            + preview
        )


def _compute_split_ap(gt_dict, pred_records, min_height, max_height):
    split_gt = {
        image_key: [gt for gt in image_gt
                    if gt['h'] >= min_height and
                    (max_height is None or gt['h'] < max_height)]
        for image_key, image_gt in gt_dict.items()
    }
    total_gt = sum(len(image_gt) for image_gt in split_gt.values())
    if total_gt == 0:
        return 0.0

    matched_gt = {
        image_key: [False] * len(image_gt)
        for image_key, image_gt in split_gt.items()
    }
    true_positives = []
    false_positives = []

    for pred in sorted(pred_records, key=lambda record: record['score'], reverse=True):
        image_key = normalize_widerface_image_key(pred['image'])
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_item in enumerate(split_gt.get(image_key, [])):
            if matched_gt[image_key][gt_idx]:
                continue
            iou = calculate_iou(pred['box'], gt_item['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= 0.50 and best_gt_idx >= 0:
            matched_gt[image_key][best_gt_idx] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    cumulative_tp = []
    cumulative_fp = []
    running_tp = 0
    running_fp = 0
    for true_positive, false_positive in zip(true_positives, false_positives):
        running_tp += true_positive
        running_fp += false_positive
        cumulative_tp.append(running_tp)
        cumulative_fp.append(running_fp)
    recalls = [true_positive / total_gt for true_positive in cumulative_tp]
    precisions = [
        true_positive / (true_positive + false_positive + 1e-7)
        for true_positive, false_positive in zip(cumulative_tp, cumulative_fp)
    ]
    return compute_ap_from_pr(recalls, precisions)


def compute_widerface_metrics(gt_dict, pred_records):
    """Compute AP splits using each matched GT box's height."""
    easy_ap = _compute_split_ap(gt_dict, pred_records, 50, None)
    medium_ap = _compute_split_ap(gt_dict, pred_records, 30, 50)
    hard_ap = _compute_split_ap(gt_dict, pred_records, 0, 30)
    return {
        'easy_ap': easy_ap,
        'medium_ap': medium_ap,
        'hard_ap': hard_ap,
        'overall_map': float((easy_ap + medium_ap + hard_ap) / 3.0),
    }

class WiderFaceEvaluator:
    """
    Ground-truth-backed WIDER FACE evaluator.

    This reference evaluator uses the repository's annotation heights for
    coarse Easy/Medium/Hard diagnostics. For official leaderboard numbers,
    use the official WIDER FACE evaluation package and split lists.
    """
    def __init__(self, weights_dir: str = "./weights", device: str = None):
        if torch is None:
            raise ImportError('WIDER FACE model evaluation requires torch.')
        from models.detector import FasterRCNNFaceDetector

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"========================================================================")
        print(f"--- OccuPose-BroadDictNet WIDER FACE Benchmark Evaluator ---")
        print(f"Device: {self.device} | Weights Directory: {weights_dir}")
        print(f"========================================================================\n")
        
        self.detector = FasterRCNNFaceDetector(weights_dir=weights_dir).to(self.device)
        self.detector.eval()

    def evaluate_and_benchmark(self, image_dir: str = "./Test_dataser", gt_file: str = "", output_dir: str = "./results/widerface_eval", score_threshold: float = 0.3):
        """
        Runs model inference across images, matches predictions against ground truth boxes (IoU >= 0.50),
        and computes exact Average Precision (AP) for Easy, Medium, and Hard splits.
        """
        os.makedirs(output_dir, exist_ok=True)
        txt_output_dir = os.path.join(output_dir, "widerface_txt_predictions")
        os.makedirs(txt_output_dir, exist_ok=True)

        image_files = []
        if os.path.exists(image_dir):
            for root, _, files in os.walk(image_dir):
                for f in sorted(files):
                    if f.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        image_files.append(os.path.join(root, f))
        image_files.sort()

        if len(image_files) == 0:
            print(f"❌ Error: No evaluation images found in '{image_dir}'")
            return

        gt_file = find_widerface_gt_file(gt_file, image_dir)
        gt_dict = load_widerface_gt(gt_file)
        validate_widerface_gt_coverage(gt_dict, image_files, image_dir)
        print(f"✓ Ground Truth File Loaded: '{gt_file}' ({len(gt_dict)} annotated images)")

        if torch is None or np is None or Image is None:
            raise ImportError(
                'WIDER FACE image evaluation requires torch, numpy, and Pillow.'
            )

        print(f"Processing {len(image_files)} benchmark images in '{image_dir}'...")
        start_time = time.time()

        all_pred_records = []
        for idx, img_path in enumerate(image_files, 1):
            image_key = image_key_for_path(img_path, image_dir)
            fname = os.path.basename(image_key)
            try:
                pil_img = Image.open(img_path).convert('RGB')
                orig_w, orig_h = pil_img.size
                np_img = np.array(pil_img).transpose(2, 0, 1)
                img_tensor = (torch.tensor(np_img).unsqueeze(0).float() / 127.5) - 1.0
                img_tensor = img_tensor.to(self.device)
            except Exception:
                continue

            with torch.no_grad():
                boxes, scores, _ = self.detector(img_tensor, score_threshold=score_threshold)

            # Export official WIDER FACE TXT format prediction file
            rel_dir = os.path.dirname(image_key) or '.'
            event_txt_dir = os.path.join(txt_output_dir, rel_dir)
            os.makedirs(event_txt_dir, exist_ok=True)

            txt_path = os.path.join(event_txt_dir, os.path.splitext(fname)[0] + ".txt")
            with open(txt_path, 'w', encoding='utf-8') as f_txt:
                f_txt.write(f"{image_key}\n")
                f_txt.write(f"{boxes.size(0)}\n")
                for b_i in range(boxes.size(0)):
                    box = boxes[b_i].cpu().numpy()
                    score = float(scores[b_i].item())
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    w, h = max(0, x2 - x1), max(0, y2 - y1)
                    f_txt.write(f"{int(x1)} {int(y1)} {int(w)} {int(h)} {score:.4f}\n")
                    all_pred_records.append({
                        'image': image_key,
                        'box': [x1, y1, x2, y2],
                        'score': score,
                        'height': h
                    })

        elapsed = time.time() - start_time

        metrics = compute_widerface_metrics(gt_dict, all_pred_records)
        easy_ap = metrics['easy_ap']
        medium_ap = metrics['medium_ap']
        hard_ap = metrics['hard_ap']
        overall_map = metrics['overall_map']
        status_label = "EXACT MATCH (GT Ground-Truth)"

        # Format & Output Exact Metrics Table
        print(f"\n========================================================================")
        print(f"   WIDER FACE BENCHMARK EVALUATION RESULTS ({status_label})")
        print(f"========================================================================")
        print(f" Metric Split                 | Average Precision (AP@IoU=0.50) | Status")
        print(f"------------------------------+----------------------------------+--------")
        print(f" WIDER FACE (Easy AP)         | {easy_ap*100:6.2f}%                           | EVALUATED")
        print(f" WIDER FACE (Medium AP)       | {medium_ap*100:6.2f}%                           | EVALUATED")
        print(f" WIDER FACE (Hard AP)         | {hard_ap*100:6.2f}%                           | EVALUATED")
        print(f"------------------------------+----------------------------------+--------")
        print(f" Overall Mean AP (mAP@0.50)   | {overall_map*100:6.2f}%                           | PASSED")
        print(f" Total Evaluation Time        | {elapsed:6.2f}s ({elapsed/max(1, len(image_files)):.2f}s/img)            | FAST")
        print(f"========================================================================\n")

        # Save Metrics CSV Report
        metrics_csv_path = os.path.join(output_dir, "widerface_industry_metrics.csv")
        with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(['Split', 'AP_Score', 'Metric_IoU', 'GroundTruth_Loaded'])
            writer.writerow(['WIDER_Easy_AP', f"{easy_ap:.4f}", 'IoU=0.50', 'True'])
            writer.writerow(['WIDER_Medium_AP', f"{medium_ap:.4f}", 'IoU=0.50', 'True'])
            writer.writerow(['WIDER_Hard_AP', f"{hard_ap:.4f}", 'IoU=0.50', 'True'])
            writer.writerow(['Overall_mAP', f"{overall_map:.4f}", 'IoU=0.50', 'True'])

        print(f"✓ Saved WIDER FACE metrics CSV report to : {metrics_csv_path}")
        print(f"✓ Saved WIDER FACE format TXT prediction files to  : {txt_output_dir}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Single-Step WIDER FACE Benchmark Evaluator")
    parser.add_argument("--image_dir", type=str, default="./Test_dataser", help="Path to evaluation images folder")
    parser.add_argument("--gt_file", type=str, default="", help="Path to ground truth wider_face_val_bbx_gt.txt file")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Path to weights folder")
    parser.add_argument("--output_dir", type=str, default="./results/widerface_eval", help="Directory to save evaluation metrics")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Confidence cutoff for face proposals")
    args = parser.parse_args()

    evaluator = WiderFaceEvaluator(weights_dir=args.weights_dir)
    evaluator.evaluate_and_benchmark(image_dir=args.image_dir, gt_file=args.gt_file, output_dir=args.output_dir, score_threshold=args.score_threshold)

if __name__ == "__main__":
    main()

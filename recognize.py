"""Improved recognition with our own trained weights only (no retraining).

Phase 0 boxes are cropped the same way for gallery and query images, embedded
by our IResNet-100 (with horizontal-flip averaging), and matched to the
gallery by cosine similarity. Two settings are chosen on the gallery by
leave-one-out, without touching the test set:

* ``scale``: how far each detector box is widened before resizing to 112x112,
  so test crops look like the LFW-style training crops;
* ``alpha``: weight-space interpolation between our Phase 3 and Phase 1
  checkpoints, theta = (1 - alpha) * theta_phase3 + alpha * theta_phase1.

    python recognize.py --gallery_dir G --calibrate
    python recognize.py --gallery_dir G --input_dir IMGS --truth truth.csv
"""

import argparse, csv, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

from models.backbone import iresnet100
from models.detector import TorchvisionFaceDetector

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCALES = [1.0, 1.4, 1.8, 2.2]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_backbone_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("backbone_state", checkpoint.get("backbone"))
    return {k.removeprefix("module."): v for k, v in state.items()}


def crop_face(rgb, box, scale):
    """Square crop around a detector box, widened by ``scale``, resized to 112x112."""
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale
    pad = int(side)
    padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    a, b = int(round(cx - side / 2)) + pad, int(round(cy - side / 2)) + pad
    crop = padded[b:b + int(round(side)), a:a + int(round(side))]
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class Recognizer:
    def __init__(self, weights_dir, det_threshold=0.3):
        w = Path(weights_dir)
        self.detector = TorchvisionFaceDetector(checkpoint_path=str(w / "face_detector.pt")).eval()
        self.det_threshold = det_threshold
        self.phase3 = load_backbone_state(w / "phase3_adapted_backbone.pt")
        phase1_path = w / "recognition_backbone.pt"
        self.phase1 = load_backbone_state(phase1_path) if phase1_path.is_file() else None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.backbone = iresnet100().eval().to(self.device)
        self.set_alpha(0.0)

    def set_alpha(self, alpha):
        """Load theta = (1 - alpha) * phase3 + alpha * phase1 into the backbone."""
        if self.phase1 is None or alpha == 0.0:
            state = self.phase3
        else:
            state = {k: ((1 - alpha) * v + alpha * self.phase1[k]) if v.is_floating_point() else v
                     for k, v in self.phase3.items()}
        self.backbone.load_state_dict(state)
        self.alpha = alpha

    @torch.no_grad()
    def faces(self, rgb):
        """Phase 0 detections as a list of (box, score)."""
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0)
        boxes, scores, _ = self.detector(tensor, score_threshold=self.det_threshold)
        return [(b, float(s)) for b, s in zip(boxes.numpy(), scores.numpy())]

    @torch.no_grad()
    def embed(self, crops):
        """Unit-length embeddings of 112x112 RGB crops, averaged with their mirror images."""
        x = torch.from_numpy(np.stack(crops).astype(np.float32)).permute(0, 3, 1, 2).div(127.5).sub(1)
        out = []
        for i in range(0, len(x), 32):
            batch = x[i:i + 32].to(self.device)
            e = self.backbone(batch) + self.backbone(torch.flip(batch, dims=[3]))
            out.append(torch.nn.functional.normalize(e, dim=1).cpu())
        return torch.cat(out).numpy()


def gallery_faces(rec, gallery_dir):
    """Return names, labels and, per image, the largest Phase 0 box (None for tight crops).

    Detections are cached in ``<gallery_dir>/.phase0_boxes.json`` so later runs skip Phase 0.
    """
    cache_path = Path(gallery_dir) / ".phase0_boxes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    names, labels, items = [], [], []
    for person in sorted(p for p in Path(gallery_dir).iterdir() if p.is_dir()):
        name = person.name.replace("_", " ")
        for path in sorted(p for p in person.iterdir() if p.suffix.lower() in IMAGE_EXT):
            rgb = read_rgb(path)
            if rgb is None:
                continue
            key = f"{person.name}/{path.name}"
            if key not in cache:
                found = rec.faces(rgb)
                largest = max(found, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))[0] if found else None
                cache[key] = None if largest is None else [float(v) for v in largest]
                if len(cache) % 100 == 0:
                    cache_path.write_text(json.dumps(cache))
                    print(f"  Phase 0 on gallery: {len(cache)} images", flush=True)
            box = None if cache[key] is None else np.array(cache[key], np.float32)
            if name not in names:
                names.append(name)
            labels.append(names.index(name))
            items.append((rgb, box))
    cache_path.write_text(json.dumps(cache))
    print(f"Gallery: {len(items)} images of {len(names)} people "
          f"({sum(b is None for _, b in items)} without a detected face)", flush=True)
    return names, np.array(labels), items


def embed_gallery(rec, items, scale):
    crops = [crop_face(rgb, box, scale) if box is not None
             else cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA) for rgb, box in items]
    return rec.embed(crops)


def class_scores(sim, labels, n_classes):
    scores = np.full((sim.shape[0], n_classes), -1.0, np.float32)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            scores[:, c] = sim[:, mask].max(axis=1)
    return scores


def leave_one_out(names, labels, emb):
    """Rank-1 accuracy, 1%-FAR threshold and TAR when each gallery image is a probe."""
    sim = emb @ emb.T
    genuine, impostor, correct = [], [], 0
    for i in range(len(labels)):
        others = np.arange(len(labels)) != i
        if not (labels[others] == labels[i]).any():
            continue
        scores = class_scores(sim[i:i + 1, others], labels[others], len(names))[0]
        correct += int(scores.argmax() == labels[i])
        genuine.append(scores[labels[i]])
        impostor.append(np.delete(scores, labels[i]).max())
    if not genuine:
        raise SystemExit("Calibration needs at least two gallery images for some person")
    threshold = float(np.quantile(impostor, 0.99))
    return correct / len(genuine), threshold, float(np.mean(np.array(genuine) >= threshold)), len(genuine)


def calibrate(rec, names, labels, items):
    """Pick the crop scale at alpha=0, then the blend weight at that scale."""
    results = {}

    def trial(alpha, scale):
        if (alpha, scale) not in results:
            rec.set_alpha(alpha)
            acc, threshold, tar, probes = leave_one_out(names, labels, embed_gallery(rec, items, scale))
            print(f"alpha={alpha:.2f} scale={scale:.1f}  rank-1={100 * acc:.1f}%  "
                  f"threshold@1%FAR={threshold:.3f}  TAR={100 * tar:.1f}%  (probes={probes})", flush=True)
            results[(alpha, scale)] = (acc, tar, threshold)
        return results[(alpha, scale)]

    scale = max(SCALES, key=lambda sc: trial(0.0, sc)[:2])
    alphas = ALPHAS if rec.phase1 is not None else [0.0]
    alpha = max(alphas, key=lambda a: trial(a, scale)[:2])
    threshold = results[(alpha, scale)][2]
    print(f"Chosen alpha={alpha:.2f}, scale={scale:.1f}, threshold={threshold:.3f}")
    return alpha, scale, threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery_dir", required=True)
    ap.add_argument("--input_dir")
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--output_dir", default="recognition_results")
    ap.add_argument("--alpha", type=float, help="Phase 1 weight in the blend (default: calibrated)")
    ap.add_argument("--scale", type=float, help="box widening factor (default: calibrated)")
    ap.add_argument("--threshold", type=float, help="unknown threshold (default: calibrated)")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--det_threshold", type=float, default=0.3)
    ap.add_argument("--truth", help="CSV with columns image,names (names separated by ';')")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    rec = Recognizer(args.weights_dir, args.det_threshold)
    names, labels, items = gallery_faces(rec, args.gallery_dir)
    alpha, scale, threshold = args.alpha, args.scale, args.threshold
    if None in (alpha, scale, threshold):
        c_alpha, c_scale, c_threshold = calibrate(rec, names, labels, items)
        alpha = c_alpha if alpha is None else alpha
        scale = c_scale if scale is None else scale
        threshold = c_threshold if threshold is None else threshold
    if args.calibrate or not args.input_dir:
        return
    rec.set_alpha(alpha)
    gallery = embed_gallery(rec, items, scale)

    out = Path(args.output_dir)
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    truth = {}
    if args.truth:
        with open(args.truth) as fh:
            for row in csv.DictReader(fh):
                truth[row["image"]] = {n.strip().lower() for n in row["names"].split(";") if n.strip()}
    tp = fp = fn = 0
    with open(out / "predictions.jsonl", "w") as log:
        for path in sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in IMAGE_EXT):
            rgb = read_rgb(path)
            if rgb is None:
                continue
            found = rec.faces(rgb)
            records = []
            if found:
                scores = class_scores(rec.embed([crop_face(rgb, b, scale) for b, _ in found]) @ gallery.T,
                                      labels, len(names))
                for (box, det), row in zip(found, scores):
                    order = np.argsort(-row)
                    top, second = row[order[0]], (row[order[1]] if len(row) > 1 else -1.0)
                    name = names[order[0]] if top >= threshold and top - second >= args.margin else "unknown"
                    records.append({"box": [int(round(v)) for v in box], "identity": name,
                                    "similarity": round(float(top), 4), "detector_score": round(det, 4)})
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for r in records:
                x1, y1, x2, y2 = r["box"]
                color = (0, 200, 0) if r["identity"] != "unknown" else (0, 165, 255)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(bgr, f'{r["identity"]} ({r["similarity"]:.2f})', (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.imwrite(str(out / "annotated" / path.name), bgr)
            log.write(json.dumps({"image": path.name, "predictions": records}) + "\n")
            predicted = {r["identity"].lower() for r in records if r["identity"] != "unknown"}
            if path.name in truth:
                tp += len(predicted & truth[path.name])
                fp += len(predicted - truth[path.name])
                fn += len(truth[path.name] - predicted)
            print(f"{path.name}: {len(records)} faces, named: {sorted(predicted)}", flush=True)
    if truth:
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
        print(f"Identity precision {100 * p:.1f}%  recall {100 * r:.1f}%  "
              f"F1 {100 * 2 * p * r / max(1e-9, p + r):.1f}%  (TP={tp} FP={fp} FN={fn})")
    print(f"Wrote {out / 'predictions.jsonl'} (alpha={alpha}, scale={scale}, threshold={threshold:.3f})")


if __name__ == "__main__":
    sys.exit(main())

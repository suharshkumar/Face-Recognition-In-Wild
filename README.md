# Face Recognition in the Wild

CS6350 term project (TPA 13) by **Suharsh Kumar** (BE22B037) and **Akshat Porwal** (NA22B025).
Original development repository: [akshat-1/Face-Recognition-In-Wild](https://github.com/akshat-1/Face-Recognition-In-Wild).

The system finds every face in an ordinary photograph, names each person from an
enrolled gallery or returns `unknown`, and describes each face with 40 attributes.

```text
photo ─► Phase 0: Faster R-CNN detector ─► 112×112 crops ─┬─► Phase 1/3: IResNet-100 ─► 512-D vector ─► DDRC gallery match ─► name / unknown
                                                          └─► Phase 2: attribute network ─► 40 attributes + occlusion flag
```

## The four training phases

| Phase | Model | Training data | Loss | Checkpoint |
|---|---|---|---|---|
| 0 | Faster R-CNN, ResNet-50 + FPN (COCO init), 2 classes | 14,869 images with face boxes | RPN BCE + Smooth-L1, head CE + Smooth-L1 | `face_detector.pt` |
| 1 | IResNet-100 → 512-D unit vector | 23,994 crops of 5,931 people | ArcFace margin + CurricularFace + BroadFace queue | `recognition_backbone.pt` |
| 2 | Dual-path attribute CNN (global + upper face) | CelebA, 40 attributes | BCE over 40 attributes | `phase2_anet_attributes.pt` |
| 3 | IResNet-100 fine-tuned on pseudo-labels | 378,818 detected unlabelled crops → 127,849 pseudo-labelled | labelled loss + 0.5 · detector score · pseudo loss | `phase3_adapted_backbone.pt` |

At inference the Phase 3 backbone is used when present, otherwise the Phase 1
backbone. The DDRC classifier reconstructs a query from gallery embeddings and
accepts the best identity only if it passes the residual and margin gates.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the checkpoints in `weights/` (they are not stored in this repository).
A gallery is a folder with one sub-folder of face images per person.

## Run inference

```bash
python inference.py \
  --input_dir path/to/images \
  --weights_dir weights \
  --gallery_dir path/to/gallery \
  --output_dir inference_results \
  --score_threshold 0.3
```

This writes `inference_results/predictions.jsonl` (one record per image) and
annotated copies of the images. `--score_threshold` only controls the detector:
around 0.05 favours recall, 0.3 to 0.5 removes most false boxes.

Detector-only check:

```bash
python detector_inference.py --image path/to/image.jpg --weights_dir weights --overlay_dir detector_results
```

## Train

```bash
python train.py \
  --bb_dir Face_Dataset/bb_label \
  --data_dir Face_Dataset/name_label \
  --celeba_dir Face_Dataset/celeba \
  --unlabeled_dir Face_Dataset/unlabeled \
  --checkpoint_dir weights --fp16
```

The script runs Phase 0 → 1 → 2 → 3 and skips any phase whose checkpoint already
exists (use `--retrain_phase1`, `--retrain_all`, etc. to force retraining).
Multi-GPU training uses `torchrun`.

## Improved recognition (`recognize.py`)

Uses only our own trained weights (no retraining). Gallery and test images go
through the same Phase 0 detection and cropping, each crop is embedded by our
IResNet-100 with horizontal-flip averaging, and identities are matched by
cosine similarity. Two settings are chosen on the gallery by leave-one-out:
the box widening `scale` (so crops resemble the LFW-style training images) and
`alpha`, a weight-space interpolation between our checkpoints,
`theta = (1 - alpha) * phase3 + alpha * phase1`.

```bash
python recognize.py --gallery_dir path/to/gallery --calibrate
python recognize.py --gallery_dir path/to/gallery --input_dir path/to/test_images \
  --truth truth.csv --output_dir recognition_results
```

`truth.csv` (optional) has columns `image,names`, with names separated by `;`
and spelled like the gallery folder names.

## Evaluation results

Full pipeline, detector threshold 0.05, IoU ≥ 0.5 (details in the project report).

| Benchmark | Precision | Recall | F1 | AP@0.5 |
|---|---|---|---|---|
| WIDER FACE val | 80.18% | 35.47% | 49.18% | 34.49% |
| UFDD val | 45.57% | 46.48% | 46.02% | 40.59% |
| AFW | 72.86% | 98.08% | 83.61% | 97.63% |

`eval_widerface.py` re-computes the WIDER FACE numbers from
`wider_face_split/wider_face_val_bbx_gt.txt`. LFW is part of the training data,
so it must not be used to measure recognition accuracy.

## Known limitations

- Recall is low for faces smaller than 30 pixels.
- The spatial occlusion head has no training labels, so the occlusion flag is not reliable.
- The open-set thresholds are not calibrated on held-out identities.

## Repository layout

```text
train.py                 trains all four phases
inference.py             end-to-end inference on images or folders
detector_inference.py    detector-only diagnostic
eval_widerface.py        WIDER FACE evaluation
eval_lfw.py, calibrate_open_set.py   verification and threshold tools
config.py                all hyper-parameters
dataset.py               data loaders for every phase
models/                  detector, IResNet backbone, attribute net, GCN, DDRC
losses/                  CurricularFace and BroadFace losses
pipeline/                alignment and the end-to-end inference pipeline
tests/                   unit tests (python -m unittest discover -s tests)
```

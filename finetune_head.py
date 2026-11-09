"""Fine-tune the embedding head of our IResNet-100 on a new gallery (CPU/M1 friendly).

The convolutional body (stem + four residual stages) stays frozen; its 7x7x512
feature maps are computed once and cached. Only the head (BN -> FC -> BN, about
12.8M parameters) is trained with the same CurricularFace loss as Phase 1. One
image per identity is held out to measure rank-1 identification before and after.

    python finetune_head.py --gallery_dir Face_Dataset/celeb_gallery --base weights/recognition_backbone.pt
"""

import argparse, copy, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from losses.curricular_loss import CurricularFaceLoss
from recognize import Recognizer, crop_face, gallery_faces, load_backbone_state
import cv2


def body_features(backbone, crops, device):
    """Frozen stem + residual stages -> (N, 2, 512, 7, 7) for each crop and its mirror."""
    x = torch.from_numpy(np.stack(crops).astype(np.float32)).permute(0, 3, 1, 2).div(127.5).sub(1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            out = []
            for batch in (x[i:i + 32], torch.flip(x[i:i + 32], dims=[3])):
                h = backbone.prelu(backbone.bn1(backbone.conv1(batch.to(device))))
                h = backbone.layer4(backbone.layer3(backbone.layer2(backbone.layer1(h))))
                out.append(h.half().cpu())
            feats.append(torch.stack(out, dim=1))
    return torch.cat(feats)


def head(backbone, h):
    """The trainable part: bn2 -> dropout -> flatten -> fc -> features BN -> L2 normalise."""
    h = backbone.bn2(h)
    if backbone.dropout is not None:
        h = backbone.dropout(h)
    return F.normalize(backbone.features(backbone.fc(torch.flatten(h, 1))), dim=1)


def rank1(backbone, feats, labels, train_idx, val_idx, device):
    backbone.eval()
    with torch.no_grad():
        emb = []
        for i in range(0, len(feats), 256):
            f = feats[i:i + 256].float().to(device)
            emb.append(F.normalize(head(backbone, f[:, 0]) + head(backbone, f[:, 1]), dim=1).cpu())
        emb = torch.cat(emb)
    sim = emb[val_idx] @ emb[train_idx].T
    predicted = labels[train_idx][sim.argmax(dim=1)]
    return float((predicted == labels[val_idx]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery_dir", required=True)
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--base", default="weights/recognition_backbone.pt")
    ap.add_argument("--scale", type=float, default=2.2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output", default="weights/finetuned_backbone.pt")
    args = ap.parse_args()
    random.seed(0); torch.manual_seed(0)

    rec = Recognizer(args.weights_dir)
    backbone = rec.backbone
    backbone.load_state_dict(load_backbone_state(args.base))
    device = rec.device
    names, labels, items = gallery_faces(rec, args.gallery_dir)
    crops = [crop_face(rgb, box, args.scale) if box is not None
             else cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_AREA) for rgb, box in items]
    print("Computing frozen body features...", flush=True)
    feats = body_features(backbone, crops, device)
    labels = torch.from_numpy(labels)

    # Hold out one image per identity that has at least three images.
    val_idx, train_idx = [], []
    for c in range(len(names)):
        idx = (labels == c).nonzero().flatten().tolist()
        random.shuffle(idx)
        if len(idx) >= 3:
            val_idx.append(idx[0]); train_idx.extend(idx[1:])
        else:
            train_idx.extend(idx)
    val_idx, train_idx = torch.tensor(val_idx), torch.tensor(train_idx)
    print(f"{len(names)} people, {len(train_idx)} training images, {len(val_idx)} held-out images")
    before = rank1(backbone, feats, labels, train_idx, val_idx, device)
    print(f"Held-out rank-1 before fine-tuning: {100 * before:.1f}%", flush=True)

    loss_fn = CurricularFaceLoss(512, len(names)).to(device)
    params = [p for m in (backbone.bn2, backbone.fc, backbone.features) for p in m.parameters()]
    optimizer = torch.optim.Adam(params + list(loss_fn.parameters()), lr=args.lr)
    best, best_state = before, copy.deepcopy(backbone.state_dict())
    for epoch in range(1, args.epochs + 1):
        backbone.train(); loss_fn.train()
        order = train_idx[torch.randperm(len(train_idx))]
        total = 0.0
        for i in range(0, len(order), 128):
            batch = order[i:i + 128]
            if len(batch) < 2:
                continue
            flip = torch.randint(0, 2, (len(batch),))
            f = feats[batch, flip].float().to(device)
            loss = loss_fn(head(backbone, f), labels[batch].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss) * len(batch)
        acc = rank1(backbone, feats, labels, train_idx, val_idx, device)
        print(f"epoch {epoch:2d}  loss {total / len(order):.3f}  held-out rank-1 {100 * acc:.1f}%", flush=True)
        if acc > best:
            best, best_state = acc, copy.deepcopy(backbone.state_dict())
    print(f"Best held-out rank-1: {100 * best:.1f}% (was {100 * before:.1f}%)")
    torch.save({"schema_version": 2, "kind": "recognition_backbone",
                "model_config": {"backbone_type": "iresnet100", "embedding_dim": 512},
                "input_size": [112, 112], "recognition_alignment": "resize",
                "finetuned_from": args.base, "crop_scale": args.scale,
                "backbone_state": {k: v.cpu() for k, v in best_state.items()}}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

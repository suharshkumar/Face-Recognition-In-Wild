"""Retrain Phase 1 (IResNet-100 + CurricularFace + BroadFace) on a large
identity dataset, starting from our own Phase 1 weights. Built for one GPU
(e.g. a Colab T4): mixed precision, and a resumable checkpoint every
``--save_every`` steps so a disconnected session can continue.

The default dataset is CASIA-WebFace (490,592 aligned 112x112 faces of 10,572
people, Hugging Face ``SaffalPoosh/casia_web_face``). Augmentation adds
rotation, shift, zoom, blur and colour changes so the network also handles the
unaligned Phase 0 box crops used at inference.

    python train_phase1_large.py --data_dir /content/casia \
        --init weights/recognition_backbone.pt --out_dir /content/drive/MyDrive/frw_phase1
"""

import argparse, io, math, os, random, time
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from losses.broadface_queue import BroadFaceCurricularLoss
from models.backbone import iresnet100


def prepare_casia(data_dir, repo="SaffalPoosh/casia_web_face", shards=20):
    """Unpack the parquet shards into data_dir/<label>/<n>.jpg once; return (paths, labels)."""
    data_dir = Path(data_dir)
    index = data_dir / "index.tsv"
    if not index.is_file():
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        rows = []
        for s in range(shards):
            name = f"data/train-{s:05d}-of-{shards:05d}.parquet"
            local = hf_hub_download(repo, name, repo_type="dataset")
            table = pq.read_table(local, columns=["image", "label"])
            for n, (img, label) in enumerate(zip(table.column("image").to_pylist(), table.column("label").to_pylist())):
                path = data_dir / str(label) / f"{s}_{n}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(img["bytes"])
                rows.append(f"{path}\t{label}")
            os.remove(local)  # keep disk usage low
            print(f"unpacked shard {s + 1}/{shards} ({len(rows)} images)", flush=True)
        index.write_text("\n".join(rows))
    pairs = [line.split("\t") for line in index.read_text().splitlines()]
    return [p for p, _ in pairs], [int(l) for _, l in pairs]


class LowResolution:
    """Randomly shrink and re-enlarge, imitating small faces in crowd photos."""

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            side = random.randint(24, 64)
            img = img.resize((side, side), Image.BILINEAR).resize((112, 112), Image.BILINEAR)
        return img


TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.8, 1.15))], p=0.7),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.3, 0.05)], p=0.8),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([transforms.Lambda(lambda im: im.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5))))], p=0.2),
    LowResolution(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


class FaceFolder(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if img.size != (112, 112):
            img = img.resize((112, 112), Image.BILINEAR)
        return TRAIN_TRANSFORM(img), self.labels[i]


def export_backbone(backbone, path, note):
    torch.save({"schema_version": 2, "kind": "recognition_backbone",
                "model_config": {"backbone_type": "iresnet100", "embedding_dim": 512},
                "input_size": [112, 112], "recognition_alignment": "resize", "note": note,
                "backbone_state": {k: v.detach().cpu() for k, v in backbone.state_dict().items()}}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/content/casia")
    ap.add_argument("--init", default="weights/recognition_backbone.pt", help="our Phase 1 checkpoint to start from")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--queue_size", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels = prepare_casia(args.data_dir)
    num_classes = max(labels) + 1
    print(f"{len(paths)} images, {num_classes} identities, device {device}", flush=True)

    backbone = iresnet100().to(device)
    loss_fn = BroadFaceCurricularLoss(in_features=512, num_classes=num_classes, scale_factor=64.0, margin=0.5,
                                      alpha=0.99, queue_size=args.queue_size, compensate=True,
                                      sync_curriculum=False).to(device)
    params = list(backbone.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    steps_per_epoch = len(paths) // args.batch_size
    total_steps, warmup = args.epochs * steps_per_epoch, 1000
    schedule = lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    step = 0
    last = out / "last.pt"
    if last.is_file():
        ck = torch.load(last, map_location=device, weights_only=False)
        backbone.load_state_dict(ck["backbone"]); loss_fn.load_state_dict(ck["loss_fn"])
        optimizer.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"]); step = ck["step"]
        print(f"Resumed from step {step}", flush=True)
    else:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        state = ck.get("backbone_state", ck.get("backbone"))
        backbone.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        print(f"Initialised backbone from {args.init}", flush=True)

    def save():
        torch.save({"backbone": backbone.state_dict(), "loss_fn": loss_fn.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step}, str(last) + ".tmp")
        os.replace(str(last) + ".tmp", last)

    dataset = FaceFolder(paths, labels)
    while step < total_steps:
        epoch, offset = divmod(step, steps_per_epoch)
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(epoch)).tolist()
        order = order[offset * args.batch_size:steps_per_epoch * args.batch_size]  # skip batches done before a resume
        loader = DataLoader(torch.utils.data.Subset(dataset, order), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=False)
        backbone.train(); loss_fn.train()
        tic, running = time.time(), 0.0
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                embeddings = backbone(images)
            loss = loss_fn(embeddings.float(), targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); scheduler.step()
            loss_fn.enqueue(embeddings.detach().float(), targets)
            step += 1
            running += float(loss.detach())
            if step % 100 == 0:
                rate = 100 * args.batch_size / (time.time() - tic)
                print(f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} loss {running / 100:.3f} "
                      f"lr {scheduler.get_last_lr()[0]:.4f} t {loss_fn.t.item():.3f} {rate:.0f} img/s", flush=True)
                tic, running = time.time(), 0.0
            if step % args.save_every == 0:
                save()
        save()
        export_backbone(backbone, out / "phase1_casia_backbone.pt", f"after epoch {epoch + 1}")
        print(f"Finished epoch {epoch + 1}; exported {out / 'phase1_casia_backbone.pt'}", flush=True)


if __name__ == "__main__":
    main()

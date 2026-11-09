"""Evaluate a recognition backbone on a labelled gallery (no test images used).

* Identification: each image is a probe against all other images (rank-1/5).
* Verification: all image pairs, genuine vs impostor (EER, TAR at FAR 1% / 0.1%).
* Open-set: 20% of identities are removed from the gallery and act as strangers;
  reports how often enrolled people are named correctly and strangers rejected.

    python eval_gallery.py --gallery_dir Face_Dataset/celeb_gallery --backbone weights/phase1_casia_backbone.pt --scale 1.0
"""

import argparse

import numpy as np

from recognize import Recognizer, class_scores, embed_gallery, gallery_faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery_dir", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--unknown_fraction", type=float, default=0.2)
    args = ap.parse_args()

    rec = Recognizer(args.weights_dir, backbone_path=args.backbone)
    names, labels, items = gallery_faces(rec, args.gallery_dir)
    emb = embed_gallery(rec, items, args.scale)
    sim = emb @ emb.T
    n = len(labels)

    # Identification (leave-one-out, identities with 2+ images).
    top1 = top5 = probes = 0
    for i in range(n):
        others = np.arange(n) != i
        if not (labels[others] == labels[i]).any():
            continue
        scores = class_scores(sim[i:i + 1, others], labels[others], len(names))[0]
        ranking = np.argsort(-scores)
        probes += 1
        top1 += int(ranking[0] == labels[i])
        top5 += int(labels[i] in ranking[:5])

    # Verification over all pairs.
    iu = np.triu_indices(n, 1)
    pair_scores = sim[iu]
    same = labels[iu[0]] == labels[iu[1]]
    genuine, impostor = pair_scores[same], pair_scores[~same]
    ts = np.linspace(-0.2, 1.0, 1201)
    frr = np.array([(genuine < t).mean() for t in ts])
    far = np.array([(impostor >= t).mean() for t in ts])
    eer_i = int(np.argmin(np.abs(frr - far)))

    def tar_at(target):
        t = np.quantile(impostor, 1 - target)
        return 100 * (genuine >= t).mean(), t

    # Open-set: hold out a random 20% of identities as strangers.
    rng = np.random.default_rng(0)
    people = np.unique(labels)
    unknown = set(rng.choice(people, int(len(people) * args.unknown_fraction), replace=False).tolist())
    known_mask = np.array([l not in unknown for l in labels])
    correct = wrong = rejected_known = 0
    rejected_unknown = accepted_unknown = 0
    for i in range(n):
        enrolled = known_mask & (np.arange(n) != i)
        if labels[i] not in unknown and not (labels[enrolled] == labels[i]).any():
            continue  # a known person with no other enrolled image
        scores = class_scores(sim[i:i + 1, enrolled], labels[enrolled], len(names))[0]
        order = np.argsort(-scores)
        best, second = scores[order[0]], scores[order[1]]
        accept = best >= args.threshold and best - second >= args.margin
        if labels[i] in unknown:
            accepted_unknown += int(accept); rejected_unknown += int(not accept)
        elif not accept:
            rejected_known += 1
        elif order[0] == labels[i]:
            correct += 1
        else:
            wrong += 1
    known_total = correct + wrong + rejected_known
    unknown_total = accepted_unknown + rejected_unknown

    print(f"\nBackbone: {args.backbone}  (crop scale {args.scale})")
    print(f"Gallery: {n} images, {len(names)} people\n")
    print(f"Identification ({probes} probes):  rank-1 {100 * top1 / probes:.1f}%   rank-5 {100 * top5 / probes:.1f}%")
    t1, th1 = tar_at(0.01); t01, th01 = tar_at(0.001)
    print(f"Verification ({len(genuine)} genuine / {len(impostor)} impostor pairs):")
    print(f"  EER {100 * frr[eer_i]:.1f}% (threshold {ts[eer_i]:.3f})")
    print(f"  TAR @ FAR 1%   = {t1:.1f}% (threshold {th1:.3f})")
    print(f"  TAR @ FAR 0.1% = {t01:.1f}% (threshold {th01:.3f})")
    print(f"Open-set at threshold {args.threshold} ({len(unknown)} people held out as strangers):")
    print(f"  enrolled people: named correctly {100 * correct / known_total:.1f}%, wrong name {100 * wrong / known_total:.1f}%, "
          f"returned unknown {100 * rejected_known / known_total:.1f}%  ({known_total} probes)")
    print(f"  strangers: correctly rejected {100 * rejected_unknown / unknown_total:.1f}%, "
          f"wrongly named {100 * accepted_unknown / unknown_total:.1f}%  ({unknown_total} probes)")


if __name__ == "__main__":
    main()

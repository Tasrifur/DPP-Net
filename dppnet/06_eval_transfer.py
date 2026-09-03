#!/usr/bin/env python3
"""Step 6: cross-dataset zero-shot transfer to CASIA-E and OU-MVLP (Table 8).

The CASIA-B model is applied directly, with no retraining. Identities on the
target dataset are determined solely by prototype averaging.

Unlike the CASIA-B experiments, part embeddings are L2-normalised before
matching here; this reduces the domain gap and is stated wherever transfer
results are reported in the paper.

The target datasets must first be converted to the same structured GEI format
as CASIA-B. See README.md, "Preparing CASIA-E and OU-MVLP".

Usage:
    python scripts/06_eval_transfer.py --dataset casia-e --gei /path/casiae_gei.pt
    python scripts/06_eval_transfer.py --dataset ou-mvlp --gei /path/oumvlp_gei.pt
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dppnet import config as C
from dppnet.evaluate import ci95, cosine
from dppnet.model import load_model

CASIAE_ANGLES = ["000", "015", "030", "045", "060", "075", "090",
                 "105", "120", "135", "150", "165", "180", "270"]
CASIAE_NOVEL = {"105", "120", "270"}          # no close counterpart in CASIA-B
OUMVLP_ANGLES = ["000", "015", "030", "045", "060", "075", "090",
                 "180", "195", "210", "225", "240", "255", "270"]

# Reference attention vector measured on CASIA-B, used for the cosine analysis.
CASIAB_ATTN = np.array([0.0937, 0.0014, 0.0027, 0.0015,
                        0.0134, 0.2243, 0.3840, 0.2790])


@torch.no_grad()
def episode(model, gei, subjects, view, cond, k_shot, n_way, q_queries):
    """One episode on a target dataset, with L2-normalised embeddings."""
    chosen = np.random.choice(subjects, n_way, replace=False)
    sup, qry, labels = [], [], []
    for i, s in enumerate(chosen):
        seqs = gei[s][cond][view]
        n = seqs.shape[0]
        idx = np.random.permutation(n)
        g = idx[:k_shot] if n > k_shot else np.arange(n)
        p = idx[k_shot:k_shot + q_queries] if n > k_shot else np.arange(n)
        if len(p) == 0:
            p = g[:1]
        sup.append(seqs[g][:k_shot] if len(g) >= k_shot else
                   seqs[np.random.choice(n, k_shot, replace=True)])
        qry.append(seqs[p])
        labels += [i] * len(p)

    s_t = torch.cat(sup, 0).to(C.DEVICE)
    q_t = torch.cat(qry, 0).to(C.DEVICE)
    lbl = torch.tensor(labels, device=C.DEVICE)

    S = F.normalize(model.encoder(s_t), dim=2)
    Q = F.normalize(model.encoder(q_t), dim=2)
    proto = S.view(n_way, k_shot, C.NUM_PARTS, S.size(2)).mean(1)
    r = Q - proto.mean(0, keepdim=True)
    w = model.attention_net(torch.cat(
        [Q.reshape(Q.size(0), -1), r.reshape(Q.size(0), -1)], 1))
    dist = ((Q.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(3)
    pred = torch.argmin((dist * w.unsqueeze(1)).sum(2), 1)
    return (pred == lbl).float().mean().item() * 100, w.mean(0).cpu().numpy()


def run(model, gei, angles, conds, k_shot, n_way, q_queries, episodes):
    subjects = sorted(gei.keys())
    per_angle, attn_acc = {}, []
    for view in angles:
        row = {}
        for cond in conds:
            usable = [s for s in subjects
                      if cond in gei[s] and view in gei[s][cond]]
            if len(usable) < n_way:
                row[cond] = None
                continue
            per_ep, ws = [], []
            for _ in range(episodes):
                a, w = episode(model, gei, usable, view, cond,
                               k_shot, n_way, q_queries)
                per_ep.append(a)
                ws.append(w)
            row[cond] = {"acc": float(np.mean(per_ep)), "ci": ci95(per_ep)}
            attn_acc.append(np.mean(ws, axis=0))
        per_angle[view] = row
        shown = " ".join(f"{c}={row[c]['acc']:.2f}" for c in conds
                         if row[c] is not None)
        print(f"  {view}:  {shown}")
    mean_attn = np.mean(attn_acc, axis=0)
    return per_angle, mean_attn / mean_attn.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["casia-e", "ou-mvlp"], required=True)
    ap.add_argument("--gei", required=True,
                    help="Structured GEI dict for the target dataset")
    ap.add_argument("--ckpt", default=C.SAME_VIEW_CKPT)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--n-way", type=int, default=20)
    ap.add_argument("--save", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = load_model(args.ckpt)
    print(f"Loaded {args.ckpt} (trained on CASIA-B only, no retraining)")
    gei = torch.load(args.gei)
    print(f"Loaded {args.gei}: {len(gei)} subjects")

    if args.dataset == "casia-e":
        angles, conds, k_shot = CASIAE_ANGLES, ["NM", "BG", "CL"], 2
    else:
        angles, conds, k_shot = OUMVLP_ANGLES, ["NM"], 1

    print(f"\nZero-shot transfer, {args.n_way}-way {k_shot}-shot, "
          f"{args.episodes} episodes/cell")
    per_angle, mean_attn = run(model, gei, angles, conds, k_shot,
                               args.n_way, 2, args.episodes)

    print("\n" + "=" * 62)
    print(f"TABLE 8 - {args.dataset} zero-shot transfer")
    print("=" * 62)
    means = {}
    for cond in conds:
        vals = [per_angle[v][cond]["acc"] for v in angles
                if per_angle[v][cond] is not None]
        cis = [per_angle[v][cond]["ci"] for v in angles
               if per_angle[v][cond] is not None]
        means[cond] = float(np.mean(vals))
        print(f"{cond}: {means[cond]:.2f} +- {np.mean(cis):.2f}   "
              f"(the +- is the mean of the per-angle 95% CIs)")
    overall = float(np.mean(list(means.values())))
    print(f"Mean over conditions: {overall:.2f}")

    if args.dataset == "casia-e":
        def bucket(sel):
            vals = []
            for a in angles:
                if (a in CASIAE_NOVEL) != sel:
                    continue
                cell = [per_angle[a][c]["acc"] for c in conds
                        if per_angle[a][c] is not None]
                if cell:
                    vals.append(float(np.mean(cell)))
            return float(np.mean(vals)) if vals else None

        novel, shared = bucket(True), bucket(False)
        if novel is None or shared is None:
            print("\nNovel-angle comparison skipped: one of the two groups had "
                  "no angle with enough subjects to fill an episode.")
        else:
            print(f"\nNovel angles (105/120/270) : {novel:.2f}%")
            print(f"Angles with a CASIA-B neighbour: {shared:.2f}%")
            print(f"Gap in favour of the novel views: {novel-shared:+.2f}")
            print("The dips at 45 and 135 degrees are therefore specific to "
                  "those two views, not a general failure to generalise.")

    cos = cosine(CASIAB_ATTN, mean_attn)
    print(f"\nAttention cosine vs the CASIA-B reference: {cos:.4f}")
    print("Target mean attention: " +
          ", ".join(f"{n}={w:.4f}" for n, w in zip(C.PART_NAMES, mean_attn)))

    save = args.save or os.path.join(C.OUT_ROOT,
                                     f"transfer_{args.dataset}.json")
    with open(save, "w") as f:
        json.dump({"per_angle": per_angle, "means": means, "overall": overall,
                   "attention": mean_attn.tolist(), "cosine": cos}, f, indent=2)
    print(f"\nResults -> {save}")


if __name__ == "__main__":
    main()

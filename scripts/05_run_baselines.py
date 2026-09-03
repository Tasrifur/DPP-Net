#!/usr/bin/env python3
"""Step 5: the controlled comparison of five matching heads (Table 7).

Every head wraps the identical part-aware encoder and is trained under the same
budget, objective, optimiser, seed and stopping rule, so the only thing that
differs is the matching head.

Two properties of this harness apply equally to all five methods and are stated
in Section 4.5 of the paper:
  * L2 normalisation (used only for cross-dataset transfer) is omitted;
  * normal-condition probes are drawn from a pool that includes the gallery
    sequences, so the NM column is optimistic in absolute terms.

Usage:
    python scripts/05_run_baselines.py                # train then evaluate
    python scripts/05_run_baselines.py --eval-only    # reuse saved heads
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dppnet import config as C
from dppnet.baselines import METHODS, save_name
from dppnet.data import check_view_major, load_casiab, subject_splits
from dppnet.episodes import sample_episode
from dppnet.evaluate import ci95

PROBES = {"NM": C.PROBE_LOCAL_NM, "BG": C.PROBE_LOCAL_BG, "CL": C.PROBE_LOCAL_CL}
BASELINE_EPISODES = 10000
BASELINE_VAL_EVERY = 500
BASELINE_PATIENCE = 6
EVAL_EPISODES = 600
SEED = 42


def episode_batch(data, gallery_local, probe_local):
    v = np.random.randint(0, C.NUM_VIEWS)
    return sample_episode(data, v, v, gallery_local, probe_local,
                          C.N_WAY, C.K_SHOT, C.Q_QUERIES, C.DEVICE)


@torch.no_grad()
def evaluate(model, data, probe_local, episodes=EVAL_EPISODES):
    model.eval()
    per_ep = []
    for _ in range(episodes):
        s, q, lbl = episode_batch(data, C.GALLERY_LOCAL, probe_local)
        logits = model(s, q, C.N_WAY, C.K_SHOT)
        per_ep.append((torch.argmax(logits, 1) == lbl).float().mean().item() * 100)
    return float(np.mean(per_ep)), ci95(per_ep)


def train_head(name, cls, train_data, val_data, ckpt_dir):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = cls().to(C.DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- {name} ({n_params:,} parameters) ---")

    opt = optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=BASELINE_EPISODES, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    best, patience = 0.0, 0
    path = os.path.join(ckpt_dir, save_name(name))
    for ep in range(BASELINE_EPISODES):
        model.train()
        s, q, lbl = episode_batch(train_data, C.GALLERY_LOCAL, C.PROBE_LOCAL_ALL)
        opt.zero_grad()
        loss = loss_fn(model(s, q, C.N_WAY, C.K_SHOT), lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step()
        sched.step()

        if (ep + 1) % BASELINE_VAL_EVERY == 0:
            acc, _ = evaluate(model, val_data, C.PROBE_LOCAL_NM, episodes=150)
            print(f"  ep {ep+1:5d} | loss {loss.item():.4f} | val {acc:.2f}%",
                  end="")
            if acc > best:
                best, patience = acc, 0
                torch.save(model.state_dict(), path)
                print("  <- best")
            else:
                patience += 1
                print(f"  ({patience}/{BASELINE_PATIENCE})")
                if patience >= BASELINE_PATIENCE:
                    print(f"  early stop at {ep+1}")
                    break
    model.load_state_dict(torch.load(path, map_location=C.DEVICE))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=os.path.join(C.CKPT_ROOT, "baselines"))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--save", default=os.path.join(C.OUT_ROOT,
                                                   "baseline_results.json"))
    args = ap.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    data = load_casiab()
    check_view_major(data)
    train_data, val_data, test_data = subject_splits(data)

    print("\nParameter counts")
    print("-" * 40)
    for nm, M in METHODS.items():
        print(f"{nm:<20} {sum(p.numel() for p in M().parameters()):>12,}")
    print("DPP-Net must report 685,192 to match the paper.")

    results = {}
    for name, cls in METHODS.items():
        path = os.path.join(args.ckpt_dir, save_name(name))
        if args.eval_only:
            model = cls().to(C.DEVICE)
            model.load_state_dict(torch.load(path, map_location=C.DEVICE))
        else:
            model = train_head(name, cls, train_data, val_data, args.ckpt_dir)
        row = {}
        for cond, probe in PROBES.items():
            acc, ci = evaluate(model, test_data, probe)
            row[cond] = {"acc": acc, "ci": ci}
        results[name] = row

    print("\n" + "=" * 66)
    print(f"TABLE 7 - Controlled comparison, {EVAL_EPISODES} episodes, "
          f"mean +- 95% CI")
    print("=" * 66)
    print(f"{'Method':<20}{'NM':>15}{'BG':>15}{'CL':>15}")
    for name, row in results.items():
        print(f"{name:<20}" + "".join(
            f"{row[c]['acc']:>10.2f}+-{row[c]['ci']:<4.2f}" for c in PROBES))

    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {args.save}")


if __name__ == "__main__":
    main()

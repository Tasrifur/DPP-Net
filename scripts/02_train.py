#!/usr/bin/env python3
"""Step 2: train DPP-Net on CASIA-B.

Two models are trained and they are never mixed:

    --mode same-view    gallery and probe drawn from the SAME angle
                        -> checkpoints/best_dpp_net_v3_vm.pt
                        -> produces Tables 3, 4, 5, 10, 12

    --mode cross-view   gallery and probe angles sampled INDEPENDENTLY
                        -> checkpoints/best_dpp_net_xview_vm.pt
                        -> produces Table 6

Usage:
    python scripts/02_train.py --mode same-view
    python scripts/02_train.py --mode cross-view
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dppnet import config as C
from dppnet.data import check_view_major, load_casiab, subject_splits
from dppnet.episodes import (CrossViewEpisodeDataset, ViewAwareEpisodeDataset)
from dppnet.model import build_model


@torch.no_grad()
def validate(model, loader):
    model.eval()
    correct = total = 0
    for s, q, lbl in loader:
        s = s.squeeze(0).to(C.DEVICE)
        q = q.squeeze(0).to(C.DEVICE)
        lbl = lbl.squeeze(0).to(C.DEVICE)
        logits, _ = model(s, q, C.N_WAY, C.K_SHOT)
        correct += (torch.argmax(logits, 1) == lbl).sum().item()
        total += lbl.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["same-view", "cross-view"],
                    default="same-view")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cross = args.mode == "cross-view"
    episodes = args.episodes or (C.XV_TRAIN_EPISODES if cross else C.SV_TRAIN_EPISODES)
    log_every = C.XV_LOG_INTERVAL if cross else C.SV_LOG_INTERVAL
    patience_max = C.XV_PATIENCE if cross else C.SV_PATIENCE
    out = args.out or (C.CROSS_VIEW_CKPT if cross else C.SAME_VIEW_CKPT)

    data = load_casiab()
    check_view_major(data)
    train_data, val_data, _ = subject_splits(data)

    model = build_model()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=episodes, eta_min=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    # Training probes span all conditions so the model sees covariate diversity.
    # Validation probes are normal-condition only, matching the reported protocol.
    if cross:
        train_ds = CrossViewEpisodeDataset(
            train_data, C.N_WAY, C.K_SHOT, C.Q_QUERIES, episodes,
            C.GALLERY_LOCAL, C.PROBE_LOCAL_ALL, p_same=C.XV_P_SAME)
        val_ds = CrossViewEpisodeDataset(
            val_data, C.N_WAY, C.K_SHOT, C.Q_QUERIES, C.XV_VAL_EPISODES,
            C.GALLERY_LOCAL, C.PROBE_LOCAL_NM, p_same=0.0)
    else:
        train_ds = ViewAwareEpisodeDataset(
            train_data, C.N_WAY, C.K_SHOT, C.Q_QUERIES, episodes,
            C.GALLERY_LOCAL, C.PROBE_LOCAL_ALL)
        val_ds = ViewAwareEpisodeDataset(
            val_data, C.N_WAY, C.K_SHOT, C.Q_QUERIES, C.SV_VAL_EPISODES,
            C.GALLERY_LOCAL, C.PROBE_LOCAL_NM)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    print(f"\nTraining ({args.mode}) — up to {episodes} episodes, "
          f"patience {patience_max} validation checks\n")

    best, patience = 0.0, 0
    for ep, (s, q, lbl) in enumerate(train_loader):
        model.train()
        s = s.squeeze(0).to(C.DEVICE)
        q = q.squeeze(0).to(C.DEVICE)
        lbl = lbl.squeeze(0).to(C.DEVICE)

        opt.zero_grad()
        logits, _ = model(s, q, C.N_WAY, C.K_SHOT)
        loss = loss_fn(logits, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step()
        sched.step()

        if (ep + 1) % log_every == 0:
            acc = validate(model, val_loader)
            print(f"Ep {ep+1:5d} | loss {loss.item():.4f} | val {acc:.4f}", end="")
            if acc > best:
                best, patience = acc, 0
                torch.save(model.state_dict(), out)
                print("  <- new best, saved")
            else:
                patience += 1
                print(f"  ({patience}/{patience_max})")
                if patience >= patience_max:
                    print(f"\nEarly stopping at episode {ep+1}")
                    break

    print(f"\nBest validation accuracy {best:.4f}")
    print(f"Checkpoint -> {out}")


if __name__ == "__main__":
    main()

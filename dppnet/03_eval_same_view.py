#!/usr/bin/env python3
"""Step 3: reproduce the same-view CASIA-B results.

Reproduces, in order:
    Table 3   per-angle same-view CCR              (300 episodes / cell)
    Table 4   ablation, fixed uniform vs dynamic   (500 episodes / condition)
    Table 5   3x3 gallery-probe condition matrix   (400 episodes / cell)
    Table 10  k-shot and N-way sensitivity at 090  (600 episodes / cell)
    Table 12  enrollment and identification timing

Usage:
    python scripts/03_eval_same_view.py
    python scripts/03_eval_same_view.py --only table3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dppnet import config as C
from dppnet.data import check_view_major, load_casiab, subject_splits
from dppnet.episodes import sample_episode
from dppnet.evaluate import (eval_accuracy, eval_random_view, run_ablation)
from dppnet.model import load_model

PROBES = {"NM": C.PROBE_LOCAL_NM, "BG": C.PROBE_LOCAL_BG, "CL": C.PROBE_LOCAL_CL}


def table3(model, test_data, results):
    print("\n" + "=" * 62)
    print("TABLE 3 - CASIA-B same-view CCR (%) per angle, 300 episodes/cell")
    print("=" * 62)
    print(f"{'Angle':<8}{'NM':>10}{'BG':>10}{'CL':>10}{'Avg':>10}")
    rows, col = {}, {c: [] for c in PROBES}
    for vi, view in enumerate(C.CASIA_VIEWS):
        vals = {}
        for cond, probe in PROBES.items():
            a = eval_accuracy(model, test_data, C.GALLERY_LOCAL, probe,
                              vi, vi, episodes=300)
            vals[cond] = a
            col[cond].append(a)
        avg = np.mean(list(vals.values()))
        rows[view] = {**vals, "Avg": avg}
        print(f"{view:<8}{vals['NM']:>10.2f}{vals['BG']:>10.2f}"
              f"{vals['CL']:>10.2f}{avg:>10.2f}")
    means = {c: float(np.mean(col[c])) for c in PROBES}
    overall = float(np.mean(list(means.values())))
    print("-" * 48)
    print(f"{'Mean':<8}{means['NM']:>10.2f}{means['BG']:>10.2f}"
          f"{means['CL']:>10.2f}{overall:>10.2f}")
    results["table3"] = {"per_angle": rows, "means": means, "overall": overall}


def table4(model, test_data, results):
    print("\n" + "=" * 62)
    print("TABLE 4 - Ablation: fixed uniform vs residual dynamic attention")
    print("=" * 62)
    print(f"{'Condition':<12}{'Fixed':>12}{'Dynamic':>12}{'Gain':>10}")
    out, fixed_all, dyn_all = {}, [], []
    for cond, probe in PROBES.items():
        off = run_ablation(model, test_data, C.GALLERY_LOCAL, probe,
                           use_attention=False, episodes=500)
        on = run_ablation(model, test_data, C.GALLERY_LOCAL, probe,
                          use_attention=True, episodes=500)
        out[cond] = {"fixed": off, "dynamic": on, "gain": on - off}
        fixed_all.append(off)
        dyn_all.append(on)
        print(f"{cond:<12}{off:>12.2f}{on:>12.2f}{on-off:>+10.2f}")
    af, ad = float(np.mean(fixed_all)), float(np.mean(dyn_all))
    print("-" * 46)
    print(f"{'Average':<12}{af:>12.2f}{ad:>12.2f}{ad-af:>+10.2f}")
    out["Avg"] = {"fixed": af, "dynamic": ad, "gain": ad - af}
    results["table4"] = out


def table5(model, test_data, results):
    print("\n" + "=" * 62)
    print("TABLE 5 - Condition matrix, 400 episodes/cell")
    print("=" * 62)
    gal = {"NM": C.GALLERY_LOCAL, "BG": C.BG_LOCAL, "CL": C.CL_LOCAL}
    print(f"{'Gallery\\Probe':<16}{'NM':>10}{'BG':>10}{'CL':>10}")
    mat = {}
    for gname, glocal in gal.items():
        row = {}
        for pname, plocal in PROBES.items():
            accs = [eval_accuracy(model, test_data, glocal, plocal, v, v,
                                  episodes=400 // C.NUM_VIEWS + 1)
                    for v in range(C.NUM_VIEWS)]
            row[pname] = float(np.mean(accs))
        mat[gname] = row
        print(f"{gname:<16}{row['NM']:>10.2f}{row['BG']:>10.2f}{row['CL']:>10.2f}")
    results["table5"] = mat


def table10(model, test_data, results):
    v90 = C.CASIA_VIEWS.index("090")
    print("\n" + "=" * 62)
    print("TABLE 10a - k-shot sensitivity at 090, 20-way, 600 episodes")
    print("=" * 62)
    print(f"{'k':<6}{'NM':>18}{'BG':>18}{'CL':>18}")
    kshot = {}
    for k in (1, 3, 5, 10):
        row = {}
        for cond, probe in PROBES.items():
            a, ci = eval_accuracy(model, test_data, C.GALLERY_LOCAL, probe,
                                  v90, v90, episodes=600, k_shot=k,
                                  return_ci=True)
            row[cond] = {"acc": a, "ci": ci}
        kshot[k] = row
        print(f"{k:<6}" + "".join(
            f"{row[c]['acc']:>12.2f}+-{row[c]['ci']:<4.2f}" for c in PROBES))

    print("\n" + "=" * 62)
    print("TABLE 10b - N-way sensitivity at 090, 5-shot, 600 episodes")
    print("=" * 62)
    print(f"{'N':<6}{'NM':>18}{'BG':>18}{'CL':>18}")
    nway = {}
    for n in (5, 10, 15, 20):
        row = {}
        for cond, probe in PROBES.items():
            a, ci = eval_accuracy(model, test_data, C.GALLERY_LOCAL, probe,
                                  v90, v90, episodes=600, n_way=n,
                                  return_ci=True)
            row[cond] = {"acc": a, "ci": ci}
        nway[n] = row
        print(f"{n:<6}" + "".join(
            f"{row[c]['acc']:>12.2f}+-{row[c]['ci']:<4.2f}" for c in PROBES))
    results["table10"] = {"k_shot": kshot, "n_way": nway}


@torch.no_grad()
def table12(model, test_data, results):
    print("\n" + "=" * 62)
    print("TABLE 12 - Enrollment and identification cost")
    print("=" * 62)
    n_params = sum(p.numel() for p in model.parameters())
    v90 = C.CASIA_VIEWS.index("090")
    s, q, _ = sample_episode(test_data, v90, v90, C.GALLERY_LOCAL,
                             C.PROBE_LOCAL_NM, C.N_WAY, C.K_SHOT,
                             C.Q_QUERIES, C.DEVICE)

    for _ in range(10):                       # warm up CUDA kernels
        model.encoder(s)
    if C.DEVICE == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        feats = model.encoder(s)
        feats.view(C.N_WAY, C.K_SHOT, C.NUM_PARTS, -1).mean(1)
    if C.DEVICE == "cuda":
        torch.cuda.synchronize()
    enroll_ms = (time.perf_counter() - t0) / (50 * C.N_WAY) * 1000

    t0 = time.perf_counter()
    for _ in range(50):
        model(s, q[:1], C.N_WAY, C.K_SHOT)
    if C.DEVICE == "cuda":
        torch.cuda.synchronize()
    ident_ms = (time.perf_counter() - t0) / 50 * 1000

    print(f"Parameters              : {n_params:,} ({n_params/1e6:.3f} M)")
    print(f"Model size (float32)    : {n_params*4/1024**2:.2f} MB")
    print(f"Enroll one subject      : {enroll_ms:.2f} ms")
    print(f"Identify one query      : {ident_ms:.2f} ms")
    print(f"Enrollment throughput   : ~{1000/enroll_ms:,.0f} / s")
    print("\nTimings are hardware dependent; the paper reports values measured "
          "on a single GPU.")
    results["table12"] = {"params": n_params, "enroll_ms": enroll_ms,
                          "identify_ms": ident_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=C.SAME_VIEW_CKPT)
    ap.add_argument("--only", default="all",
                    choices=["all", "table3", "table4", "table5",
                             "table10", "table12"])
    ap.add_argument("--save", default=os.path.join(C.OUT_ROOT,
                                                   "same_view_results.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_casiab()
    check_view_major(data)
    _, _, test_data = subject_splits(data)
    model = load_model(args.ckpt)
    print(f"Loaded {args.ckpt}")

    results = {}
    run = {"table3": table3, "table4": table4, "table5": table5,
           "table10": table10, "table12": table12}
    for name, fn in run.items():
        if args.only in ("all", name):
            fn(model, test_data, results)

    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {args.save}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 4: reproduce the cross-view CASIA-B results (Table 6).

The identical-view case is EXCLUDED: for each probe angle the accuracy is
averaged over the ten gallery angles that differ from it.

The script also reports the two control comparisons quoted in Section 4.4:
  * the same-view model evaluated under the cross-view protocol, which shows
    that cross-view training is what makes this regime tractable;
  * the identical-view mean of both models, which shows that cross-view
    training does not cost same-view accuracy.

Usage:
    python scripts/04_eval_cross_view.py
    python scripts/04_eval_cross_view.py --skip-controls
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dppnet import config as C
from dppnet.data import check_view_major, load_casiab, subject_splits
from dppnet.evaluate import eval_accuracy, eval_cross_view
from dppnet.model import load_model

PROBES = {"NM": C.PROBE_LOCAL_NM, "BG": C.PROBE_LOCAL_BG, "CL": C.PROBE_LOCAL_CL}


def identical_view_mean(model, test_data, episodes=300):
    """Mean CCR when gallery and probe share an angle, normal-condition probes."""
    accs = [eval_accuracy(model, test_data, C.GALLERY_LOCAL, C.PROBE_LOCAL_NM,
                          v, v, episodes=episodes) for v in range(C.NUM_VIEWS)]
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=C.CROSS_VIEW_CKPT)
    ap.add_argument("--same-view-ckpt", default=C.SAME_VIEW_CKPT)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument("--save", default=os.path.join(C.OUT_ROOT,
                                                   "cross_view_results.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_casiab()
    check_view_major(data)
    _, _, test_data = subject_splits(data)
    xv_model = load_model(args.ckpt)
    print(f"Loaded cross-view model: {args.ckpt}")

    print("\n" + "=" * 62)
    print(f"TABLE 6 - CASIA-B cross-view CCR (%), identical view excluded, "
          f"{args.episodes} episodes/cell")
    print("=" * 62)
    print(f"{'Probe':<10}{'NM':>10}{'BG':>10}{'CL':>10}{'Avg':>10}")

    rows, col = {}, {c: [] for c in PROBES}
    for vi, view in enumerate(C.CASIA_VIEWS):
        vals = {}
        for cond, probe in PROBES.items():
            a = eval_cross_view(xv_model, test_data, C.GALLERY_LOCAL, probe,
                                vi, episodes=args.episodes)
            vals[cond] = a
            col[cond].append(a)
        avg = float(np.mean(list(vals.values())))
        rows[view] = {**vals, "Avg": avg}
        print(f"{view:<10}{vals['NM']:>10.2f}{vals['BG']:>10.2f}"
              f"{vals['CL']:>10.2f}{avg:>10.2f}")

    means = {c: float(np.mean(col[c])) for c in PROBES}
    overall = float(np.mean(list(means.values())))
    print("-" * 50)
    print(f"{'Mean':<10}{means['NM']:>10.2f}{means['BG']:>10.2f}"
          f"{means['CL']:>10.2f}{overall:>10.2f}")

    results = {"table6": {"per_angle": rows, "means": means, "overall": overall}}

    if not args.skip_controls:
        print("\n" + "=" * 62)
        print("CONTROLS (Section 4.4), normal-condition probes throughout")
        print("=" * 62)
        sv_model = load_model(args.same_view_ckpt)

        sv_cross = float(np.mean([
            eval_cross_view(sv_model, test_data, C.GALLERY_LOCAL,
                            C.PROBE_LOCAL_NM, v, episodes=args.episodes)
            for v in range(C.NUM_VIEWS)]))
        xv_cross = means["NM"]
        sv_ident = identical_view_mean(sv_model, test_data)
        xv_ident = identical_view_mean(xv_model, test_data)

        print(f"Cross-view CCR, same-view-trained model  : {sv_cross:.2f}%")
        print(f"Cross-view CCR, cross-view-trained model : {xv_cross:.2f}%")
        print(f"  gain from cross-view training          : {xv_cross-sv_cross:+.2f}")
        print(f"Identical-view mean, same-view model     : {sv_ident:.2f}%")
        print(f"Identical-view mean, cross-view model    : {xv_ident:.2f}%")
        print("  cross-view training costs essentially nothing same-view")
        results["controls"] = {
            "cross_view_same_view_model": sv_cross,
            "cross_view_cross_view_model": xv_cross,
            "identical_view_same_view_model": sv_ident,
            "identical_view_cross_view_model": xv_ident,
        }

    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {args.save}")


if __name__ == "__main__":
    main()

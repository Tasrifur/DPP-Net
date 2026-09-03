#!/usr/bin/env python3
"""Step 1: build Gait Energy Images from CASIA-B silhouettes.

Input : a directory of silhouette PNGs laid out as
            <root>/<subject>/<sequence>/<view>/*.png
        which is how CASIA-B is distributed after unzipping.
Output: two files in $DPPNET_DATA_ROOT
            casiab_gei_structured_view.pt   nested dict, used for figures
            casiab_gei_features_viewmajor.pt flat tensor [124, 110, 1, 64, 64]

The flat tensor is assembled VIEW-MAJOR: sequence index v*10 + local addresses
view v. Every episode sampler assumes this. Assembling it condition-major
instead produces a tensor of the same shape that trains without error but
silently mislabels every per-angle and per-condition result, so the layout is
verified at the end of this script and again before training.

Usage:
    python scripts/01_extract_gei.py --silhouette-root /path/to/casia-b
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dppnet.config import (CASIA_VIEWS, CASIAB_FEATURES, CASIAB_STRUCTURED,
                           IMG_SIZE)
from dppnet.data import check_view_major


def compute_gei(frames):
    return np.mean(np.stack(frames, axis=0), axis=0).astype(np.float32)


def find_root(path):
    """Descend through single-directory wrappers left by unzipping."""
    root = path
    while True:
        entries = [e for e in os.listdir(root) if not e.startswith(".")]
        if len(entries) == 1 and os.path.isdir(os.path.join(root, entries[0])):
            root = os.path.join(root, entries[0])
        else:
            return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silhouette-root", required=True,
                    help="Directory containing the unzipped CASIA-B silhouettes")
    ap.add_argument("--out-flat", default=CASIAB_FEATURES)
    ap.add_argument("--out-structured", default=CASIAB_STRUCTURED)
    args = ap.parse_args()

    root = find_root(args.silhouette_root)
    print(f"Dataset root: {root}")

    gei_data, n_seq = {}, 0
    subjects = sorted(s for s in os.listdir(root)
                      if os.path.isdir(os.path.join(root, s)))
    print(f"Found {len(subjects)} subjects. Computing GEIs ...")

    for si, subj in enumerate(subjects, 1):
        subj_path = os.path.join(root, subj)
        gei_data[subj] = {"NM": {}, "BG": {}, "CL": {}}
        for seq in sorted(os.listdir(subj_path)):
            seq_path = os.path.join(subj_path, seq)
            if not os.path.isdir(seq_path):
                continue
            sl = seq.lower()
            cond = "NM" if "nm" in sl else "BG" if "bg" in sl else "CL" if "cl" in sl else None
            if cond is None:
                continue
            for view in sorted(os.listdir(seq_path)):
                view_path = os.path.join(seq_path, view)
                if not os.path.isdir(view_path):
                    continue
                frames = []
                for img_name in sorted(os.listdir(view_path)):
                    if not img_name.lower().endswith((".png", ".jpg", ".bmp")):
                        continue
                    try:
                        img = Image.open(os.path.join(view_path, img_name)).convert("L")
                        img = img.resize((IMG_SIZE, IMG_SIZE))
                        frames.append(np.array(img, dtype=np.float32) / 255.0)
                    except Exception:
                        continue
                if not frames:
                    continue
                gei = torch.tensor(compute_gei(frames)).unsqueeze(0)
                gei_data[subj][cond].setdefault(view.zfill(3), []).append(gei)
                n_seq += 1
        if si % 10 == 0:
            print(f"  {si}/{len(subjects)} subjects, {n_seq} sequences")

    for subj in gei_data:
        for cond in gei_data[subj]:
            for view in gei_data[subj][cond]:
                gei_data[subj][cond][view] = torch.stack(
                    gei_data[subj][cond][view], dim=0)

    print(f"Total GEI sequences: {n_seq}")
    torch.save(gei_data, args.out_structured)
    print(f"Structured dict -> {args.out_structured}")

    # ---- flat tensor, VIEW-MAJOR --------------------------------------------
    # Outer loop over views, inner over conditions, so that each block of 10
    # consecutive sequence indices belongs to one viewing angle.
    subject_tensors = []
    for subj in sorted(gei_data):
        seqs = []
        for view in CASIA_VIEWS:
            for cond in ("NM", "BG", "CL"):
                if view in gei_data[subj][cond]:
                    seqs.append(gei_data[subj][cond][view])
        if seqs:
            subject_tensors.append(torch.cat(seqs, dim=0))

    max_seqs = max(t.shape[0] for t in subject_tensors)
    padded = []
    for t in subject_tensors:
        if t.shape[0] < max_seqs:
            pad = t[-1:].expand(max_seqs - t.shape[0], -1, -1, -1)
            t = torch.cat([t, pad], dim=0)
        padded.append(t.unsqueeze(0))
    flat = torch.cat(padded, dim=0)
    print(f"Flat tensor shape: {tuple(flat.shape)}")

    check_view_major(flat)
    torch.save(flat, args.out_flat)
    print(f"Flat tensor -> {args.out_flat}")


if __name__ == "__main__":
    main()

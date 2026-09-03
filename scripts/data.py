"""Loading the CASIA-B GEI tensor and reproducing the 74/25/25 subject split."""

import numpy as np
import torch

from .config import (CASIAB_FEATURES, DEVICE, NUM_VIEWS, SEQS_PER_VIEW,
                     SPLIT_SEED, TOTAL_SUBJECTS, TRAIN_SUBJECTS, VAL_SUBJECTS)


def load_casiab(path=CASIAB_FEATURES):
    """Load the flat GEI tensor of shape [124, 110, 1, 64, 64]."""
    data = torch.load(path)
    print(f"Loaded {path}  shape={tuple(data.shape)}")
    return data


def check_view_major(data, tol_ratio=1.5):
    """Verify the tensor is ordered view-major, not condition-major.

    The tensor must be laid out so that sequence index ``v * 10 + local``
    addresses view ``v``. If it was assembled condition-major instead, every
    per-angle and per-condition result would be mislabelled while still looking
    plausible, so this check is run before training.

    The mean pixel intensity of a GEI varies smoothly with viewing angle. Under
    view-major ordering the 11 blocks of 10 therefore have clearly distinct
    means; under condition-major ordering they do not.
    """
    profile = data.mean(dim=(0, 2, 3, 4)).numpy()
    blocks = profile[:NUM_VIEWS * SEQS_PER_VIEW].reshape(NUM_VIEWS, SEQS_PER_VIEW)
    between = blocks.mean(axis=1).std()
    within = blocks.std(axis=1).mean()
    ratio = between / max(within, 1e-8)
    ok = ratio > tol_ratio
    print(f"View-major check: between-block std {between:.5f}, "
          f"within-block std {within:.5f}, ratio {ratio:.2f} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(
            "The GEI tensor does not look view-major. Rebuild it with "
            "scripts/01_extract_gei.py before training; a condition-major "
            "tensor silently mislabels every per-angle result.")
    return ratio


def subject_splits(data, device=DEVICE, seed=SPLIT_SEED):
    """Return (train, val, test) tensors under the fixed 74/25/25 split.

    The permutation is seeded so the split is identical on every machine.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    perm = np.random.permutation(TOTAL_SUBJECTS)
    tr = perm[:TRAIN_SUBJECTS]
    va = perm[TRAIN_SUBJECTS:TRAIN_SUBJECTS + VAL_SUBJECTS]
    te = perm[TRAIN_SUBJECTS + VAL_SUBJECTS:]
    print(f"Split (seed {seed}): train {len(tr)} | val {len(va)} | test {len(te)}")
    return data[tr], data[va], data[te].to(device)

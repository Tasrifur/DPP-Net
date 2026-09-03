"""Global configuration for DPP-Net.

Every constant that defines the experimental protocol lives here so that a
reader can verify the setup in one place. Paths are resolved from environment
variables where possible, so nothing is hard-coded to one machine.
"""

import os

import torch

# --------------------------------------------------------------------------
# Paths. Override with environment variables, e.g.
#   export DPPNET_DATA_ROOT=/path/to/data
# --------------------------------------------------------------------------
DATA_ROOT = os.environ.get("DPPNET_DATA_ROOT", "./data")
CKPT_ROOT = os.environ.get("DPPNET_CKPT_ROOT", "./checkpoints")
OUT_ROOT = os.environ.get("DPPNET_OUT_ROOT", "./outputs")

for _d in (DATA_ROOT, CKPT_ROOT, OUT_ROOT):
    os.makedirs(_d, exist_ok=True)

# Derived artifacts produced by scripts/01_extract_gei.py
CASIAB_FEATURES = os.path.join(DATA_ROOT, "casiab_gei_features_viewmajor.pt")
CASIAB_STRUCTURED = os.path.join(DATA_ROOT, "casiab_gei_structured_view.pt")

# Checkpoints
SAME_VIEW_CKPT = os.path.join(CKPT_ROOT, "best_dpp_net_v3_vm.pt")
CROSS_VIEW_CKPT = os.path.join(CKPT_ROOT, "best_dpp_net_xview_vm.pt")

# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------
IMAGE_CHANNELS = 1
HIDDEN_DIM = 128
NUM_PARTS = 8
IMG_SIZE = 64

# The parameter count is asserted before every run; it is the fastest way to
# catch an accidental architecture edit.
EXPECTED_PARAMS = 685_192

PART_NAMES = [
    "Head", "Shoulders", "UpperTorso", "LowerTorso",
    "Pelvis", "Thighs", "Calves", "Feet",
]

# --------------------------------------------------------------------------
# CASIA-B layout
# --------------------------------------------------------------------------
CASIA_VIEWS = ["000", "018", "036", "054", "072",
               "090", "108", "126", "144", "162", "180"]
NUM_VIEWS = 11
SEQS_PER_VIEW = 10

# Sequence layout WITHIN each view block of 10 (view-major ordering).
#   local 0-5 : nm-01 .. nm-06
#   local 6-7 : bg-01, bg-02
#   local 8-9 : cl-01, cl-02
NM_LOCAL = list(range(0, 6))
BG_LOCAL = [6, 7]
CL_LOCAL = [8, 9]
GALLERY_LOCAL = [4, 5]          # NM-05 and NM-06, the standard CASIA-B gallery

PROBE_LOCAL_NM = NM_LOCAL
PROBE_LOCAL_BG = BG_LOCAL
PROBE_LOCAL_CL = CL_LOCAL
PROBE_LOCAL_ALL = list(range(0, 10))

# --------------------------------------------------------------------------
# Subject splits
# --------------------------------------------------------------------------
TOTAL_SUBJECTS = 124
TRAIN_SUBJECTS = 74
VAL_SUBJECTS = 25
TEST_SUBJECTS = 25
SPLIT_SEED = 42                 # fixes the 74/25/25 partition

# --------------------------------------------------------------------------
# Episodic protocol
# --------------------------------------------------------------------------
N_WAY = 20
K_SHOT = 5
Q_QUERIES = 5

# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

SV_TRAIN_EPISODES = 5000
SV_VAL_EPISODES = 300
SV_LOG_INTERVAL = 100
SV_PATIENCE = 10

XV_TRAIN_EPISODES = 8000
XV_VAL_EPISODES = 300
XV_LOG_INTERVAL = 200
XV_PATIENCE = 12
XV_P_SAME = 0.2                 # fraction of same-view episodes mixed into training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

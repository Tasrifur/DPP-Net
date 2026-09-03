# DPP-Net

Reference implementation of **DPP-Net: A Dynamic Part-Aware Framework for Few-Shot Rapid Enrollment in Cross-Condition Gait Recognition**.

DPP-Net enrolls a new gait identity by averaging the embeddings of a handful of samples, with no gradient update. A part-aware encoder splits each Gait Energy Image into eight horizontal body regions, and a residual-conditioned attention module reweights those regions *per query*, so the parts a coat or a bag has corrupted are suppressed at matching time. The complete model has **685,192 parameters (2.61 MB)**.

Manuscript under review at *Intelligent Systems with Applications*.


---

## Installation

```bash
git clone https://github.com/Tasrifur/DPP-Net.git
cd DPP-Net
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.x. A GPU is recommended for training but not required; every evaluation script runs on CPU.

---

## Getting the datasets

**The datasets are not distributed here, and must not be.** CASIA-B, CASIA-E and OU-MVLP are released under research licence agreements that forbid redistribution. Request them directly:

| Dataset | Where to request |
|---|---|
| CASIA-B | Institute of Automation, Chinese Academy of Sciences — <http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp> |
| CASIA-E | Institute of Automation, Chinese Academy of Sciences — <https://www.scidb.cn/en/detail?dataSetId=57be0e918db743279baf44a38d013a06> |
| OU-MVLP | Osaka University, Institute of Scientific and Industrial Research — <http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitMVLP.html> |

Each requires a signed licence agreement. Approval typically takes days to weeks. This repository contains code only.

Once you have CASIA-B, unzip the silhouettes so they are laid out as:

```
<root>/<subject>/<sequence>/<view>/*.png
e.g.  001/nm-01/090/001-nm-01-090-042.png
```

---

## Configuration

Paths are read from environment variables, with sensible local defaults:

```bash
export DPPNET_DATA_ROOT=./data          # GEI tensors
export DPPNET_CKPT_ROOT=./checkpoints   # trained weights
export DPPNET_OUT_ROOT=./outputs        # JSON results
```

Everything else — the 74/25/25 split seed, the 20-way 5-shot protocol, episode budgets — is in `dppnet/config.py`.

---

## Pipeline

### 1. Build Gait Energy Images

```bash
python scripts/01_extract_gei.py --silhouette-root /path/to/casia-b
```

Produces `casiab_gei_features_viewmajor.pt` (a `[124, 110, 1, 64, 64]` tensor) and `casiab_gei_structured_view.pt` (a nested dict used for figures).

> **The flat tensor must be view-major.** Sequence index `v*10 + local` addresses view `v`. Assembling it condition-major instead yields a tensor of the *same shape* that trains without any error but silently mislabels every per-angle and per-condition result. `dppnet.data.check_view_major` verifies the layout, and is called by the extraction script and again before every training run. Do not disable it.

### 2. Train

```bash
python scripts/02_train.py --mode same-view     # -> best_dpp_net_v3_vm.pt
python scripts/02_train.py --mode cross-view    # -> best_dpp_net_xview_vm.pt
```

Two models are trained and **they are never mixed**. The same-view model draws gallery and probe from the same angle and produces Tables 3, 4, 5, 10 and 12. The cross-view model samples the two angles independently and produces Table 6.

### 3. Same-view results

```bash
python scripts/03_eval_same_view.py
python scripts/03_eval_same_view.py --only table4   # just the ablation
```

Reproduces Table 3 (per-angle CCR), Table 4 (ablation), Table 5 (condition matrix), Table 10 (k-shot and N-way sensitivity), Table 12 (timing).

### 4. Cross-view results

```bash
python scripts/04_eval_cross_view.py
```

Reproduces Table 6 with the identical-view case excluded, plus the two control comparisons from Section 4.4: the same-view model scored under the cross-view protocol, and the identical-view mean of both models.

### 5. Controlled baseline comparison

```bash
python scripts/05_run_baselines.py
```

Trains Prototypical, Matching, Relation, Meta-Baseline and DPP-Net heads on the *identical* encoder, budget, objective, optimiser and seed, then evaluates all five (Table 7). Only the matching head differs.

### 6. Cross-dataset zero-shot transfer

```bash
python scripts/06_eval_transfer.py --dataset casia-e --gei /path/to/casiae_gei.pt
python scripts/06_eval_transfer.py --dataset ou-mvlp --gei /path/to/oumvlp_gei.pt
```

Applies the CASIA-B model directly, with no retraining (Table 8). Part embeddings are L2-normalised before matching here, which is not done for the CASIA-B experiments.

### 7. Figures

```bash
python scripts/07_make_figures.py --gei ./data/casiab_gei_structured_view.pt
```

Regenerates every figure from the recorded numbers, which are hard-coded and cross-checked against the manuscript before anything is drawn. Only Figure 6 needs the GEI dict, for its silhouette row; omit `--gei` to skip it.

---

## Preparing CASIA-E and OU-MVLP

`06_eval_transfer.py` expects a structured GEI dict in the same nested form as CASIA-B:

```python
gei[subject_id][condition][view] -> tensor [n_sequences, 1, 64, 64]
```

with `condition` in `{"NM", "BG", "CL"}` for CASIA-E and `"NM"` for OU-MVLP. Adapt the GEI-building loop in `scripts/01_extract_gei.py` to each dataset's directory layout; the GEI computation itself — mean-pool the silhouette frames, pad to square, resize to 64×64 — is unchanged.

---

## Repository layout

```
dppnet/
  config.py      protocol constants, paths, hyper-parameters
  model.py       part-aware encoder + residual-conditioned attention
  data.py        GEI loading, view-major check, 74/25/25 split
  episodes.py    same-view and cross-view episode samplers
  evaluate.py    CCR, confidence intervals, ablation, attention analysis
  baselines.py   the four comparison heads
scripts/
  01_extract_gei.py      silhouettes -> GEIs
  02_train.py            same-view and cross-view training
  03_eval_same_view.py   Tables 3, 4, 5, 10, 12
  04_eval_cross_view.py  Table 6 + controls
  05_run_baselines.py    Table 7
  06_eval_transfer.py    Table 8
  07_make_figures.py     all figures
results/
  results.json   every published number, machine-readable
```

---

## Reproducibility notes

- `build_model()` asserts the parameter count is exactly 685,192 before any run. An architecture edit fails loudly rather than producing plausible-looking numbers from a different model.
- The subject split is seeded (`SPLIT_SEED = 42`), so the 74/25/25 partition is identical on every machine.
- Episode sampling is stochastic. Re-running an evaluation gives numbers within roughly the reported confidence intervals, not bit-identical values. Pass `--seed` to fix a run.
- Timings in Table 12 are hardware dependent; the published values were measured on a single GPU.
- The normal-condition probe pool includes the gallery sequences, in both the main pipeline and the baseline harness. This is stated in Section 4.5 of the paper and applies equally to every method compared.

---

## Citation

The manuscript is under review. Until it appears, please cite this via the Zenodo DOI (see `CITATION.cff`).

```bibtex
@software{riahi_dppnet,
  author  = {Riahi, Tasrifur and Mustafa, Hossen A},
  title   = {{DPP-Net}: A Dynamic Part-Aware Framework for Few-Shot Rapid
             Enrollment in Cross-Condition Gait Recognition},
  year    = {2026},
  url     = {https://doi.org/10.5281/zenodo.22274621}
}
```

---

## License

Code released under the MIT License (see `LICENSE`). The gait datasets are **not** covered by this licence and remain subject to their own agreements.

## Acknowledgements

Post Graduate Fellowship, Bangladesh University of Engineering and Technology (BUET), Office Order No. AC/PG Fellowship/2025/R-9380. Thanks to the Institute of Automation, Chinese Academy of Sciences, and Osaka University for making the CASIA-B, CASIA-E and OU-MVLP databases available to the research community.

"""Evaluation routines: CCR, confidence intervals, ablation, attention."""

import numpy as np
import torch
from scipy import stats

from .config import (DEVICE, GALLERY_LOCAL, K_SHOT, N_WAY, NUM_PARTS,
                     NUM_VIEWS, Q_QUERIES, SEQS_PER_VIEW)
from .episodes import sample_episode


def ci95(values):
    """Half-width of the 95% confidence interval of a list of per-episode CCRs."""
    v = np.asarray(values, dtype=float)
    m = len(v)
    if m < 2:
        return 0.0
    return float(stats.t.ppf(0.975, m - 1) * v.std(ddof=1) / np.sqrt(m))


@torch.no_grad()
def eval_accuracy(model, data, gallery_local, probe_local, gview, pview,
                  episodes=300, n_way=N_WAY, k_shot=K_SHOT,
                  q_queries=Q_QUERIES, device=DEVICE, return_ci=False):
    """Correct classification rate (%) for a fixed gallery/probe view pair."""
    model.eval()
    per_episode, correct, total = [], 0, 0
    for _ in range(episodes):
        s, q, lbl = sample_episode(data, gview, pview, gallery_local,
                                   probe_local, n_way, k_shot, q_queries, device)
        logits, _ = model(s, q, n_way, k_shot)
        hit = (torch.argmax(logits, 1) == lbl).sum().item()
        per_episode.append(hit / lbl.size(0) * 100)
        correct += hit
        total += lbl.size(0)
    acc = correct / total * 100
    return (acc, ci95(per_episode)) if return_ci else acc


@torch.no_grad()
def eval_random_view(model, data, gallery_local, probe_local, episodes=500,
                     n_way=N_WAY, k_shot=K_SHOT, q_queries=Q_QUERIES,
                     device=DEVICE, return_ci=False):
    """CCR with the viewing angle resampled in every episode (same-view)."""
    model.eval()
    per_episode, correct, total = [], 0, 0
    for _ in range(episodes):
        v = np.random.randint(0, NUM_VIEWS)
        s, q, lbl = sample_episode(data, v, v, gallery_local, probe_local,
                                   n_way, k_shot, q_queries, device)
        logits, _ = model(s, q, n_way, k_shot)
        hit = (torch.argmax(logits, 1) == lbl).sum().item()
        per_episode.append(hit / lbl.size(0) * 100)
        correct += hit
        total += lbl.size(0)
    acc = correct / total * 100
    return (acc, ci95(per_episode)) if return_ci else acc


@torch.no_grad()
def eval_cross_view(model, data, gallery_local, probe_local, pview,
                    episodes=300, n_way=N_WAY, k_shot=K_SHOT,
                    q_queries=Q_QUERIES, device=DEVICE, num_views=NUM_VIEWS):
    """Cross-view CCR (%) for one probe angle, identical view EXCLUDED.

    Averaged over every gallery angle other than the probe angle.
    """
    model.eval()
    accs = []
    for gview in range(num_views):
        if gview == pview:
            continue                      # the identical-view case is excluded
        accs.append(eval_accuracy(model, data, gallery_local, probe_local,
                                  gview, pview, episodes=episodes,
                                  n_way=n_way, k_shot=k_shot,
                                  q_queries=q_queries, device=device))
    return float(np.mean(accs))


@torch.no_grad()
def run_ablation(model, data, gallery_local, probe_local, use_attention=True,
                 episodes=500, n_way=N_WAY, k_shot=K_SHOT,
                 q_queries=Q_QUERIES, device=DEVICE):
    """Same-view CCR with either the learned weights or a fixed uniform 1/P.

    Both variants share the SAME trained encoder, so only the part weighting
    differs. Note that the encoder was trained jointly with the attention, so
    the measured gap quantifies the contribution of the weighting within this
    jointly trained model; it is not the gap that would separate an
    independently optimised unweighted part metric from DPP-Net.
    """
    model.eval()
    correct = total = 0
    for _ in range(episodes):
        v = np.random.randint(0, NUM_VIEWS)
        s, q, lbl = sample_episode(data, v, v, gallery_local, probe_local,
                                   n_way, k_shot, q_queries, device)
        S = model.encoder(s).view(n_way, k_shot, NUM_PARTS, -1)
        proto = S.mean(dim=1)
        Q = model.encoder(q)
        dist = torch.sum((Q.unsqueeze(1) - proto.unsqueeze(0)) ** 2, dim=3)

        if use_attention:
            residual = Q - proto.mean(dim=0, keepdim=True)
            w = model.attention_net(
                torch.cat([Q.reshape(Q.size(0), -1),
                           residual.reshape(Q.size(0), -1)], 1)).unsqueeze(1)
        else:
            w = torch.full((Q.size(0), 1, NUM_PARTS), 1.0 / NUM_PARTS,
                           device=device)

        preds = torch.argmin(torch.sum(dist * w, dim=2), dim=1)
        correct += (preds == lbl).sum().item()
        total += lbl.size(0)
    return correct / total * 100


@torch.no_grad()
def mean_attention(model, data, gallery_local, probe_local, episodes=200,
                   n_way=N_WAY, k_shot=K_SHOT, q_queries=Q_QUERIES,
                   device=DEVICE, gview=None):
    """Mean attention vector over many episodes, normalised to sum to 1."""
    model.eval()
    acc = torch.zeros(NUM_PARTS, device=device)
    n = 0
    for _ in range(episodes):
        v = np.random.randint(0, NUM_VIEWS) if gview is None else gview
        s, q, _ = sample_episode(data, v, v, gallery_local, probe_local,
                                 n_way, k_shot, q_queries, device)
        _, w = model(s, q, n_way, k_shot)
        acc += w.sum(0)
        n += w.size(0)
    vec = (acc / n).cpu().numpy()
    return vec / vec.sum()


def cosine(a, b):
    """Cosine similarity between two attention vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

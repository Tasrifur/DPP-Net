"""Episode samplers for the N-way K-shot protocol."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import NUM_VIEWS, SEQS_PER_VIEW


def _global_indices(view, locals_, max_seq, svp=SEQS_PER_VIEW):
    base = view * svp
    idx = np.array([base + i for i in locals_ if base + i < max_seq])
    return idx


class ViewAwareEpisodeDataset(Dataset):
    """Same-view episodes: support and query drawn from ONE random view.

    Sampling both sides from the same angle is what keeps the same-view
    protocol honest; drawing them independently would leak cross-view
    information into a result reported as same-view.
    """

    def __init__(self, data, n_way, k_shot, q_queries, num_episodes,
                 gallery_local, probe_local,
                 num_views=NUM_VIEWS, seqs_per_view=SEQS_PER_VIEW):
        self.data = data
        self.n_subjects = data.shape[0]
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.num_episodes = num_episodes
        self.num_views = num_views
        self.svp = seqs_per_view
        self.gallery_local = gallery_local
        self.probe_local = probe_local

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, index):
        v = np.random.randint(0, self.num_views)
        max_s = self.data.shape[1]
        g_idx = _global_indices(v, self.gallery_local, max_s, self.svp)
        p_idx = _global_indices(v, self.probe_local, max_s, self.svp)
        if len(g_idx) == 0:
            g_idx = np.arange(min(self.k_shot, max_s))
        if len(p_idx) == 0:
            p_idx = np.arange(min(self.q_queries, max_s))

        selected = np.random.choice(self.n_subjects, self.n_way, replace=False)
        sup, qry = [], []
        for s in selected:
            seqs = self.data[s]
            sup.append(seqs[np.random.choice(g_idx, self.k_shot, replace=True)])
            qry.append(seqs[np.random.choice(p_idx, self.q_queries, replace=True)])
        labels = torch.arange(self.n_way).repeat_interleave(self.q_queries)
        return torch.cat(sup, 0), torch.cat(qry, 0), labels


class CrossViewEpisodeDataset(Dataset):
    """Cross-view episodes: support from one view, query from a different one.

    ``p_same`` mixes a fraction of same-view episodes back in so the model does
    not lose identical-view accuracy while learning angle invariance.
    """

    def __init__(self, data, n_way, k_shot, q_queries, num_episodes,
                 gallery_local, probe_local, p_same=0.2,
                 num_views=NUM_VIEWS, seqs_per_view=SEQS_PER_VIEW):
        self.data = data
        self.n_subjects = data.shape[0]
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.num_episodes = num_episodes
        self.num_views = num_views
        self.svp = seqs_per_view
        self.gallery_local = gallery_local
        self.probe_local = probe_local
        self.p_same = p_same

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, index):
        gview = np.random.randint(0, self.num_views)
        if np.random.rand() < self.p_same:
            pview = gview
        else:
            pview = np.random.randint(0, self.num_views)
            while pview == gview:
                pview = np.random.randint(0, self.num_views)

        max_s = self.data.shape[1]
        g_idx = _global_indices(gview, self.gallery_local, max_s, self.svp)
        p_idx = _global_indices(pview, self.probe_local, max_s, self.svp)
        if len(g_idx) == 0:
            g_idx = np.arange(min(self.k_shot, max_s))
        if len(p_idx) == 0:
            p_idx = np.arange(min(self.q_queries, max_s))

        selected = np.random.choice(self.n_subjects, self.n_way, replace=False)
        sup, qry = [], []
        for s in selected:
            seqs = self.data[s]
            sup.append(seqs[np.random.choice(g_idx, self.k_shot, replace=True)])
            qry.append(seqs[np.random.choice(p_idx, self.q_queries, replace=True)])
        labels = torch.arange(self.n_way).repeat_interleave(self.q_queries)
        return torch.cat(sup, 0), torch.cat(qry, 0), labels


def sample_episode(data, gview, pview, gallery_local, probe_local,
                   n_way, k_shot, q_queries, device, svp=SEQS_PER_VIEW):
    """Draw a single episode with explicit gallery and probe views.

    Used by the evaluation routines, which need to pin the angles rather than
    sample them.
    """
    max_s = data.shape[1]
    g_idx = _global_indices(gview, gallery_local, max_s, svp)
    p_idx = _global_indices(pview, probe_local, max_s, svp)
    if len(g_idx) == 0:
        g_idx = np.arange(min(k_shot, max_s))
    if len(p_idx) == 0:
        p_idx = np.arange(min(q_queries, max_s))

    selected = np.random.choice(data.shape[0], n_way, replace=False)
    sup, qry = [], []
    for s in selected:
        seqs = data[s]
        sup.append(seqs[np.random.choice(g_idx, k_shot, replace=True)])
        qry.append(seqs[np.random.choice(p_idx, q_queries, replace=True)])
    labels = torch.arange(n_way).repeat_interleave(q_queries).to(device)
    return (torch.cat(sup, 0).to(device), torch.cat(qry, 0).to(device), labels)

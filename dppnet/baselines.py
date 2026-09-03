"""The four canonical few-shot matching heads used in the controlled comparison.

Every head wraps the SAME encoder definition as the proposed model, so any
accuracy difference is attributable to the matching head alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HIDDEN_DIM, NUM_PARTS
from .model import PartAwareGaitEncoder


class ProtoNet(nn.Module):
    """Snell et al., 2017 - squared Euclidean distance to the class mean."""

    def __init__(self):
        super().__init__()
        self.enc = PartAwareGaitEncoder()

    def forward(self, s, q, n, k):
        S, Q = self.enc(s), self.enc(q)
        proto = S.view(n, k, NUM_PARTS, S.size(2)).mean(1).reshape(n, -1)
        return -(torch.cdist(Q.reshape(Q.size(0), -1), proto) ** 2)


class MatchingNet(nn.Module):
    """Vinyals et al., 2016 - attention over individual support samples."""

    def __init__(self):
        super().__init__()
        self.enc = PartAwareGaitEncoder()
        self.tau = nn.Parameter(torch.tensor(10.0))

    def forward(self, s, q, n, k):
        S, Q = self.enc(s), self.enc(q)
        Sf = F.normalize(S.reshape(n * k, -1), dim=1)
        Qf = F.normalize(Q.reshape(Q.size(0), -1), dim=1)
        attn = F.softmax(self.tau * (Qf @ Sf.t()), dim=1)
        lbl = torch.arange(n, device=Q.device).repeat_interleave(k)
        return torch.log(attn @ F.one_hot(lbl, n).float() + 1e-8)


class RelationNet(nn.Module):
    """Sung et al., 2018 - learned pairwise relation score."""

    def __init__(self):
        super().__init__()
        self.enc = PartAwareGaitEncoder()
        self.relation = nn.Sequential(
            nn.Linear(2 * NUM_PARTS * HIDDEN_DIM, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.tau = nn.Parameter(torch.tensor(10.0))

    def forward(self, s, q, n, k):
        S, Q = self.enc(s), self.enc(q)
        proto = F.normalize(
            S.view(n, k, NUM_PARTS, S.size(2)).mean(1).reshape(n, -1), dim=1)
        Qf = F.normalize(Q.reshape(Q.size(0), -1), dim=1)
        nq = Qf.size(0)
        pair = torch.cat([Qf.unsqueeze(1).expand(nq, n, -1),
                          proto.unsqueeze(0).expand(nq, n, -1)], 2)
        return self.tau * self.relation(pair.reshape(nq * n, -1)).view(nq, n)


class MetaBaseline(nn.Module):
    """Chen et al., 2021 - scaled cosine similarity to the class mean."""

    def __init__(self):
        super().__init__()
        self.enc = PartAwareGaitEncoder()
        self.tau = nn.Parameter(torch.tensor(10.0))

    def forward(self, s, q, n, k):
        S, Q = self.enc(s), self.enc(q)
        proto = F.normalize(
            S.view(n, k, NUM_PARTS, S.size(2)).mean(1).reshape(n, -1), dim=1)
        Qf = F.normalize(Q.reshape(Q.size(0), -1), dim=1)
        return self.tau * (Qf @ proto.t())


class DPPNetHead(nn.Module):
    """Proposed head, wrapped to the same (s, q, n, k) -> logits signature."""

    def __init__(self):
        super().__init__()
        self.enc = PartAwareGaitEncoder()
        self.attn = nn.Sequential(
            nn.Linear(2 * NUM_PARTS * HIDDEN_DIM, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, NUM_PARTS), nn.Softmax(dim=1))

    def forward(self, s, q, n, k):
        S, Q = self.enc(s), self.enc(q)
        proto = S.view(n, k, NUM_PARTS, S.size(2)).mean(1)
        r = Q - proto.mean(0, keepdim=True)
        w = self.attn(torch.cat([Q.reshape(Q.size(0), -1),
                                 r.reshape(r.size(0), -1)], 1))
        dist = ((Q.unsqueeze(1) - proto.unsqueeze(0)) ** 2).sum(3)
        return -(dist * w.unsqueeze(1)).sum(2)


# Order matches Table 7 in the paper.
METHODS = {
    "Prototypical Net": ProtoNet,
    "Matching Net": MatchingNet,
    "Relation Net": RelationNet,
    "Meta-Baseline": MetaBaseline,
    "DPP-Net (ours)": DPPNetHead,
}


def save_name(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "") + ".pt"

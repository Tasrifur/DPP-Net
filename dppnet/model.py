"""DPP-Net architecture: part-aware encoder + residual-conditioned attention."""

import torch
import torch.nn as nn

from .config import (DEVICE, EXPECTED_PARAMS, HIDDEN_DIM, IMAGE_CHANNELS,
                     NUM_PARTS)


class PartAwareGaitEncoder(nn.Module):
    """Three convolutional blocks followed by horizontal part pooling.

    Three pooling stages on a 64-pixel input give a feature-map height of 8,
    which matches NUM_PARTS exactly, so each strip is one feature row.

    Output: [B, NUM_PARTS, HIDDEN_DIM], e.g. [B, 8, 128].
    """

    def __init__(self, input_channels=IMAGE_CHANNELS, hidden_dim=HIDDEN_DIM,
                 num_parts=NUM_PARTS):
        super().__init__()
        self.num_parts = num_parts
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        feat = self.cnn(x)                                   # [B, C, H, W]
        _, _, H, _ = feat.size()
        ph = H // self.num_parts
        parts = [feat[:, :, i * ph:(i + 1) * ph, :].mean(dim=(2, 3)).unsqueeze(1)
                 for i in range(self.num_parts)]
        return torch.cat(parts, dim=1)                        # [B, P, D]


class DynamicProtoNetV3(nn.Module):
    """Residual-conditioned dynamic part attention over a prototypical metric.

    The attention MLP is conditioned on ``[query ; query - mean_prototype]``.
    The residual isolates where the query deviates from the episode average,
    which is exactly the signal that indicates which body parts have been
    disturbed by a covariate.

    ``forward`` returns ``(logits, weights)``:
        logits  : [N_q, N_way]  negative weighted distance, for cross-entropy
        weights : [N_q, P]      part attention weights, summing to 1
    """

    def __init__(self, encoder=None, hidden_dim=HIDDEN_DIM, num_parts=NUM_PARTS):
        super().__init__()
        self.encoder = encoder if encoder is not None else PartAwareGaitEncoder()
        self.num_parts = num_parts
        attn_input_dim = 2 * hidden_dim * num_parts
        self.attention_net = nn.Sequential(
            nn.Linear(attn_input_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_parts),
            nn.Softmax(dim=1),
        )

    def forward(self, support_set, query_set, n_way, k_shot):
        S = self.encoder(support_set)                        # [n*k, P, D]
        Q = self.encoder(query_set)                          # [N_q, P, D]
        D = S.size(2)

        proto = S.view(n_way, k_shot, self.num_parts, D).mean(dim=1)   # [n, P, D]
        mean_proto = proto.mean(dim=0, keepdim=True)                   # [1, P, D]
        residual = Q - mean_proto                                      # [N_q, P, D]

        attn_input = torch.cat(
            [Q.reshape(Q.size(0), -1), residual.reshape(Q.size(0), -1)], dim=1)
        weights = self.attention_net(attn_input)                       # [N_q, P]

        dist = torch.sum((Q.unsqueeze(1) - proto.unsqueeze(0)) ** 2, dim=3)
        logits = -torch.sum(dist * weights.unsqueeze(1), dim=2)        # [N_q, n]
        return logits, weights


def build_model(device=DEVICE, verify=True):
    """Instantiate DPP-Net and assert the published parameter count."""
    model = DynamicProtoNetV3(PartAwareGaitEncoder()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verify and n_params != EXPECTED_PARAMS:
        raise RuntimeError(
            f"Architecture mismatch: {n_params:,} parameters, "
            f"expected {EXPECTED_PARAMS:,}. The paper reports "
            f"{EXPECTED_PARAMS:,}; do not proceed with a modified encoder.")
    return model


def load_model(ckpt_path, device=DEVICE):
    """Build the model and load a trained checkpoint."""
    model = build_model(device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model

"""Stage 3 — models (edge flux takes pipe conductance `k_ij` as input).

Common interface `forward(E, i_idx, j_idx, B, S, k) -> (dEdt, internal_imbalance)`.
`internal_imbalance = sum_i(dE_i - S_i)` — exactly 0 for the hard mimetic model,
a real number to penalise/measure otherwise.

All classes are verbatim from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgeFluxNet(nn.Module):
    """(E_i, E_j, k_ij) -> scalar flux. Shared by hard + soft so capacity matches."""

    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, E, i_idx, j_idx, k):
        x = torch.stack([E[i_idx], E[j_idx], k], dim=-1)
        return self.net(x).squeeze(-1)


class MimeticOperator(nn.Module):          # HARD conservation: dE = B@F + S
    def __init__(self, hidden=32):
        super().__init__()
        self.flux = EdgeFluxNet(hidden)

    def forward(self, E, i_idx, j_idx, B, S, k):
        dEdt = B @ self.flux(E, i_idx, j_idx, k) + S
        return dEdt, torch.zeros((), device=E.device)


class SoftMimeticOperator(nn.Module):      # ABLATION: same flux net + node leak g, penalised
    def __init__(self, hidden=32):
        super().__init__()
        self.flux = EdgeFluxNet(hidden)
        self.leak = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, E, i_idx, j_idx, B, S, k):
        g = self.leak(E.unsqueeze(-1)).squeeze(-1)
        dEdt = B @ self.flux(E, i_idx, j_idx, k) + S + g
        return dEdt, g.sum()


class BlackBoxGNN(nn.Module):              # unstructured baseline, also gets k
    def __init__(self, hidden=32):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        self.upd = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, E, i_idx, j_idx, B, S, k):
        n = E.shape[0]
        m_ij = self.msg(torch.stack([E[i_idx], E[j_idx], k], -1))
        m_ji = self.msg(torch.stack([E[j_idx], E[i_idx], k], -1))
        agg = torch.zeros(n, m_ij.shape[-1])
        agg.index_add_(0, j_idx, m_ij)
        agg.index_add_(0, i_idx, m_ji)
        dEdt = self.upd(torch.cat([agg, S.unsqueeze(-1)], -1)).squeeze(-1)
        return dEdt, (dEdt - S).sum()


#: name -> class, used by checkpoint loading and the CLI
MODEL_CLASSES = {
    "Mimetic (hard)": MimeticOperator,
    "Soft (ablation)": SoftMimeticOperator,
    "Black-box GNN": BlackBoxGNN,
}

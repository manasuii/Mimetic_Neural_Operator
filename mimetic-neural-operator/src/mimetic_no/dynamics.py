"""Stage 1-2 — ground-truth physics: pipe flux law, sources, simulators.

Also holds the nonlinear flux law and its simulator used by robustness
experiment (D), and the Stage-2 closed-system conservation unit test.
All bodies are verbatim from the notebook.
"""

from __future__ import annotations

import numpy as np
import torch

from .graph import build_graph, edge_index, incidence_matrix


def true_flux(E, i_idx, j_idx, k):
    """Ground-truth Fourier-type pipe law F_ij = k_ij (E_i - E_j)."""
    return k * (E[i_idx] - E[j_idx])


def source_term(t, n_nodes, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.5, 2.0, n_nodes)
    p = rng.uniform(0, 2 * np.pi, n_nodes)
    a = rng.uniform(-1, 1, n_nodes)
    return torch.tensor(a * np.sin(f * t + p), dtype=torch.float32)


def simulate_true(n_nodes, edges, B, k, T=200, dt=0.05, E0=None, seed=0, sources=True):
    i_idx, j_idx = edge_index(edges)
    E = torch.rand(n_nodes) if E0 is None else E0.clone()
    traj = [E.clone()]
    for step in range(T):
        S = source_term(step * dt, n_nodes, seed=seed) if sources else torch.zeros(n_nodes)
        E = E + dt * (B @ true_flux(E, i_idx, j_idx, k) + S)
        traj.append(E.clone())
    return torch.stack(traj)


# --------------------------------------------------------------------------
# Stage 2 unit test — closed system (no sources) conserves to machine precision
# --------------------------------------------------------------------------
# Isolated network, S=0: total energy must be constant for ANY flux law, because
# sum(B @ F) = 0 exactly. Run in float64 to expose the true (machine-precision) residual.
def conservation_unit_test(seed=3, T=400, dt=0.05):
    n, e, kf = build_graph(seed=seed)
    Bd = incidence_matrix(n, e).double()
    kd = kf.double()
    i_idx, j_idx = edge_index(e)
    E = torch.rand(n, dtype=torch.float64)
    E0sum = E.sum().item()
    res = [0.0]
    for _ in range(T):
        E = E + dt * (Bd @ true_flux(E, i_idx, j_idx, kd))  # no source term
        res.append(abs(E.sum().item() - E0sum))
    return np.array(res)


# --------------------------------------------------------------------------
# Experiment (D) — nonlinear flux law and its simulator
# --------------------------------------------------------------------------
def flux_nonlinear(E, i_idx, j_idx, k, alpha=0.5):
    return k * (E[i_idx] - E[j_idx]) * (1.0 + alpha * 0.5 * (E[i_idx] + E[j_idx]))


def simulate_nl(n, edges, B, k, T, dt, E0, seed):
    i_idx, j_idx = edge_index(edges)
    E = E0.clone()
    tr = [E.clone()]
    for s in range(T):
        S = source_term(s * dt, n, seed=seed)
        E = E + dt * (B @ flux_nonlinear(E, i_idx, j_idx, k) + S)
        tr.append(E.clone())
    return torch.stack(tr)

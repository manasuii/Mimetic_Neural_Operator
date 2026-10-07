"""Stage 1 — graphs, hand-built signed incidence operator, edge indexing.

Functions are copied verbatim from the notebook.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch


def build_graph(n_nodes=15, edge_prob=0.25, seed=None, k_low=0.5, k_high=1.5):
    """Random connected graph + per-edge pipe conductance k_ij ~ U(k_low,k_high)."""
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=int(rng.integers(0, 1_000_000)))
    edges = [(min(i, j), max(i, j)) for i, j in G.edges()]
    k = torch.tensor(rng.uniform(k_low, k_high, len(edges)), dtype=torch.float32)
    return n_nodes, edges, k


def incidence_matrix(n_nodes, edges):
    """Signed incidence B: one -1 and one +1 per column => every column sums to 0.
    This is what makes B @ F conserve energy for ANY flux F (discrete divergence thm)."""
    B = torch.zeros(n_nodes, len(edges))
    for e, (i, j) in enumerate(edges):
        B[i, e] = -1.0
        B[j, e] = 1.0
    return B


def edge_index(edges):
    return torch.tensor([e[0] for e in edges]), torch.tensor([e[1] for e in edges])

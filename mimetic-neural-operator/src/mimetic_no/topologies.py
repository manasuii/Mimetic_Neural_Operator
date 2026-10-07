"""Stage 6 — structured & realistic district-heating topology generators.

Each returns (n_nodes, edges, k) exactly like `build_graph`, so they drop
straight into `incidence_matrix(...)` and `rollout_eval(...)`.
All bodies are verbatim from the notebook.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from scipy.spatial import Delaunay


def _finalize_top(G, seed, k_low=0.5, k_high=1.5):
    """Largest connected component -> contiguous 0..n-1 labels -> (n, edges, k_tensor)."""
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    n = G.number_of_nodes()
    edges = sorted((min(int(i), int(j)), max(int(i), int(j))) for i, j in G.edges())
    k = torch.tensor(np.random.default_rng(seed).uniform(k_low, k_high, len(edges)),
                     dtype=torch.float32)
    return n, edges, k


def build_tree_graph(n_nodes=30, seed=None, **kw):
    """Radial: central plant (node 0) feeding branching substations (random recursive tree)."""
    r = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for node in range(1, n_nodes):
        G.add_edge(int(r.integers(0, node)), node)   # attach to an earlier node -> tree rooted at 0
    return _finalize_top(G, seed, **kw)


def build_delaunay_graph(n_nodes=30, seed=None, **kw):
    """Planar street-level looped mesh: Delaunay triangulation of random 2-D points."""
    r = np.random.default_rng(seed)
    pts = r.uniform(0, 1, (n_nodes, 2))
    tri = Delaunay(pts)
    G = nx.Graph()
    for s in tri.simplices:
        G.add_edges_from([(int(s[0]), int(s[1])), (int(s[1]), int(s[2])), (int(s[0]), int(s[2]))])
    return _finalize_top(G, seed, **kw)


def build_grid_graph(rows=5, cols=6, seed=None, **kw):
    """Regular street lattice with loops."""
    return _finalize_top(nx.grid_2d_graph(rows, cols), seed, **kw)


def build_geometric_graph(n_nodes=30, seed=None, radius=0.34, **kw):
    """Realistic spatial planar-ish network: pipes connect nodes within a distance radius."""
    return _finalize_top(nx.random_geometric_graph(n_nodes, radius, seed=seed), seed, **kw)


def build_pandapipes_graph(seed=0, **kw):
    """Real utility topology via pandapipes if installed; otherwise a spatial fallback.
    To enable the real network:  pip install pandapipes"""
    try:
        import pandapipes as pp  # noqa: F401
        from pandapipes.networks import water_network_schutterwald as _example_net
        net = _example_net()
        G = nx.Graph()
        for _, row in net.pipe.iterrows():
            G.add_edge(int(row["from_junction"]), int(row["to_junction"]))
        print(f"  [pandapipes] loaded example net: {G.number_of_nodes()} junctions, "
              f"{G.number_of_edges()} pipes")
        return _finalize_top(G, seed, **kw)
    except Exception as ex:
        print(f"  [pandapipes] unavailable ({type(ex).__name__}); using NetworkX spatial "
              f"fallback. (pip install pandapipes to use a real utility topology.)")
        return build_geometric_graph(40, seed=seed, radius=0.30, **kw)

"""All figures from the notebook, one function each.

The plotting commands are the notebook's verbatim; the only changes are that
each block is wrapped in a function, figures are written under
`results/figures/`, and `plt.show()` is behind a `show` flag so the scripts also
run headless (CI, servers, `--no-show`).
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import config
from .evaluation import robust
from .topologies import (build_delaunay_graph, build_geometric_graph,
                         build_grid_graph, build_tree_graph)


def use_headless_backend() -> None:
    """Switch Matplotlib to Agg — call before plotting when no display exists."""
    matplotlib.use("Agg")


def _finish(fname: str, show: bool) -> str:
    path = config.figure(fname)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()
    print(f"  figure -> {path}")
    return path


# --------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------
def plot_unit_test(res64, show=False):
    plt.figure(figsize=(7, 4))
    plt.semilogy(np.clip(res64, 1e-18, None))
    plt.xlabel("timestep")
    plt.ylabel("|total energy - initial| (log)")
    plt.title("Stage 2 unit test: closed-system conservation residual (float64)")
    return _finish("stage2_unit_test.png", show)


# --------------------------------------------------------------------------
# Stage 5 — headline figures
# --------------------------------------------------------------------------
def plot_claim1_conservation(res, show=False):
    plt.figure(figsize=(8, 5))
    for nm, r in res.items():
        plt.plot(robust(r["drift"]), label=nm)
    plt.yscale("log")
    plt.xlabel("timestep")
    plt.ylabel("|total energy - expected| (log)")
    plt.title("Claim 1: conservation drift over long rollout")
    plt.legend()
    return _finish("claim1_conservation.png", show)


def plot_claim2_accuracy(res, show=False):
    plt.figure(figsize=(8, 5))
    for nm, r in res.items():
        plt.plot(robust(r["rel_rmse"], 1e-6, 1e2),
                 label=nm + (" [diverged]" if r["diverged"] else ""))
    plt.yscale("log")
    plt.ylim(1e-4, 1e2)
    plt.xlabel("timestep")
    plt.ylabel("relative RMSE vs truth (log, capped)")
    plt.title("Claim 2: long-horizon accuracy")
    plt.legend()
    return _finish("claim2_accuracy.png", show)


# --------------------------------------------------------------------------
# Stage 5+ robustness
# --------------------------------------------------------------------------
def plot_experiment_A(lam_grid, lam_drift, lam_rel, hard_ref, show=False):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.semilogy(lam_grid, robust(lam_drift), "o-", color="tab:red", label="soft drift")
    ax1.axhline(hard_ref, ls="--", color="k", label="hard mimetic drift")
    ax1.set_xlabel("penalty weight lambda")
    ax1.set_ylabel("conservation drift (log)", color="tab:red")
    ax2 = ax1.twinx()
    ax2.semilogy(lam_grid, robust(lam_rel, 1e-6, 1e3), "s--", color="tab:blue")
    ax2.set_ylabel("relative RMSE (log)", color="tab:blue")
    ax1.set_title("(A) soft penalty never reaches hard-structure conservation")
    ax1.legend(loc="center right")
    return _finish("expA_lambda.png", show)


def plot_experiment_B(K_grid, K_rel, show=False):
    plt.figure(figsize=(7, 4))
    plt.semilogy(K_grid, robust(K_rel, 1e-4, 1e3), "o-")
    plt.xlabel("K (rollout-training steps)")
    plt.ylabel("final relative RMSE (log)")
    plt.title("(B) exposure-bias fix: accuracy vs K (black-box)")
    return _finish("expB_K.png", show)


def plot_experiment_E(scale_n, mim_s, bb_s, show=False):
    plt.figure(figsize=(7, 4))
    plt.plot(scale_n, robust(mim_s, 1e-4, 1e2), "o-", label="Mimetic (hard)")
    plt.plot(scale_n, robust(bb_s, 1e-4, 1e2), "s-", label="Black-box GNN")
    plt.yscale("log")
    plt.xlabel("graph size (nodes)")
    plt.ylabel("final relative RMSE (log)")
    plt.title("(E) generalization to larger graphs")
    plt.legend()
    return _finish("expE_scale.png", show)


# --------------------------------------------------------------------------
# Stage 6
# --------------------------------------------------------------------------
def plot_topology_gallery(show=False):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    gallery = [("Tree / radial",           build_tree_graph(30, seed=1)),
               ("Delaunay (planar loops)", build_delaunay_graph(30, seed=1)),
               ("Grid (street lattice)",   build_grid_graph(5, 6, seed=1)),
               ("Geometric (spatial)",     build_geometric_graph(30, seed=1))]
    for ax, (title, (n, e, k)) in zip(axs, gallery):
        G = nx.Graph(e)
        pos = nx.spring_layout(G, seed=1)
        nx.draw(G, pos, ax=ax, node_size=45, width=0.8,
                node_color="tab:blue", edge_color="0.5")
        ax.set_title(f"{title}\n(n={n}, {len(e)} edges)")
    return _finish("stage6_topology_gallery.png", show)


def plot_topology_drift(topo_res, families, models, topo_T, show=False):
    # summary bars: mean final conservation drift per family per model (log scale)
    fam_names = list(families)
    mdl_names = list(models)
    x = np.arange(len(fam_names))
    w = 0.25
    plt.figure(figsize=(11, 5))
    for mi, nm in enumerate(mdl_names):
        vals = [max(np.mean(topo_res[f][nm]["drift"]), 1e-8) for f in fam_names]
        plt.bar(x + (mi - 1) * w, vals, width=w, label=nm)
    plt.yscale("log")
    plt.xticks(x, fam_names, rotation=15)
    plt.ylabel("final conservation drift (log)")
    plt.title(f"Stage 6: conservation drift across DH topologies ({topo_T}-step rollout)")
    plt.legend()
    return _finish("stage6_topology_drift.png", show)

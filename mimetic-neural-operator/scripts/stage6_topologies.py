"""Stage 6 — structured & realistic district-heating topologies (external-validity test).

Tests the already-trained mimetic / soft / blackbox models (fit on 15-node ER
graphs) on tree, Delaunay, grid, geometric and (optionally) pandapipes
topologies. Conductances are held in the TRAINING range [0.5,1.5] so any
degradation is attributable to TOPOLOGY, not to a conductance shift.

    python scripts/stage6_topologies.py [--show] [--skip-pandapipes]

Writes stage6_topology_gallery.png and stage6_topology_drift.png.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import networkx as nx
import numpy as np
import torch

from mimetic_no import config
from mimetic_no.evaluation import rollout_eval
from mimetic_no.graph import build_graph, incidence_matrix
from mimetic_no.plotting import (plot_topology_drift, plot_topology_gallery,
                                 use_headless_backend)
from mimetic_no.topologies import (build_delaunay_graph, build_geometric_graph,
                                   build_grid_graph, build_pandapipes_graph,
                                   build_tree_graph)
from mimetic_no.training import load_all_models


def structural_sanity() -> None:
    # structural sanity: connected, and incidence columns sum to zero (the mimetic guarantee)
    print(f"{'family':12s}{'n':>4}{'edges':>7}{'max|colsum(B)|':>16}{'connected':>11}")
    for nm, gen in [("tree", lambda s: build_tree_graph(30, seed=s)),
                    ("delaunay", lambda s: build_delaunay_graph(30, seed=s)),
                    ("grid", lambda s: build_grid_graph(5, 6, seed=s)),
                    ("geometric", lambda s: build_geometric_graph(30, seed=s))]:
        n, e, k = gen(0)
        B = incidence_matrix(n, e)
        print(f"{nm:12s}{n:4d}{len(e):7d}{B.sum(0).abs().max().item():16.1e}"
              f"{str(nx.is_connected(nx.Graph(e))):>11}")


def main(show: bool = False, skip_pandapipes: bool = False) -> None:
    if not show:
        use_headless_backend()
    config.set_seeds()

    models = load_all_models()

    structural_sanity()
    plot_topology_gallery(show=show)

    # 400-500 step rollout on structured/realistic topologies.
    # Metrics: final conservation drift + final relative state RMSE (mean +- std over instances).
    # Conductances held in the TRAINING range [0.5,1.5] -> isolates the effect of TOPOLOGY.
    TOPO_T = config.TOPO_T
    N_INST = config.N_INST
    families = {
        "ER (reference)":      lambda s: build_graph(30, 0.12, seed=s),   # train-like structure, size-matched
        "Tree / radial":       lambda s: build_tree_graph(30, seed=s),
        "Delaunay (planar)":   lambda s: build_delaunay_graph(30, seed=s),
        "Grid (loops)":        lambda s: build_grid_graph(5, 6, seed=s),
        "Geometric (spatial)": lambda s: build_geometric_graph(30, seed=s, radius=0.34),
    }

    topo_res = {fam: {nm: dict(drift=[], rel=[], div=0) for nm in models} for fam in families}
    for fam, gen in families.items():
        for inst in range(N_INST):
            n, e, k = gen(3000 + 17 * inst)
            B = incidence_matrix(n, e)
            E0 = torch.rand(n)
            for nm, m in models.items():
                r = rollout_eval(m, n, e, B, k, T=TOPO_T, E0=E0, seed=1234 + inst)
                topo_res[fam][nm]["drift"].append(r["drift"][-1])
                topo_res[fam][nm]["rel"].append(r["rel_rmse"][-1])
                topo_res[fam][nm]["div"] += int(r["diverged"])

    print(f"{'topology':20s}{'model':16s}{'final drift (mean+-std)':>26s}"
          f"{'final relRMSE (mean+-std)':>28s}{'div':>7s}")
    print("-" * 97)
    for fam in families:
        for nm in models:
            d = np.array(topo_res[fam][nm]["drift"])
            r = np.array(topo_res[fam][nm]["rel"])
            print(f"{fam:20s}{nm:16s}{d.mean():>13.2e}+-{d.std():<10.1e}"
                  f"{r.mean():>14.2e}+-{r.std():<11.1e}{topo_res[fam][nm]['div']:>4d}/{N_INST}")
        print()

    plot_topology_drift(topo_res, families, models, TOPO_T, show=show)

    if skip_pandapipes:
        return

    # (optional) Real utility topology via pandapipes
    n, e, k = build_pandapipes_graph(seed=0)
    B = incidence_matrix(n, e)
    E0 = torch.rand(n)
    print(f"\n{'model':16s}{'final drift':>14s}{'final relRMSE':>16s}{'diverged':>10s}")
    for nm, m in models.items():
        r = rollout_eval(m, n, e, B, k, T=450, E0=E0, seed=1234)
        print(f"{nm:16s}{r['drift'][-1]:14.2e}{r['rel_rmse'][-1]:16.2e}{str(r['diverged']):>10s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 6: structured DH topologies")
    ap.add_argument("--show", action="store_true", help="display figures interactively")
    ap.add_argument("--skip-pandapipes", action="store_true",
                    help="skip the optional real-utility-topology run")
    args = ap.parse_args()
    main(show=args.show, skip_pandapipes=args.skip_pandapipes)

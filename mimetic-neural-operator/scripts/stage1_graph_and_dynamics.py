"""Stage 1 — graph, hand-built incidence operator, true dynamics.

Builds the demo graph and prints the incidence summary line from the notebook,
then runs a short ground-truth simulation to show the dynamics work.

    python scripts/stage1_graph_and_dynamics.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

import argparse

from mimetic_no import config
from mimetic_no.dynamics import simulate_true
from mimetic_no.graph import build_graph, incidence_matrix


def main(seed: int = config.DEMO_GRAPH_SEED) -> None:
    config.set_seeds()

    n_nodes, edges, k = build_graph(seed=seed)
    B = incidence_matrix(n_nodes, edges)
    print("incidence", B.shape, "| max|colsum|=", B.sum(0).abs().max().item(),
          "| #edges", len(edges), "| k in [%.2f,%.2f]" % (k.min(), k.max()))

    traj = simulate_true(n_nodes, edges, B, k, T=200, dt=0.05, seed=0)
    print(f"simulated ground-truth trajectory: {tuple(traj.shape)} (timesteps+1, nodes)")
    print(f"  E[0] sum = {traj[0].sum().item():.4f}   E[-1] sum = {traj[-1].sum().item():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 1: graph + incidence + true dynamics")
    ap.add_argument("--seed", type=int, default=config.DEMO_GRAPH_SEED,
                    help="graph seed (default: %(default)s)")
    args = ap.parse_args()
    main(seed=args.seed)

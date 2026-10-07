"""Stage 5 — evaluation: RMSE, conservation drift, long-horizon plausibility, OOD.

Requires stages 4 and 3.

    python scripts/stage5_evaluate.py [--show]

Writes results/figures/claim1_conservation.png and claim2_accuracy.png.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import numpy as np
import torch

from mimetic_no import config
from mimetic_no.data import load_trajectories
from mimetic_no.evaluation import eval_split, rollout_eval
from mimetic_no.graph import build_graph, incidence_matrix
from mimetic_no.plotting import (plot_claim1_conservation, plot_claim2_accuracy,
                                 use_headless_backend)
from mimetic_no.training import load_all_models


def main(show: bool = False) -> None:
    if not show:
        use_headless_backend()
    config.set_seeds()

    _, id_test, ood_test = load_trajectories()
    models = load_all_models()

    # in-distribution test graph
    n_t, e_t, k_t = build_graph(**config.EVAL_GRAPH)
    B_t = incidence_matrix(n_t, e_t)
    E0 = torch.rand(n_t)
    res = {nm: rollout_eval(m, n_t, e_t, B_t, k_t, T=config.EVAL_T, E0=E0)
           for nm, m in models.items()}
    for nm, r in res.items():
        print(f"{nm:16s} drift_final={r['drift'][-1]:.2e}  "
              f"relRMSE_final={r['rel_rmse'][-1]:.2e}  diverged={r['diverged']}")

    # headline figures
    plot_claim1_conservation(res, show=show)
    plot_claim2_accuracy(res, show=show)

    # long-horizon physical plausibility (roll far beyond training horizon T=120)
    print()
    print(f"{'model':16s} {'min_energy':>12s} {'max_abs':>12s} {'neg_frac':>10s} {'runaway':>9s}")
    for nm, m in models.items():
        r = rollout_eval(m, n_t, e_t, B_t, k_t, T=config.PLAUSIBILITY_T, E0=E0)  # ~7.5x training horizon
        p = r["plaus"]
        print(f"{nm:16s} {p['min_energy']:12.3e} {p['max_abs']:12.3e} "
              f"{p['neg_frac']:10.2f} {str(p['runaway']):>9s}")

    # OOD generalization — unseen larger topology + shifted conductance range
    for split_name, split in [("IN-DISTRIBUTION", id_test),
                              ("OOD (22 nodes, k in [1.5,2.5])", ood_test)]:
        o = eval_split(split, models, T=config.SPLIT_T)
        print(f"\n== {split_name} ==")
        print(f"{'model':16s} {'drift(mean+-std)':>24s} "
              f"{'relRMSE(mean+-std)':>24s} {'diverged':>10s}")
        for nm, d in o.items():
            dr, rl = np.array(d["drift"]), np.array(d["rel"])
            print(f"{nm:16s} {dr.mean():.2e}+-{dr.std():.1e}   "
                  f"{rl.mean():.2e}+-{rl.std():.1e}   {d['div']}/{len(split)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 5: full evaluation")
    ap.add_argument("--show", action="store_true", help="display figures interactively")
    args = ap.parse_args()
    main(show=args.show)

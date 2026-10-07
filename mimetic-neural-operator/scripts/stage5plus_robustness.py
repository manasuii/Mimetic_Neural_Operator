"""Stage 5+ — robustness & sensitivity (reviewer-response experiments A-E).

(A) does soft-penalty conservation drift *regardless* of weight lambda;
(B) accuracy vs the K-step rollout horizon;
(C) is conservation integrator-dependent;
(D) does structure preservation survive a nonlinear flux law;
(E) does it hold on larger graphs than trained on.

    python scripts/stage5plus_robustness.py              # all experiments
    python scripts/stage5plus_robustness.py --only C E   # pick a subset

Requires stages 4 and 3. Writes expA_lambda.png, expB_K.png, expE_scale.png.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import torch

from mimetic_no import config
from mimetic_no.data import make_nl, make_trajectories
from mimetic_no.evaluation import rollout_eval, rollout_integrator, rollout_nl
from mimetic_no.graph import build_graph, incidence_matrix
from mimetic_no.models import BlackBoxGNN, MimeticOperator, SoftMimeticOperator
from mimetic_no.plotting import (plot_experiment_A, plot_experiment_B,
                                 plot_experiment_E, use_headless_backend)
from mimetic_no.training import load_all_models, train_model

ALL = ["A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------
def experiment_A(ctx, show=False):
    """(A) lambda sweep — does soft-penalty conservation drift regardless of tuning?"""
    n_t, e_t, B_t, k_t, E0 = ctx["graph"]
    sweep_trajs = ctx["sweep_trajs"]
    hard_ref = ctx["hard_ref"]
    lam_grid = config.LAM_GRID
    lam_drift, lam_rel = [], []
    print(f"hard mimetic reference drift = {hard_ref:.2e}")
    print(f"{'lambda':>8} {'drift_final':>12} {'relRMSE_final':>14}")
    for lam in lam_grid:
        sm = train_model(SoftMimeticOperator(), sweep_trajs, epochs=8, K=4, lam=lam)
        r = rollout_eval(sm, n_t, e_t, B_t, k_t, T=500, E0=E0)
        lam_drift.append(r["drift"][-1])
        lam_rel.append(r["rel_rmse"][-1])
        print(f"{lam:8.1f} {r['drift'][-1]:12.2e} {r['rel_rmse'][-1]:14.2e}")
    plot_experiment_A(lam_grid, lam_drift, lam_rel, hard_ref, show=show)


def experiment_B(ctx, show=False):
    """(B) K sweep — sensitivity to rollout-training horizon (black-box)."""
    n_t, e_t, B_t, k_t, E0 = ctx["graph"]
    sweep_trajs = ctx["sweep_trajs"]
    K_grid = config.K_GRID
    K_rel, K_drift = [], []
    print(f"{'K':>4} {'relRMSE_final':>14} {'drift_final':>12}")
    for Kv in K_grid:
        b = train_model(BlackBoxGNN(), sweep_trajs, epochs=8, K=Kv, lam=0.0)
        r = rollout_eval(b, n_t, e_t, B_t, k_t, T=500, E0=E0)
        K_rel.append(r["rel_rmse"][-1])
        K_drift.append(r["drift"][-1])
        print(f"{Kv:4d} {r['rel_rmse'][-1]:14.2e} {r['drift'][-1]:12.2e}")
    plot_experiment_B(K_grid, K_rel, show=show)


def experiment_C(ctx, show=False):
    """(C) Integrator robustness — conservation is a property of the operator, not the stepper."""
    n_t, e_t, B_t, k_t, _ = ctx["graph"]
    models = ctx["models"]
    print(f"{'model':16}{'euler':>12}{'backward':>12}{'rk4':>12}")
    for nm in ["Mimetic (hard)", "Black-box GNN"]:
        m = models[nm]
        vals = [rollout_integrator(m, n_t, e_t, B_t, k_t, integ=ig)
                for ig in ["euler", "backward", "rk4"]]
        print(f"{nm:16}" + "".join(f"{v:12.2e}" for v in vals))


def experiment_D(ctx, show=False):
    """(D) Nonlinear flux law — F = k(E_i - E_j)(1 + 0.5*avg): structure survives nonlinearity."""
    n_t, e_t, B_t, k_t, E0 = ctx["graph"]
    nl_train = make_nl(12, 0)
    mim_nl = train_model(MimeticOperator(), nl_train, epochs=8, K=4, lam=0.0)
    soft_nl = train_model(SoftMimeticOperator(), nl_train, epochs=8, K=4, lam=1.0)
    bb_nl = train_model(BlackBoxGNN(), nl_train, epochs=8, K=4, lam=0.0)
    print(f"{'model (nonlinear data)':22}{'drift_final':>12}{'relRMSE_final':>14}")
    for nm, m in [("Mimetic (hard)", mim_nl), ("Soft (ablation)", soft_nl),
                  ("Black-box GNN", bb_nl)]:
        d, rl = rollout_nl(m, n_t, e_t, B_t, k_t, T=500, E0=E0)
        print(f"{nm:22}{d:12.2e}{rl:14.2e}")


def experiment_E(ctx, show=False):
    """(E) Larger-graph scale — trained on 15 nodes, evaluated on 30/60/100."""
    models = ctx["models"]
    mimetic = models["Mimetic (hard)"]
    blackbox = models["Black-box GNN"]
    print(f"{'n_nodes':>8}{'mim_drift':>12}{'mim_relRMSE':>13}{'bb_drift':>12}{'bb_relRMSE':>12}")
    scale_n = config.SCALE_N
    mim_s, bb_s = [], []
    for N in scale_n:
        n, e, kk = build_graph(N, 0.12, seed=7000 + N)
        Bn = incidence_matrix(n, e)
        e0 = torch.rand(n)
        rm = rollout_eval(mimetic, n, e, Bn, kk, T=400, E0=e0)
        rb = rollout_eval(blackbox, n, e, Bn, kk, T=400, E0=e0)
        mim_s.append(rm["rel_rmse"][-1])
        bb_s.append(rb["rel_rmse"][-1])
        print(f"{N:8d}{rm['drift'][-1]:12.2e}{rm['rel_rmse'][-1]:13.2e}"
              f"{rb['drift'][-1]:12.2e}{rb['rel_rmse'][-1]:12.2e}")
    plot_experiment_E(scale_n, mim_s, bb_s, show=show)


EXPERIMENTS = {"A": experiment_A, "B": experiment_B, "C": experiment_C,
               "D": experiment_D, "E": experiment_E}


# --------------------------------------------------------------------------
def main(only=None, show: bool = False) -> None:
    if not show:
        use_headless_backend()
    config.set_seeds()

    selected = [x.upper() for x in (only or ALL)]
    models = load_all_models()

    n_t, e_t, k_t = build_graph(**config.EVAL_GRAPH)
    B_t = incidence_matrix(n_t, e_t)
    E0 = torch.rand(n_t)

    ctx = {"graph": (n_t, e_t, B_t, k_t, E0), "models": models}

    if "A" in selected or "B" in selected:
        ctx["sweep_trajs"] = make_trajectories(**config.SWEEP_SPEC)
    if "A" in selected:
        ref = rollout_eval(models["Mimetic (hard)"], n_t, e_t, B_t, k_t,
                           T=config.EVAL_T, E0=E0)
        ctx["hard_ref"] = ref["drift"][-1]

    for key in ALL:
        if key not in selected:
            continue
        print(f"\n===== Experiment ({key}) =====")
        EXPERIMENTS[key](ctx, show=show)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 5+: robustness experiments A-E")
    ap.add_argument("--only", nargs="+", choices=ALL + [x.lower() for x in ALL],
                    help="run only these experiments, e.g. --only C E")
    ap.add_argument("--show", action="store_true", help="display figures interactively")
    args = ap.parse_args()
    main(only=args.only, show=args.show)

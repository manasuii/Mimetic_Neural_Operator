"""Stage 5 — evaluation: RMSE, conservation drift, long-horizon plausibility, OOD,
integrator sweep, nonlinear-data rollout.

All function bodies are verbatim from the notebook. The single interface change
is that `eval_split` takes the `models` dict as an argument instead of reading it
from a notebook global — the computation it performs is identical.
"""

from __future__ import annotations

import numpy as np
import torch

from .dynamics import flux_nonlinear, source_term, true_flux
from .graph import edge_index


def rollout_eval(model, n_nodes, edges, B, k, T=500, dt=0.05, E0=None, seed=99):
    i_idx, j_idx = edge_index(edges)
    E0 = torch.rand(n_nodes) if E0 is None else E0.clone()
    Et, Em = E0.clone(), E0.clone()
    tt = [Et.clone()]
    mt = [Em.clone()]
    tot = [Em.sum().item()]
    exp = [Em.sum().item()]
    cS = 0.0
    div = False
    with torch.no_grad():
        for s in range(T):
            S = source_term(s * dt, n_nodes, seed=seed)
            Et = Et + dt * (B @ true_flux(Et, i_idx, j_idx, k) + S)
            tt.append(Et.clone())
            dEdt, _ = model(Em, i_idx, j_idx, B, S, k)
            Em = Em + dt * dEdt
            if not torch.isfinite(Em).all():
                div = True
                Em = torch.nan_to_num(Em, nan=0., posinf=1e6, neginf=-1e6)
            mt.append(Em.clone())
            tot.append(Em.sum().item())
            cS += dt * S.sum().item()
            exp.append(exp[0] + cS)
    tt = torch.stack(tt)
    mt = torch.stack(mt)
    rmse = torch.sqrt(((tt - mt) ** 2).mean(1)).numpy()
    scale = torch.sqrt((tt ** 2).mean(1)).clamp_min(1e-8).numpy()
    drift = np.abs(np.array(tot) - np.array(exp))
    # long-horizon physical plausibility of the MODEL trajectory
    ref = tt.abs().max().item()
    plaus = dict(min_energy=mt.min().item(), max_abs=mt.abs().max().item(),
                 runaway=bool(mt.abs().max().item() > 100 * ref),
                 neg_frac=float((mt.min(1).values < 0).float().mean().item()))
    return dict(drift=drift, rmse=rmse, rel_rmse=rmse / scale, diverged=div, plaus=plaus)


def robust(y, floor=1e-8, cap=1e3):
    y = np.nan_to_num(np.array(y, float), nan=cap, posinf=cap, neginf=cap)
    return np.clip(y, floor, cap)


def eval_split(trajs, models, T=400):
    out = {nm: dict(drift=[], rel=[], div=0) for nm in models}
    for tr in trajs:
        n, e, B, k, E0s, seed = (tr["n_nodes"], tr["edges"], tr["B"], tr["k"],
                                 tr["traj"][0], tr["seed"])
        for nm, m in models.items():
            r = rollout_eval(m, n, e, B, k, T=T, E0=E0s, seed=seed)
            out[nm]["drift"].append(r["drift"][-1])
            out[nm]["rel"].append(r["rel_rmse"][-1])
            out[nm]["div"] += int(r["diverged"])
    return out


# --------------------------------------------------------------------------
# (C) Integrator robustness
# --------------------------------------------------------------------------
def rollout_integrator(model, n, edges, B, k, T=400, dt=0.05, E0=None, seed=99, integ="euler"):
    i_idx, j_idx = edge_index(edges)
    E0 = torch.rand(n) if E0 is None else E0.clone()
    Em = E0.clone()
    tot = [Em.sum().item()]
    exp = [Em.sum().item()]
    cS = 0.0

    def fm(E, S):
        return model(E, i_idx, j_idx, B, S, k)[0]

    with torch.no_grad():
        for s in range(T):
            S = source_term(s * dt, n, seed=seed)
            if integ == "euler":
                Em = Em + dt * fm(Em, S)
                q = dt * S.sum().item()
            elif integ == "backward":
                Sn = source_term((s + 1) * dt, n, seed=seed)
                g = Em.clone()
                for _ in range(8):
                    g = Em + dt * fm(g, Sn)
                Em = g
                q = dt * Sn.sum().item()
            elif integ == "rk4":
                S2 = source_term((s + 0.5) * dt, n, seed=seed)
                S4 = source_term((s + 1) * dt, n, seed=seed)
                k1 = fm(Em, S)
                k2 = fm(Em + 0.5 * dt * k1, S2)
                k3 = fm(Em + 0.5 * dt * k2, S2)
                k4 = fm(Em + dt * k3, S4)
                Em = Em + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                q = (dt / 6 * (S.sum() + 2 * S2.sum() + 2 * S2.sum() + S4.sum())).item()
            if not torch.isfinite(Em).all():
                Em = torch.nan_to_num(Em, nan=0., posinf=1e6, neginf=-1e6)
            tot.append(Em.sum().item())
            cS += q
            exp.append(exp[0] + cS)
    return abs(np.array(tot)[-1] - np.array(exp)[-1])


# --------------------------------------------------------------------------
# (D) Rollout against nonlinear ground truth
# --------------------------------------------------------------------------
def rollout_nl(model, n, edges, B, k, T=500, dt=0.05, E0=None, seed=99):
    i_idx, j_idx = edge_index(edges)
    E0 = torch.rand(n) if E0 is None else E0.clone()
    Et = E0.clone()
    Em = E0.clone()
    tt = [Et.clone()]
    mt = [Em.clone()]
    tot = [Em.sum().item()]
    exp = [Em.sum().item()]
    cS = 0.
    with torch.no_grad():
        for s in range(T):
            S = source_term(s * dt, n, seed=seed)
            Et = Et + dt * (B @ flux_nonlinear(Et, i_idx, j_idx, k) + S)
            tt.append(Et.clone())
            d, _ = model(Em, i_idx, j_idx, B, S, k)
            Em = Em + dt * d
            if not torch.isfinite(Em).all():
                Em = torch.nan_to_num(Em, nan=0., posinf=1e6, neginf=-1e6)
            mt.append(Em.clone())
            tot.append(Em.sum().item())
            cS += dt * S.sum().item()
            exp.append(exp[0] + cS)
    tt = torch.stack(tt)
    mt = torch.stack(mt)
    rmse = torch.sqrt(((tt - mt) ** 2).mean(1)).numpy()
    sc = torch.sqrt((tt ** 2).mean(1)).clamp_min(1e-8).numpy()
    return abs(np.array(tot)[-1] - np.array(exp)[-1]), (rmse / sc)[-1]

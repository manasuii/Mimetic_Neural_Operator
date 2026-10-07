"""Fast tests — the structural guarantees, not the training results.

    pytest -q
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

from mimetic_no.config import UNIT_TEST, UNIT_TEST_TOL
from mimetic_no.data import make_trajectories
from mimetic_no.dynamics import conservation_unit_test, simulate_true, source_term
from mimetic_no.evaluation import rollout_eval, rollout_integrator
from mimetic_no.graph import build_graph, edge_index, incidence_matrix
from mimetic_no.models import BlackBoxGNN, MimeticOperator, SoftMimeticOperator
from mimetic_no.topologies import (build_delaunay_graph, build_geometric_graph,
                                   build_grid_graph, build_tree_graph)
from mimetic_no.training import train_model


# --------------------------------------------------------------------------
# Stage 1 — the incidence operator is what makes the scheme mimetic
# --------------------------------------------------------------------------
def test_graph_is_connected_and_edges_sorted():
    n, edges, k = build_graph(seed=1)
    assert n == 15
    assert nx.is_connected(nx.Graph(edges))
    assert all(i < j for i, j in edges)
    assert k.shape[0] == len(edges)


def test_incidence_columns_sum_to_zero():
    n, edges, _ = build_graph(seed=1)
    B = incidence_matrix(n, edges)
    assert B.shape == (n, len(edges))
    assert B.sum(0).abs().max().item() == 0.0


def test_edge_index_matches_edges():
    _, edges, _ = build_graph(seed=2)
    i_idx, j_idx = edge_index(edges)
    assert [(int(a), int(b)) for a, b in zip(i_idx, j_idx)] == list(edges)


# --------------------------------------------------------------------------
# Stage 2 — the headline guarantee
# --------------------------------------------------------------------------
def test_closed_system_conserves_to_machine_precision():
    res64 = conservation_unit_test(**UNIT_TEST)
    assert res64.max() < UNIT_TEST_TOL


def test_simulate_true_shape():
    n, edges, k = build_graph(seed=1)
    B = incidence_matrix(n, edges)
    traj = simulate_true(n, edges, B, k, T=20, seed=0)
    assert traj.shape == (21, n)
    assert torch.isfinite(traj).all()


def test_source_term_is_deterministic_per_seed():
    a = source_term(0.5, 15, seed=7)
    b = source_term(0.5, 15, seed=7)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# Stage 3 — model interface contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [MimeticOperator, SoftMimeticOperator, BlackBoxGNN])
def test_model_forward_contract(cls):
    n, edges, k = build_graph(seed=1)
    B = incidence_matrix(n, edges)
    i_idx, j_idx = edge_index(edges)
    E = torch.rand(n)
    S = source_term(0.0, n, seed=0)
    dEdt, imb = cls()(E, i_idx, j_idx, B, S, k)
    assert dEdt.shape == (n,)
    assert imb.dim() == 0


def test_mimetic_imbalance_is_exactly_zero():
    n, edges, k = build_graph(seed=1)
    B = incidence_matrix(n, edges)
    i_idx, j_idx = edge_index(edges)
    E = torch.rand(n)
    S = source_term(0.0, n, seed=0)
    dEdt, imb = MimeticOperator()(E, i_idx, j_idx, B, S, k)
    assert imb.item() == 0.0
    # the learned flux cannot create or destroy energy: sum(dE - S) == 0
    assert (dEdt - S).sum().abs().item() < 1e-4


def test_training_runs_one_epoch():
    trajs = make_trajectories(n_traj=1, n_nodes=8, T=12, seed_base=0)
    model = MimeticOperator()
    before = [p.clone() for p in model.parameters()]
    train_model(model, trajs, epochs=1, K=2, stride=4, log_every=1)
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


# --------------------------------------------------------------------------
# Stage 5 — evaluation plumbing
# --------------------------------------------------------------------------
def test_rollout_eval_keys_and_conservation():
    n, edges, k = build_graph(n_nodes=10, seed=5)
    B = incidence_matrix(n, edges)
    r = rollout_eval(MimeticOperator(), n, edges, B, k, T=30, E0=torch.rand(n))
    assert set(r) == {"drift", "rmse", "rel_rmse", "diverged", "plaus"}
    assert not r["diverged"]
    # hard structure: drift stays at the float32 round-off floor
    assert r["drift"][-1] < 1e-3


@pytest.mark.parametrize("integ", ["euler", "backward", "rk4"])
def test_conservation_is_integrator_independent(integ):
    n, edges, k = build_graph(n_nodes=10, seed=5)
    B = incidence_matrix(n, edges)
    drift = rollout_integrator(MimeticOperator(), n, edges, B, k, T=30,
                               E0=torch.rand(n), integ=integ)
    assert drift < 1e-3


# --------------------------------------------------------------------------
# Stage 6 — structured topologies keep the guarantee
# --------------------------------------------------------------------------
@pytest.mark.parametrize("gen", [
    lambda s: build_tree_graph(20, seed=s),
    lambda s: build_delaunay_graph(20, seed=s),
    lambda s: build_grid_graph(4, 5, seed=s),
    lambda s: build_geometric_graph(20, seed=s, radius=0.45),
])
def test_structured_topologies_are_connected_and_mimetic(gen):
    n, edges, k = gen(0)
    B = incidence_matrix(n, edges)
    assert n > 1 and len(edges) > 0
    assert nx.is_connected(nx.Graph(edges))
    assert B.sum(0).abs().max().item() == 0.0
    assert len(k) == len(edges)


def test_make_trajectories_fields():
    trajs = make_trajectories(n_traj=2, n_nodes=8, T=10, seed_base=0)
    assert len(trajs) == 2
    for tr in trajs:
        assert set(tr) >= {"i_idx", "j_idx", "B", "k", "traj", "seed", "dt", "n_nodes", "T", "edges"}
        assert tr["traj"].shape == (tr["T"] + 1, tr["n_nodes"])
        assert np.isfinite(tr["traj"].numpy()).all()

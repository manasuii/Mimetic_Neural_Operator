# Structure-Preserving (Mimetic) Neural Operator

Code, training and results for the **mimetic-vs-black-box** study on district-heating
style networks. A hand-built mimetic discretization conserves energy to machine
precision for *any* flux law, because the signed incidence operator `B` has
column sums of exactly zero. This repo asks whether a *learned* flux operator
wrapped in that same structure keeps the guarantee — and compares it against a
soft-penalty ablation and an unstructured black-box GNN.


---

## 1. Quick start

```bash
git clone <your-repo-url> mimetic-neural-operator
cd mimetic-neural-operator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                   # optional, enables `mimetic-no` and `python -m mimetic_no.cli`

python scripts/run_all.py          # whole pipeline
python scripts/run_all.py --fast   # ~1 min smoke test
```

Figures land in `results/figures/`, data in `data/`, weights in `checkpoints/`.

---

## 2. Repository layout

```
mimetic-neural-operator/
├── src/mimetic_no/                  library code (functions/classes live here)
│   ├── __init__.py
│   ├── config.py                    paths, seeds, dataset specs
│   ├── graph.py                     Stage 1: graphs, incidence matrix, edge index
│   ├── dynamics.py                  Stage 1-2: true flux, sources, simulators, unit test
│   ├── data.py                      Stage 4: trajectories, OOD split, save/load
│   ├── models.py                    Stage 3: Mimetic / Soft / Black-box
│   ├── training.py                  Stage 3: K-step trainer, checkpoints
│   ├── evaluation.py                Stage 5: rollouts, drift, RMSE, integrators
│   ├── topologies.py                Stage 6: tree/Delaunay/grid/geometric/pandapipes
│   ├── plotting.py                  every figure, one function each
│   └── cli.py                       `python -m mimetic_no.cli <stage>`
├── scripts/                         one runnable file per stage
│   ├── _bootstrap.py                puts src/ on sys.path (no install needed)
│   ├── stage1_graph_and_dynamics.py
│   ├── stage2_conservation_test.py
│   ├── stage4_generate_data.py
│   ├── stage3_train_models.py
│   ├── stage5_evaluate.py
│   ├── stage5plus_robustness.py     experiments A-E, `--only C E` to pick
│   ├── stage6_topologies.py
│   └── run_all.py
├── tests/test_mimetic.py
├── data/  checkpoints/  results/figures/     (generated, git-ignored)
├── .vscode/settings.json, launch.json, extensions.json
├── pyproject.toml, requirements.txt, requirements-dev.txt
├── .gitignore, LICENSE, README.md
```

**Why `src/` + `scripts/`.** Library code is importable and testable; scripts are
thin drivers that wire a stage together and print its tables. Nothing in `src/`
prints a result table or saves a figure on import, so you can import any piece
into a notebook without side effects.

---

## 3. Run order (important)

Stage numbers come from the notebook's narrative, not from execution order.
**Stage 4 runs before Stage 3**, because training consumes the generated data.

| # | Script | Needs | Produces |
|---|--------|-------|----------|
| 1 | `stage1_graph_and_dynamics.py` | — | incidence summary line |
| 2 | `stage2_conservation_test.py` | — | `stage2_unit_test.png`, asserts residual < 1e-9 |
| 4 | `stage4_generate_data.py` | — | `data/trajectories.pt`, `data/dataset.npz` |
| 3 | `stage3_train_models.py` | stage 4 | `checkpoints/{mimetic,soft,blackbox}.pt` |
| 5 | `stage5_evaluate.py` | 4, 3 | `claim1_conservation.png`, `claim2_accuracy.png`, OOD tables |
| 5+ | `stage5plus_robustness.py` | 4, 3 | `expA_lambda.png`, `expB_K.png`, `expE_scale.png` |
| 6 | `stage6_topologies.py` | 4, 3 | `stage6_topology_gallery.png`, `stage6_topology_drift.png` |

```bash
python scripts/stage1_graph_and_dynamics.py
python scripts/stage2_conservation_test.py
python scripts/stage4_generate_data.py
python scripts/stage3_train_models.py
python scripts/stage5_evaluate.py
python scripts/stage5plus_robustness.py --only C E
python scripts/stage6_topologies.py
```

Equivalent via the installed CLI:

```bash
mimetic-no stage4
mimetic-no stage3
mimetic-no stage5plus --only C E
mimetic-no all --fast
```

Add `--show` to any plotting stage to open figures interactively; without it the
scripts switch to the Agg backend and only write PNGs, so they run headless.

---

## 4. What each stage does

**Stage 1 — graph and incidence operator.** Random connected Erdős–Rényi graph with
per-edge conductance `k_ij ~ U(0.5, 1.5)`. The signed incidence matrix `B` has one
`-1` and one `+1` per column, so `sum(B @ F) = 0` for *any* flux `F` — the discrete
divergence theorem, and the whole basis of the conservation claim.

**Stage 2 — conservation unit test.** Closed system (`S = 0`) in float64: total energy
must be constant. The script asserts the residual stays below `1e-9`; if the
discretization were not mimetic, this fails loudly.

**Stage 4 — data.** 12 training trajectories, 6 in-distribution test, and a held-out
OOD split with a **larger** topology (22 nodes) **and** shifted conductances
`k ∈ [1.5, 2.5]`.

**Stage 3 — models.** All three share the interface
`forward(E, i_idx, j_idx, B, S, k) -> (dEdt, internal_imbalance)`:

- `MimeticOperator` — hard structure, `dE = B @ F_θ + S`; imbalance is exactly 0.
- `SoftMimeticOperator` — same flux net plus a learned node leak `g`, penalised with
  weight `λ`. A fair ablation: identical capacity, soft instead of hard constraint.
- `BlackBoxGNN` — unstructured message passing, also given `k`.

Trained with a **K-step rollout loss** (exposure-bias fix) rather than one-step.

**Stage 5 — evaluation.** Conservation drift vs rollout length (Claim 1), relative
RMSE (Claim 2, divergence-safe), long-horizon physical plausibility at ~7.5× the
training horizon, and the in-distribution/OOD split tables.

**Stage 5+ — robustness (A–E).**
(A) λ sweep: the soft penalty never reaches hard-structure conservation at any λ.
(B) K sweep: accuracy vs rollout-training horizon.
(C) integrators: Euler / backward Euler / RK4 — conservation is a property of the
operator, not the stepper.
(D) nonlinear flux law `F = k(E_i - E_j)(1 + 0.5·avg)`.
(E) scale: trained on 15 nodes, evaluated on 30/60/100.

**Stage 6 — realistic topologies.** Real district-heating networks are radial trees
and planar street loops, not Erdős–Rényi. Conductances are held in the **training**
range so any degradation is attributable to topology alone. Optionally loads a real
municipal network via `pandapipes`; falls back to a spatial graph when it is absent,
so the stage always runs.

---

## 5. VS Code setup

1. Open the **folder** `mimetic-neural-operator/` (not a parent directory) — the
   settings are workspace-relative.
2. Install the recommended extensions when prompted (`.vscode/extensions.json`):
   Python, Pylance, debugpy, Black, Ruff.
3. `Ctrl/Cmd+Shift+P` → **Python: Select Interpreter** → pick `.venv`.
4. Press **F5** and choose a stage from the dropdown — every stage, plus
   "Stage 5+ — only C E" and "Run all stages (--fast)", is a ready launch config.
5. The Testing sidebar discovers `tests/` automatically (pytest is pre-configured).

`settings.json` puts `src/` on `PYTHONPATH` for the integrated terminal and on
Pylance's `extraPaths`, so imports resolve with or without `pip install -e .`.

---

## 6. Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the structural guarantees rather than training quality: incidence
columns sum to zero, the closed system conserves to machine precision, the mimetic
model's imbalance is exactly zero, drift stays at the round-off floor under all
three integrators, and every Stage-6 topology family is connected and mimetic.
It runs in well under a minute on small graphs.

---

## 7. Reproducibility

`config.set_seeds()` reproduces the notebook's `torch.manual_seed(0)` /
`np.random.seed(0)`, and every graph, trajectory and evaluation uses an explicit
seed. All tunables (dataset specs, epochs, `K`, λ grid, rollout lengths, topology
instance counts) live in `src/mimetic_no/config.py` — change them there rather
than editing the stage scripts.

Exact figures from a fresh run:

```
results/figures/stage2_unit_test.png
results/figures/claim1_conservation.png
results/figures/claim2_accuracy.png
results/figures/expA_lambda.png
results/figures/expB_K.png
results/figures/expE_scale.png
results/figures/stage6_topology_gallery.png
results/figures/stage6_topology_drift.png
```

---

## 8. Optional: real utility topology

```bash
pip install pandapipes
python scripts/stage6_topologies.py
```

Without it, `build_pandapipes_graph` prints a notice and uses a NetworkX spatial
fallback — the stage never breaks on a missing optional dependency.

---

## License

MIT — see [LICENSE](LICENSE).

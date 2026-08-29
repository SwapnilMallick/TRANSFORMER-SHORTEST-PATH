# VetoFormer

**Online cost-predicting transformers with a self-veto for shortest-path planning in continuous mazes.**

*(repo: `transformer_shortest_path`)*

Online transformer training for **cost-to-come estimation** in continuous 2-D
maze environments, with sampling-based planners (RRT, Go-Explore RRT) as
baselines.

A small causal transformer is trained *online*, interleaved with its own
data collection: it rolls out trajectories through a maze, a roadmap graph is
built over the visited states, Dijkstra supplies cost-to-come labels, and the
transformer is retrained on the growing buffer. During collection the
transformer's own cost/uncertainty predictions drive a **veto**: an action that
revisits a known state is rejected unless the current path is (near-)competitive
with the best cost-to-come already recorded for that state.

(S. Mallick, UBC #68686534, 2026).

## Problem setup

| | |
|---|---|
| state | `s = [x, y] ∈ R²` (position in the unit square) |
| action | `a = [a_x, a_y] ∈ R²` (intended displacement, `‖a‖ ≤ step_size`) |
| dynamics | `s' = s + a + ε`, `ε ~ N(0, σ²I)` |
| reward | `1` if `‖s − goal‖ ≤ goal_radius`, else `0` |

### Environments

Selected with `--env`:

- **`four_room`** — 2×2 grid of rooms joined by single doorways; start
  bottom-left, goal top-right.
- **`eight_room`** — 4×2 grid, same start/goal corners.
- **`decoy`** — a "bait chamber": a tunnel points straight at the goal but
  dead-ends just short of it; the real entrance is hidden around the top, so
  goal-greedy motion is lured into the dead end.

## The model

> **`v3/online_cost_transformer.py` is the current version.** The online
> training algorithm and the transformer described below as "v3" are what is
> actively in use; `v1` and `v2` are retained for reference and for the baseline
> comparisons.

A causal transformer over fused state/action tokens, shared across all versions:

- fused 4-d token `[sₓ, s_y, aₓ, a_y]` → `Linear(4, 64)`
- learned (additive) positional embeddings
- 2 × `TransformerEncoderLayer`, `d_model=64`, `nhead=2`, GELU, causal mask

**Cost / uncertainty head — this is what changed in v3:**

- **v1 / v2** — two linear heads, cost `ĉ` and aleatoric uncertainty `μ`
  (softplus, `> 0`), trained jointly with a Huberised Gaussian NLL
  `L = 0.5·log μ + Huber(c − ĉ) / μ`.
- **v3 (current)** — a single cost head `ĉ` trained with plain Huber loss.
  Uncertainty is *epistemic only* and computed at veto time by **MC dropout**:
  with `dropout = 0.1` left on, `mc_samples = 25` stochastic passes are run over
  the candidate sequence; the last-token mean is `ĉ` and the last-token standard
  deviation `std_raw` (kept in cost units, not variance) is the raw uncertainty.
  `std_raw` is noisy at small sample counts, so it is blended into a per-node
  running estimate by an EMA keyed on the destination node id,
  `μ_ema[node] ← β·μ_ema[node] + (1 − β)·std_raw` with `β = 0.6` — often-visited
  nodes get a stable estimate, rarely-visited ones track their few observations.

## The online loop (Algorithm 1)

```
D ← {}
for n iterations:
    collect k trajectories with the current transformer
        - novelty-radius merging of proposed next states
        - an action reaching an already-visited state is VETOED unless
              c_path  <  ĉ_{s'} + (μ_{s'} − α)
          (fixed number of retries per rejected step)
    D ← D ∪ (new transitions)
    build a roadmap graph over merged states; Dijkstra from start → cost-to-come
        (these Dijkstra costs are the training labels)
    train the transformer on the labelled buffer
```

In **v3** the veto threshold uses the MC-dropout EMA estimate,
`cand_cost < ĉ + (μ_ema[s'] − α)`, with a fixed `α` (default `0.05`).

**v2 only:** the first `--explore-iters` iterations run **veto-free** (a warm
start), so any collision-free step is accepted and the roadmap can cover the
maze before the veto engages. v3 does not have this flag.

## Layout

| Path | Contents |
|---|---|
| `v1/online_cost_transformer.py` | First version online loop; two-head (cost + uncertainty) transformer, `step_size = 0.15`. |
| `v1/rrt_baseline.py` | Vanilla RRT (`goal_bias = 0`), Dijkstra-on-tree path cost. Imports the maze definitions directly from the transformer script so the environments are identical. |
| `v1/go_explore_rrt.py` | Go-Explore-style count-based RRT (seed chain + least-visited-node replay/expansion). Same shared environments and Dijkstra path extractor. |
| `v1/*_vs_transformer_cost_comparison.txt` | Path-cost comparisons: RRT and Go-Explore RRT vs. the transformer's greedy rollout across an α sweep. |
| `v2/online_cost_transformer.py` | Adds the veto-free exploration warm start (`--explore-iters`) and edge re-checking on node merges; `step_size = 0.08`. |
| `v3/online_cost_transformer.py` | **Current version.** Single cost head + plain Huber loss; epistemic uncertainty via MC dropout with a per-node EMA; `--alpha-sweep` runs the loop once per veto discount; `step_size = 0.15`. |
| `v3/rrt_baseline.py`, `v3/go_explore_rrt.py` | Copies of the baselines from `v1`, kept alongside the current transformer. |

## Setup

Python 3.11. Dependencies: `torch`, `numpy`, `matplotlib`, and `pillow` (for the
v3 GIF output).

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib pillow
```

## Running

Online transformer (use `v3` — the current version):

```bash
cd v3
python online_cost_transformer.py --env four_room
python online_cost_transformer.py --env decoy --iters 15 --k 8 --alpha 0.03 --out runs/decoy
python online_cost_transformer.py --env four_room \
    --alpha-sweep 0.0,0.01,0.02,0.03,0.04,0.05 --out runs/sweep
```

Key flags: `--env {four_room,eight_room,decoy}`, `--iters`, `--k`
(trajectories per iteration), `--seed`, `--max-steps`, `--alpha` (veto
uncertainty discount; lower = looser veto), `--alpha-sweep` (v3; run the loop
once per comma-separated `α`, each into `<out>/alpha_<value>/`), `--out`.
`--explore-iters` (veto-free warm start) exists in **v2 only**.

Each v3 run writes `{env}_run.log`, `{env}_metrics.csv`, `{env}_paths.json`,
`{env}_mu_ema.csv` (per-veto `std_raw` and `μ_ema` by node), and a plot of the
learned cost-to-come field with the roadmap-Dijkstra and transformer-greedy
paths overlaid.

Baselines:

```bash
cd v1
python rrt_baseline.py                    # all 3 envs, 25 seeds, step 0.15
python go_explore_rrt.py --env decoy --seeds 25 --eps 0.15 --out ge_results
```

## Metrics

Per iteration: roadmap node count, transitions buffered, trajectory success
rate, mean trajectory length, veto count, training loss, and **field MAE**
(mean `|ĉ − g|` of the transformer's cost prediction against the Dijkstra
label). The final summary extracts the roadmap-Dijkstra shortest path and the
transformer's predicted path cost, reporting their cost
ratio (`1.0` = transforemer prediction matches the roadmap-Dijkstra optimum).

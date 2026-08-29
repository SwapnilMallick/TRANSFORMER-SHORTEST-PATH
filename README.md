# transformer_shortest_path

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
| action | `a = [aₓ, a_y] ∈ R²` (intended displacement, `‖a‖ ≤ step_size`) |
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

`T_cost(ĉ, μ | h_t, a_t)` — a causal transformer over fused state/action tokens:

- fused 4-d token `[sₓ, s_y, aₓ, a_y]` → `Linear(4, 64)`
- learned (additive) positional embeddings
- 2 × `TransformerEncoderLayer`, `d_model=64`, `nhead=2`, GELU, causal mask
- two linear heads: cost `ĉ` and uncertainty `μ` (softplus, `> 0`)
- Huberised Gaussian NLL: `L = 0.5·log μ + Huber(c − ĉ) / μ`

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

The first `--explore-iters` iterations run **veto-free** (a warm start): any
collision-free step is accepted so the roadmap can cover the maze before the
veto engages.

## Layout

| Path | Contents |
|---|---|
| `v1/online_cost_transformer.py` | Original online loop; two-head (cost + uncertainty) transformer, `step_size = 0.15`. |
| `v1/rrt_baseline.py` | Vanilla RRT (`goal_bias = 0`), Dijkstra-on-tree path cost. Imports the maze definitions directly from the transformer script so the environments are identical. |
| `v1/go_explore_rrt.py` | Go-Explore-style count-based RRT (seed chain + least-visited-node replay/expansion). Same shared environments and Dijkstra path extractor. |
| `v1/*_vs_transformer_cost_comparison.txt` | Path-cost comparisons: RRT and Go-Explore RRT vs. the transformer's greedy rollout across an α sweep. |
| `v2/online_cost_transformer.py` | Adds the veto-free exploration warm start (`--explore-iters`) and edge re-checking on node merges; `step_size = 0.08`. |
| `v3/` | Local-only (git-ignored) analysis and plotting scripts — Dijkstra-vs-transformer cost tables, field-MAE / loss-by-α plots, node-visitation summaries. |

## Setup

Python 3.11. Dependencies: `torch`, `numpy`, `matplotlib`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
```

## Running

Online transformer:

```bash
cd v1   # or v2
python online_cost_transformer.py --env four_room
python online_cost_transformer.py --env decoy --iters 15 --k 8 --alpha 0.03 --explore-iters 3 --out runs/decoy
```

Key flags: `--env {four_room,eight_room,decoy}`, `--iters`, `--k`
(trajectories per iteration), `--seed`, `--max-steps`, `--alpha` (veto
uncertainty discount; lower = looser veto), `--explore-iters` (v2), `--out`.

Each run writes `{env}_run.log`, `{env}_metrics.csv`, `{env}_paths.json`, and
`{env}_cost.png` (roadmap cost-to-come field plus the roadmap-Dijkstra and
transformer-greedy paths) to the output directory.

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
transformer's greedy descent on the predicted cost field, reporting their cost
ratio (`1.0` = greedy matches the roadmap optimum).

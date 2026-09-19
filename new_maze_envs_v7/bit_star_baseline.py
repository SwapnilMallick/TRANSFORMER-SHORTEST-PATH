"""
BIT* (Batch Informed Trees) baseline over the shared mazes, for comparison
against Go-Explore RRT / vanilla RRT / the cost-to-come transformer.

This is an ORIGINAL implementation written directly from the algorithm
description in Gammell, Srinivasa & Barfoot, "Batch Informed Trees (BIT*):
Sampling-based Optimal Planning via the Heuristically Guided Search of
Implicit Random Geometric Graphs" (arXiv:1405.5848) -- it is not ported from
any existing codebase. The reference implementation at
github.com/marleyshan21/Batch-informed-trees was consulted only to see the
algorithm's typical parameter/API shape; that repo ships with no LICENSE
file, so no code from it is used here. This version is written from scratch
against this project's own continuous wall-segment maze representation
(rasterized-occupancy-grid handling in the reference repo has no equivalent
here -- collision checks go straight through online_cost_transformer.collides).

Two simplifications relative to the full paper, both justified by the same
fact: this baseline only needs a "first solution found" measurement, matching
how the Go-Explore/vanilla-RRT baselines are scored (they also stop the
instant the goal is reached) -- not a converged/refined plan:
  * a FIXED connection radius `rbit` (no shrinking-radius schedule)
  * uniform batch sampling only (no informed/ellipsoidal sampling once a
    solution exists -- moot anyway, since the run stops at the first one)
Rewiring (updating a vertex's parent whenever a cheaper route through the
current batch is found) IS implemented, so this remains a genuine BIT*, not
a degraded batch-RRT: the vertex queue (QV, ordered by g_t + h_hat) and edge
queue (QE, ordered by g_t + c_hat + h_hat) are both real, and edges pulled
off QE are lazily invalidated (skipped without a collision check) once they
can no longer improve on a node's current best cost -- the standard BIT*
technique for avoiding decrease-key operations on a plain binary heap.

The environment (walls, start, goal, collision test) is imported directly
from online_cost_transformer.py, so the mazes are IDENTICAL to the
transformer / RRT / Go-Explore-RRT experiments. Path extraction reuses
rrt_baseline.dijkstra_shortest_path, so the path-length metric is computed
identically across all baselines -- BIT*'s own tree is exported as
(positions, parent) exactly like the RRT/Go-Explore trees, restricted to the
connected portion (BIT*'s batch of not-yet-connected samples is dropped
before export, since dijkstra_shortest_path expects every non-root node it's
given to already have a valid parent).

"Nodes required to find the path" (per the task) = the number of TREE nodes
(connected samples) at the moment BIT* first connects to the goal region --
the same definition of "tree_nodes"/"n" the RRT and Go-Explore-RRT baselines
already report. The (usually larger) total number of samples drawn, including
ones BIT* never ended up connecting, is also recorded and reported alongside
it, since it's a meaningful extra stat unique to BIT*'s batch mechanic.

rbit default: fixed at 0.15 to match the step size (STEP/EPS) the RRT and
Go-Explore-RRT baselines use for every hop. rbit only caps BIT*'s MAXIMUM
edge length -- most accepted edges will be shorter, exactly like the shortest
paths the other baselines extract are made of many sub-max-length hops -- but
capping it at the same value the other two are held to is what keeps this a
fair comparison: a larger rbit would let BIT* bridge a doorway or cut across
a corridor in one hop where RRT/Go-Explore-RRT are forced to take several,
which would advantage BIT* on exactly the mazes designed to punish that.

Usage:
    python bit_star_baseline.py                          # all envs, 25 seeds
    python bit_star_baseline.py --env spiral --seeds 25 --rbit 0.15
"""

import os
import math
import heapq
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --- reuse the EXACT environment definitions + path extractor already used
#     by the other baselines, so all three are directly comparable ---
from online_cost_transformer import (
    Config, ENVS, build_walls, apply_env, collides, env_label,
)
from rrt_baseline import dijkstra_shortest_path

SEEDS_DEFAULT = 25
RBIT_DEFAULT = 0.15          # matches STEP_DEFAULT/EPS_DEFAULT (0.15) in the
                             # RRT / Go-Explore-RRT baselines -- see module
                             # docstring for why this has to match for a fair
                             # comparison
BATCH_DEFAULT = 100
MAX_SAMPLES_DEFAULT = 30000


# --------------------------------------------------------------------------- #
# BIT*: batch sampling + heuristic vertex/edge queues + rewiring.
# --------------------------------------------------------------------------- #
def run_bit_star(cfg, walls, seed, rbit=RBIT_DEFAULT, batch_size=BATCH_DEFAULT,
                 max_samples=MAX_SAMPLES_DEFAULT):
    """Returns (P, parent, reached, goal_node, n_tree_nodes, n_samples_drawn).

    P/parent describe only the CONNECTED portion of the tree (see module
    docstring) -- directly compatible with rrt_baseline.dijkstra_shortest_path
    and the same plot_run style used by the RRT / Go-Explore-RRT baselines.
    """
    rng = np.random.default_rng(seed)
    start = np.array(cfg.start, float)
    goal = np.array(cfg.goal, float)
    L = cfg.L

    P = [start.copy()]              # all sampled points ever drawn (append-only)
    parent = {0: -1}
    g_t = {0: 0.0}                  # tree cost-to-come, defined for in_tree nodes
    in_tree = {0}
    unconnected = set()             # sampled but not yet part of the tree

    def h_hat(i):
        return float(np.linalg.norm(P[i] - goal))

    def add_batch(m):
        for _ in range(m):
            q = rng.random(2) * L
            idx = len(P)
            P.append(q)
            unconnected.add(idx)

    reached, goal_node = False, -1
    add_batch(batch_size)
    total_samples = batch_size
    P_arr = np.array(P)

    QV = [(h_hat(0), 0)]            # g_t[0] == 0
    QE = []

    while total_samples <= max_samples and not reached:
        while QV or QE:
            qv_top = QV[0][0] if QV else math.inf
            qe_top = QE[0][0] if QE else math.inf
            if QE and qe_top <= qv_top:
                _, (v, x) = heapq.heappop(QE)
                c_hat_vx = float(np.linalg.norm(P_arr[v] - P_arr[x]))
                if g_t[v] + c_hat_vx + h_hat(x) >= g_t.get(x, math.inf):
                    continue                          # stale/dominated -- skip
                if collides(P[v], P[x], walls):
                    continue                          # only actual work per edge
                new_g = g_t[v] + c_hat_vx              # exact cost == c_hat (straight edges)
                if new_g < g_t.get(x, math.inf):
                    parent[x] = v
                    g_t[x] = new_g
                    in_tree.add(x)
                    unconnected.discard(x)
                    heapq.heappush(QV, (new_g + h_hat(x), x))
                    if np.linalg.norm(P_arr[x] - goal) < cfg.goal_radius:
                        reached, goal_node = True, x
                        break
            elif QV:
                key, v = heapq.heappop(QV)
                if key > g_t[v] + h_hat(v) + 1e-9:
                    continue                          # stale: a fresher entry already used
                d = np.linalg.norm(P_arr - P_arr[v], axis=1)
                near = np.where(d <= rbit)[0]
                for x in near:
                    x = int(x)
                    if x == v:
                        continue
                    if g_t[v] + d[x] + h_hat(x) < g_t.get(x, math.inf):
                        heapq.heappush(QE, (g_t[v] + d[x] + h_hat(x), (v, x)))
            else:
                break
        if reached or total_samples >= max_samples:
            break
        add_batch(batch_size)
        total_samples += batch_size
        P_arr = np.array(P)
        for v in in_tree:
            heapq.heappush(QV, (g_t[v] + h_hat(v), v))

    # export ONLY the connected tree, re-indexed with 0 = start, so this is
    # drop-in compatible with rrt_baseline.dijkstra_shortest_path / plot_run
    tree_nodes = sorted(in_tree)
    remap = {old: i for i, old in enumerate(tree_nodes)}
    P_export = np.array([P[old] for old in tree_nodes])
    parent_export = [-1] * len(tree_nodes)
    for old in tree_nodes:
        if old != 0:
            parent_export[remap[old]] = remap[parent[old]]
    goal_node_export = remap.get(goal_node, -1)

    return (P_export, parent_export, reached, goal_node_export,
            len(tree_nodes), total_samples)


# --------------------------------------------------------------------------- #
# Per-run figure: BIT* tree + shortest path.
# --------------------------------------------------------------------------- #
def plot_run(P, parent, path_pts, cfg, walls, env, seed, length, n_samples, out):
    fig, ax = plt.subplots(figsize=(6, 6))
    for (w0, w1) in walls:
        ax.plot([w0[0], w1[0]], [w0[1], w1[1]], color="k", lw=2, zorder=4)

    segs = [[P[i], P[parent[i]]] for i in range(1, len(P))]
    ax.add_collection(LineCollection(segs, colors="0.75", linewidths=0.5, zorder=2))
    ax.scatter(P[:, 0], P[:, 1], s=3, c="0.5", zorder=3)

    if path_pts is not None:
        ax.plot(path_pts[:, 0], path_pts[:, 1], color="crimson", lw=2.2,
                zorder=5, label=f"shortest path (len={length:.3f})")

    ax.scatter(*cfg.start, c="white", edgecolors="k", s=90, zorder=6, label="start")
    ax.scatter(*cfg.goal, marker="*", c="gold", edgecolors="k", s=240,
               zorder=6, label="goal")
    ax.add_patch(plt.Circle(cfg.goal, cfg.goal_radius, color="gold",
                            alpha=0.25, zorder=1))
    ax.set_xlim(-0.02, cfg.L + 0.02)
    ax.set_ylim(-0.02, cfg.L + 0.02)
    ax.set_aspect("equal")
    ax.set_title(f"{env} ({env_label(cfg)}) - BIT*  seed={seed}\n"
                 f"tree nodes={len(P)}  (samples drawn={n_samples})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main: sweep environments x seeds.
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="BIT* baseline over the shared mazes.")
    p.add_argument("--env", choices=list(ENVS), default=None,
                   help="single environment (default: all of "
                        + ", ".join(ENVS) + ")")
    p.add_argument("--seeds", type=int, default=SEEDS_DEFAULT,
                   help=f"number of seeds, 0..N-1 (default: {SEEDS_DEFAULT})")
    p.add_argument("--rbit", type=float, default=RBIT_DEFAULT,
                   help=f"fixed RGG connection radius (default: {RBIT_DEFAULT})")
    p.add_argument("--batch-size", type=int, default=BATCH_DEFAULT, dest="batch_size",
                   help=f"samples drawn per batch (default: {BATCH_DEFAULT})")
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT,
                   dest="max_samples",
                   help=f"sample budget per run before giving up "
                        f"(default: {MAX_SAMPLES_DEFAULT})")
    p.add_argument("--out", type=str, default="bit_star_results",
                   help="output directory (default: bit_star_results); each "
                        "env's results are written to its own subfolder here")
    return p.parse_args()


def main():
    args = parse_args()
    envs = [args.env] if args.env else list(ENVS)
    os.makedirs(args.out, exist_ok=True)

    overall = []
    for env in envs:
        cfg = apply_env(Config(env=env))
        walls = build_walls(cfg)
        env_dir = os.path.join(args.out, env)
        os.makedirs(env_dir, exist_ok=True)
        print(f"\n=== {env} ({env_label(cfg)})  start={cfg.start} goal={cfg.goal}"
              f"  rbit={args.rbit}  batch={args.batch_size}  (BIT*) ===")

        records = []          # (seed, ok, length, n_tree, n_samples, path_pts)
        lengths = []
        for seed in range(args.seeds):
            P, parent, reached, gnode, n_tree, n_samples = run_bit_star(
                cfg, walls, seed, rbit=args.rbit, batch_size=args.batch_size,
                max_samples=args.max_samples)
            path_pts, length = dijkstra_shortest_path(
                P, parent, cfg, walls, np.array(cfg.goal, float))
            ok = path_pts is not None and math.isfinite(length)
            img = os.path.join(env_dir, f"{env}_seed{seed:02d}.png")
            plot_run(P, parent, path_pts, cfg, walls, env, seed,
                     length if ok else float("nan"), n_samples, img)
            if ok:
                lengths.append(length)
            records.append((seed, ok, length if ok else float("nan"),
                            n_tree, n_samples, path_pts))
            print(f"  seed {seed:2d}: "
                  f"{'reached' if ok else 'FAILED ':7s}  "
                  f"len={length:7.3f}  tree_nodes={n_tree:6d}  samples={n_samples}")

        arr = np.array(lengths, float)
        mean = float(arr.mean()) if arr.size else float("nan")
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        print(f"  --> {env}: {mean:.3f} +/- {std:.3f}  "
              f"over {len(lengths)}/{args.seeds} seeds that reached the goal")
        overall.append((env, mean, std, len(lengths), args.seeds))

        txt = os.path.join(env_dir, f"{env}_bit_star_results.txt")
        with open(txt, "w") as f:
            f.write(f"# BIT*  env={env} ({env_label(cfg)})\n")
            f.write(f"# start={cfg.start}  goal={cfg.goal}  rbit={args.rbit}  "
                    f"batch_size={args.batch_size}  goal_radius={cfg.goal_radius}\n")
            f.write(f"# path length = Euclidean length of the start->goal path "
                    f"extracted by Dijkstra on the BIT* tree (connected portion "
                    f"only; same extractor as the RRT / Go-Explore-RRT baselines)\n")
            f.write(f"# tree_nodes = connected samples at first solution; "
                    f"samples = total samples drawn (including unconnected ones)\n")
            f.write(f"# seeds = {args.seeds}\n\n")
            f.write(f"AVERAGE SHORTEST PATH: {mean:.4f} +/- {std:.4f} "
                    f"(sample std) over {len(lengths)}/{args.seeds} successful "
                    f"seeds\n")
            f.write(f"lengths = {[round(x, 4) for x in lengths]}\n\n")
            for seed, ok, length, n_tree, n_samples, path_pts in records:
                f.write("-" * 70 + "\n")
                if not ok:
                    f.write(f"seed {seed}: FAILED to reach goal "
                            f"(tree_nodes={n_tree}, samples={n_samples})\n")
                    continue
                f.write(f"seed {seed}: length={length:.4f}  tree_nodes={n_tree}  "
                        f"samples={n_samples}  path_waypoints={len(path_pts)}\n")
                pts = ", ".join(f"({x:.3f},{y:.3f})" for x, y in path_pts)
                f.write(f"  path: {pts}\n")
        print(f"  [paths] saved -> {txt}")
        print(f"  [figs]  saved -> {env_dir}/{env}_seed*.png")

    summ = os.path.join(args.out, "summary.txt")
    with open(summ, "w") as f:
        f.write("BIT* shortest-path length, mean +/- sample-std across seeds\n")
        f.write(f"rbit={args.rbit}  batch_size={args.batch_size}\n\n")
        for env, mean, std, k, tot in overall:
            f.write(f"{env:12s}: {mean:.4f} +/- {std:.4f}  "
                    f"({k}/{tot} seeds reached)\n")
    print(f"\n[summary] saved -> {summ}")
    print("\n==== SUMMARY (mean +/- std of shortest-path length) ====")
    for env, mean, std, k, tot in overall:
        print(f"  {env:12s}: {mean:.3f} +/- {std:.3f}  ({k}/{tot} reached)")


if __name__ == "__main__":
    main()

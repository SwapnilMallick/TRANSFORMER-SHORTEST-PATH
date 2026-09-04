"""
Vanilla RRT baseline (goal_bias = 0) in the four_room / eight_room / decoy
mazes, for comparison against the cost-to-come transformer.

The environments (walls, start, goal, collision test) are imported directly
from online_cost_transformer.py, so the mazes are IDENTICAL to the ones the
transformer script uses -- nothing about the environment is redefined here.

For each environment and each of N seeds:
  1. Grow a vanilla RRT with uniform sampling (NO goal bias), step size STEP,
     until a node connects to the goal (within goal_radius, collision-free) or
     the node budget is exhausted.
  2. Run Dijkstra on the resulting tree (the exact goal is linked to every tree
     node within goal_radius) to extract the shortest start->goal path.
  3. Save a figure of that run: the full RRT tree + the shortest path.

Then report the mean +/- std of the shortest-path length across seeds, and dump
all per-seed paths to a text file.

Usage:
    python rrt_baseline.py                       # all 3 envs, 25 seeds, step 0.15
    python rrt_baseline.py --env decoy --seeds 25 --step 0.15 --out rrt_results
"""

import os
import math
import heapq
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --- reuse the EXACT environment definitions from the transformer script ---
from online_cost_transformer import (
    Config, ENVS, build_walls, apply_env, collides, env_label,
)

STEP_DEFAULT = 0.15          # matches the transformer script's step size
SEEDS_DEFAULT = 25


# --------------------------------------------------------------------------- #
# Vanilla RRT (goal_bias = 0): uniform sampling, single-step extension.
# --------------------------------------------------------------------------- #
def run_rrt(cfg, walls, step, seed, max_nodes=30000, max_samples=2_000_000):
    rng = np.random.default_rng(seed)
    start = np.array(cfg.start, float)
    goal = np.array(cfg.goal, float)

    P = np.empty((max_nodes, 2))          # preallocated node positions
    P[0] = start
    n = 1
    parent = [-1]                         # parent index per node
    reached = False
    goal_node = -1

    for _ in range(max_samples):
        if n >= max_nodes:
            break
        # goal_bias = 0  ->  always sample uniformly in the free box
        q = rng.random(2) * cfg.L
        diff = P[:n] - q
        j = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))   # nearest node
        qn = P[j]
        v = q - qn
        dist = math.hypot(v[0], v[1])
        if dist < 1e-12:
            continue
        qnew = qn + v * (min(step, dist) / dist)                # steer by <= step
        if not (0.0 <= qnew[0] <= cfg.L and 0.0 <= qnew[1] <= cfg.L):
            continue
        if collides(qn, qnew, walls):                           # edge must be free
            continue
        P[n] = qnew
        parent.append(j)
        ni = n
        n += 1
        if (math.hypot(qnew[0] - goal[0], qnew[1] - goal[1]) < cfg.goal_radius
                and not collides(qnew, goal, walls)):
            reached = True
            goal_node = ni
            break

    return P[:n].copy(), parent, reached, goal_node, n


# --------------------------------------------------------------------------- #
# Dijkstra shortest path on the generated tree (goal linked to nearby nodes).
# --------------------------------------------------------------------------- #
def dijkstra_shortest_path(P, parent, cfg, walls, goal):
    n = len(P)
    adj = defaultdict(list)
    for i in range(1, n):
        p = parent[i]
        w = math.hypot(P[i][0] - P[p][0], P[i][1] - P[p][1])
        adj[i].append((p, w))
        adj[p].append((i, w))

    gi = n                                   # virtual goal node
    connected = False
    for i in range(n):
        if (math.hypot(P[i][0] - goal[0], P[i][1] - goal[1]) < cfg.goal_radius
                and not collides(P[i], goal, walls)):
            w = math.hypot(P[i][0] - goal[0], P[i][1] - goal[1])
            adj[i].append((gi, w))
            adj[gi].append((i, w))
            connected = True
    if not connected:
        return None, math.inf

    dist = {0: 0.0}
    prev = {}
    pq = [(0.0, 0)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist.get(u, math.inf):
            continue
        if u == gi:
            break
        for v, w in adj[u]:
            nd = du + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if gi not in dist:
        return None, math.inf

    seq = [gi]
    while seq[-1] != 0:
        seq.append(prev[seq[-1]])
    seq.reverse()
    pts = np.array([goal if k == gi else P[k] for k in seq])
    return pts, dist[gi]


# --------------------------------------------------------------------------- #
# Per-run figure: RRT tree + shortest path.
# --------------------------------------------------------------------------- #
def plot_run(P, parent, path_pts, cfg, walls, env, seed, length, out):
    fig, ax = plt.subplots(figsize=(6, 6))
    for (w0, w1) in walls:
        ax.plot([w0[0], w1[0]], [w0[1], w1[1]], color="k", lw=2, zorder=4)

    segs = [[P[i], P[parent[i]]] for i in range(1, len(P))]
    ax.add_collection(LineCollection(segs, colors="0.75", linewidths=0.4, zorder=2))
    ax.scatter(P[:, 0], P[:, 1], s=2, c="0.55", zorder=3)

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
    ax.set_title(f"{env} ({env_label(cfg)}) - RRT (goal_bias=0)  seed={seed}\n"
                 f"tree nodes={len(P)}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main: sweep environments x seeds.
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Vanilla RRT (goal_bias=0) baseline over the shared mazes.")
    p.add_argument("--env", choices=list(ENVS), default=None,
                   help="single environment (default: all of "
                        + ", ".join(ENVS) + ")")
    p.add_argument("--seeds", type=int, default=SEEDS_DEFAULT,
                   help=f"number of seeds, 0..N-1 (default: {SEEDS_DEFAULT})")
    p.add_argument("--step", type=float, default=STEP_DEFAULT,
                   help=f"RRT step size (default: {STEP_DEFAULT})")
    p.add_argument("--max-nodes", type=int, default=30000, dest="max_nodes",
                   help="node budget per run before giving up (default: 30000)")
    p.add_argument("--out", type=str, default="rrt_results",
                   help="output directory (default: rrt_results)")
    return p.parse_args()


def main():
    args = parse_args()
    envs = [args.env] if args.env else list(ENVS)
    os.makedirs(args.out, exist_ok=True)

    overall = []
    for env in envs:
        cfg = apply_env(Config(env=env))
        walls = build_walls(cfg)
        print(f"\n=== {env} ({env_label(cfg)})  start={cfg.start} goal={cfg.goal}"
              f"  step={args.step}  goal_bias=0 ===")

        records = []          # (seed, reached, length, nodes, path_pts)
        lengths = []
        for seed in range(args.seeds):
            P, parent, reached, gnode, n = run_rrt(
                cfg, walls, args.step, seed, max_nodes=args.max_nodes)
            path_pts, length = dijkstra_shortest_path(
                P, parent, cfg, walls, np.array(cfg.goal, float))
            ok = path_pts is not None and math.isfinite(length)
            img = os.path.join(args.out, f"{env}_seed{seed:02d}.png")
            plot_run(P, parent, path_pts, cfg, walls, env, seed,
                     length if ok else float("nan"), img)
            if ok:
                lengths.append(length)
            records.append((seed, ok, length if ok else float("nan"), n, path_pts))
            print(f"  seed {seed:2d}: "
                  f"{'reached' if ok else 'FAILED ':7s}  "
                  f"len={length:7.3f}  nodes={n}")

        arr = np.array(lengths, float)
        mean = float(arr.mean()) if arr.size else float("nan")
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        print(f"  --> {env}: {mean:.3f} +/- {std:.3f}  "
              f"over {len(lengths)}/{args.seeds} seeds that reached the goal")
        overall.append((env, mean, std, len(lengths), args.seeds))

        # dump per-seed paths + summary to a text file
        txt = os.path.join(args.out, f"{env}_rrt_results.txt")
        with open(txt, "w") as f:
            f.write(f"# Vanilla RRT (goal_bias=0)  env={env} ({env_label(cfg)})\n")
            f.write(f"# start={cfg.start}  goal={cfg.goal}  step={args.step}  "
                    f"goal_radius={cfg.goal_radius}\n")
            f.write(f"# path length = Euclidean length of the start->goal path "
                    f"extracted by Dijkstra on the RRT tree\n")
            f.write(f"# seeds = {args.seeds}\n\n")
            f.write(f"AVERAGE SHORTEST PATH: {mean:.4f} +/- {std:.4f} "
                    f"(sample std) over {len(lengths)}/{args.seeds} successful "
                    f"seeds\n")
            f.write(f"lengths = {[round(x, 4) for x in lengths]}\n\n")
            for seed, ok, length, n, path_pts in records:
                f.write("-" * 70 + "\n")
                if not ok:
                    f.write(f"seed {seed}: FAILED to reach goal "
                            f"(tree nodes={n})\n")
                    continue
                f.write(f"seed {seed}: length={length:.4f}  tree_nodes={n}  "
                        f"path_waypoints={len(path_pts)}\n")
                pts = ", ".join(f"({x:.3f},{y:.3f})" for x, y in path_pts)
                f.write(f"  path: {pts}\n")
        print(f"  [paths] saved -> {txt}")
        print(f"  [figs]  saved -> {args.out}/{env}_seed*.png")

    # combined summary
    summ = os.path.join(args.out, "summary.txt")
    with open(summ, "w") as f:
        f.write("Vanilla RRT (goal_bias=0) shortest-path length, "
                "mean +/- sample-std across seeds\n")
        f.write(f"step_size={args.step}\n\n")
        for env, mean, std, k, tot in overall:
            f.write(f"{env:12s}: {mean:.4f} +/- {std:.4f}  "
                    f"({k}/{tot} seeds reached)\n")
    print(f"\n[summary] saved -> {summ}")
    print("\n==== SUMMARY (mean +/- std of shortest-path length) ====")
    for env, mean, std, k, tot in overall:
        print(f"  {env:12s}: {mean:.3f} +/- {std:.3f}  ({k}/{tot} reached)")


if __name__ == "__main__":
    main()

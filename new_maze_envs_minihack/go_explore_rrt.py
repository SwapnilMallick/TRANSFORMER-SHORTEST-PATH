"""
Go-Explore inspired RRT (count-based, per-node-counter exploration) over the
four_room / eight_room / decoy mazes, for comparison against the cost-to-come
transformer and the vanilla-RRT baseline.

Implements Algorithms 1-2 from "GO EXPLORE RRT":
  * Every node stores position, parent, the incoming action, a visitation
    counter c, and the set tau of headings already taken from it.
  * Phase 1 (SEED): a chain of up to k diverse (angularly-distinct) collision-
    free actions from the root.
  * Phase 2 (EXPLORE): for up to m iterations, pick the least-visited node
    v* (min counter, random tie-break), increment its counter, RESET the agent
    to the root and REPLAY the stored actions to reach v* (no teleport), then
    roll out up to j diverse collision-free actions from v*.
  * Terminate when a node lands within delta of the goal, else Failure.

The environment (walls, start, goal, collision test) is imported directly from
online_cost_transformer.py, so the mazes are IDENTICAL to the transformer /
vanilla-RRT experiments. epsilon (step size) defaults to 0.15.

After the tree is built, Dijkstra is run on it (reusing rrt_baseline's extractor,
so the path metric is identical to the vanilla-RRT baseline) to get the shortest
start->goal path. Across seeds we report mean +/- std of that length.

Usage:
    python go_explore_rrt.py                          # all 3 envs, 25 seeds, eps 0.15
    python go_explore_rrt.py --env decoy --seeds 25 --eps 0.15 --out ge_results
"""

import os
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# shared environment (identical mazes) + identical Dijkstra path extractor
from online_cost_transformer import (
    Config, ENVS, build_walls, apply_env, collides, env_label, domain_size, run_name,
)
from rrt_baseline import dijkstra_shortest_path

EPS_DEFAULT = 0.15
SEEDS_DEFAULT = 25
TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def angdist(a, b):
    d = abs(a - b) % TWO_PI
    return min(d, TWO_PI - d)


def diverse_action(tau_v, eps, rng, dtheta, ntry):
    """A random heading kept angularly distinct (>= dtheta) from those already
    taken at this node, re-drawing up to ntry times (Alg 2: DiverseAction)."""
    theta = rng.uniform(0.0, TWO_PI)
    for _ in range(ntry):
        if not tau_v or min(angdist(theta, ph) for ph in tau_v) >= dtheta:
            break
        theta = rng.uniform(0.0, TWO_PI)
    return theta, eps * np.array([math.cos(theta), math.sin(theta)])


def _in_bounds(q, W, H):
    return 0.0 <= q[0] <= W and 0.0 <= q[1] <= H


# --------------------------------------------------------------------------- #
# Go-Explore RRT tree construction (Algorithm 1)
# --------------------------------------------------------------------------- #
def go_explore_rrt(cfg, walls, eps, seed,
                   k=30, j=25, m=40000, dtheta=0.5, ntry=20, max_nodes=40000):
    rng = np.random.default_rng(seed)
    start = np.array(cfg.start, float)
    goal = np.array(cfg.goal, float)
    delta = cfg.goal_radius
    W, H = domain_size(cfg)
    clearance = cfg.clearance

    pos = [start.copy()]        # node positions
    parent = [-1]               # parent index (pi)
    counter = [1]               # visitation counter (c)
    tau = [[]]                  # headings taken from each node

    def add_node(q, par, theta):
        pos.append(q.copy())
        parent.append(par)
        counter.append(1)
        tau.append([])
        tau[par].append(theta)
        return len(pos) - 1

    reached = False
    goal_node = -1

    # ---- Phase 1 | SEED: chain of up to k diverse actions from the root ----
    cur = 0
    q = start.copy()
    for _ in range(k):
        theta, a = diverse_action(tau[cur], eps, rng, dtheta, ntry)
        q2 = q + a
        if _in_bounds(q2, W, H) and not collides(q, q2, walls, clearance):
            w = add_node(q2, cur, theta)
            cur, q = w, q2
            if np.linalg.norm(q2 - goal) <= delta:
                reached, goal_node = True, w
                break
        else:
            break                                    # hit a wall -> stop chain

    # ---- Phase 2 | EXPLORE: least-visited selection + reset-replay rollout --
    if not reached:
        for _ in range(m):
            if len(pos) >= max_nodes:
                break
            # SelectBinNode: min counter, random tie-break
            cmin = min(counter)
            bucket = [i for i, c in enumerate(counter) if c == cmin]
            vstar = int(bucket[rng.integers(len(bucket))])
            counter[vstar] += 1                      # selecting == visiting

            # ResetAndReplay: under deterministic f, replaying root->v* actions
            # reproduces pos[vstar] exactly, so q_v* = pos[vstar].
            cur = vstar
            q = pos[vstar].copy()

            # Rollout(v*, j): up to j diverse collision-free actions
            for _ in range(j):
                theta, a = diverse_action(tau[cur], eps, rng, dtheta, ntry)
                q2 = q + a
                if _in_bounds(q2, W, H) and not collides(q, q2, walls, clearance):
                    w = add_node(q2, cur, theta)
                    cur, q = w, q2
                    if np.linalg.norm(q2 - goal) <= delta:
                        reached, goal_node = True, w
                        break
                else:
                    break                            # hit a wall -> stop rollout
            if reached:
                break

    return np.array(pos), parent, reached, goal_node, len(pos)


# --------------------------------------------------------------------------- #
# Per-run figure: Go-Explore tree + shortest path
# --------------------------------------------------------------------------- #
def plot_run(P, parent, path_pts, cfg, walls, label, seed, length, out):
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
    W, H = domain_size(cfg)
    pad = 0.02 * max(W, H)
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(-pad, H + pad)
    ax.set_aspect("equal")
    ax.set_title(f"{label} - Go-Explore RRT  seed={seed}\ntree nodes={len(P)}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main: sweep environments x seeds
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Go-Explore inspired RRT baseline over the shared mazes.")
    p.add_argument("--env", choices=list(ENVS), default=None,
                   help="single environment (default: all of "
                        + ", ".join(ENVS) + ")")
    p.add_argument("--seeds", type=int, default=SEEDS_DEFAULT,
                   help=f"number of seeds, 0..N-1 (default: {SEEDS_DEFAULT})")
    p.add_argument("--eps", type=float, default=None,
                   help=f"epsilon = action step size (default: {EPS_DEFAULT} for built-in "
                        "envs; step_to_width * passage_width for --layout, unless given here)")
    p.add_argument("--layout", type=str, action="append", default=None, metavar="FILE",
                   help="processed MiniHack layout JSON (repeatable); replaces --env")
    p.add_argument("--step-to-width", type=float, default=0.5, dest="step_to_width",
                   help="--layout only: eps auto-scales as this * passage_width (default 0.5)")
    p.add_argument("--clearance", type=float, default=None,
                   help="--layout only: wall clearance (default: layout's clearance_suggested)")
    p.add_argument("--k", type=int, default=30, help="seed-chain length (default 30)")
    p.add_argument("--j", type=int, default=25, help="rollout length (default 25)")
    p.add_argument("--m", type=int, default=40000,
                   help="max EXPLORE iterations (default 40000)")
    p.add_argument("--dtheta", type=float, default=0.5,
                   help="min angular separation between headings, rad (default 0.5)")
    p.add_argument("--ntry", type=int, default=20,
                   help="heading re-draw attempts (default 20)")
    p.add_argument("--max-nodes", type=int, default=40000, dest="max_nodes",
                   help="node budget per run before giving up (default 40000)")
    p.add_argument("--out", type=str, default="ge_results",
                   help="output directory (default: ge_results)")
    return p.parse_args()


def _targets(args):
    """Yields (out_dir, key, cfg, eps) for every built-in env or --layout file requested.
    key is the {env}_-style output prefix for built-ins (flat files, unchanged layout), or
    None for a --layout run (its own subfolder under args.out, named by run_name(cfg))."""
    if args.layout:
        if args.env:
            raise SystemExit("--layout cannot be combined with --env")
        for f in args.layout:
            ov = ["step_to_width"] if args.step_to_width != 0.5 else []
            cfg = Config(env="layout", layout_file=f, step_to_width=args.step_to_width)
            if args.clearance is not None:
                cfg.clearance = args.clearance
                ov.append("clearance")
            cfg.overrides = tuple(ov)
            apply_env(cfg)
            eps = args.eps if args.eps is not None else cfg.step_size
            name = run_name(cfg)
            out_dir = os.path.join(args.out, name)
            os.makedirs(out_dir, exist_ok=True)
            yield out_dir, name, cfg, eps
    else:
        for env in ([args.env] if args.env else list(ENVS)):
            cfg = apply_env(Config(env=env))
            eps = args.eps if args.eps is not None else EPS_DEFAULT
            os.makedirs(args.out, exist_ok=True)
            yield args.out, env, cfg, eps


def main():
    args = parse_args()

    overall = []
    for out_dir, name, cfg, eps in _targets(args):
        walls = build_walls(cfg)
        label = f"{name} ({env_label(cfg)})" if cfg.env != "layout" else env_label(cfg)
        print(f"\n=== {label}  start={cfg.start} goal={cfg.goal}"
              f"  eps={eps:.5g}  (Go-Explore RRT) ===")

        records = []
        lengths = []
        node_counts = []
        for seed in range(args.seeds):
            P, parent, reached, gnode, n = go_explore_rrt(
                cfg, walls, eps, seed,
                k=args.k, j=args.j, m=args.m, dtheta=args.dtheta,
                ntry=args.ntry, max_nodes=args.max_nodes)
            path_pts, length = dijkstra_shortest_path(
                P, parent, cfg, walls, np.array(cfg.goal, float), cfg.clearance)
            ok = path_pts is not None and math.isfinite(length)
            fname = f"{name}_seed{seed:02d}.png" if cfg.env != "layout" else f"seed{seed:02d}.png"
            img = os.path.join(out_dir, fname)
            plot_run(P, parent, path_pts, cfg, walls, label, seed,
                     length if ok else float("nan"), img)
            if ok:
                lengths.append(length)
            node_counts.append(n)
            records.append((seed, ok, length if ok else float("nan"), n, path_pts))
            print(f"  seed {seed:2d}: "
                  f"{'reached' if ok else 'FAILED ':7s}  "
                  f"len={length:7.3f}  nodes={n}")

        arr = np.array(lengths, float)
        mean = float(arr.mean()) if arr.size else float("nan")
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        narr = np.array(node_counts, float)         # nodes explored, over ALL seeds (reached or not)
        nmean = float(narr.mean()) if narr.size else float("nan")
        nstd = float(narr.std(ddof=1)) if narr.size > 1 else 0.0
        print(f"  --> {name}: {mean:.3f} +/- {std:.3f}  "
              f"over {len(lengths)}/{args.seeds} seeds that reached the goal  |  "
              f"nodes explored: {nmean:.1f} +/- {nstd:.1f} (all {len(node_counts)} seeds)")
        overall.append((name, mean, std, len(lengths), args.seeds, nmean, nstd))

        txt_name = f"{name}_go_explore_rrt_results.txt" if cfg.env != "layout" else "go_explore_rrt_results.txt"
        txt = os.path.join(out_dir, txt_name)
        with open(txt, "w") as f:
            f.write(f"# Go-Explore RRT  env={name} ({env_label(cfg)})\n")
            f.write(f"# start={cfg.start}  goal={cfg.goal}  eps(step)={eps:.5g}  "
                    f"goal_radius={cfg.goal_radius}  clearance={cfg.clearance:.5g}\n")
            f.write(f"# params: k={args.k} j={args.j} m={args.m} "
                    f"dtheta={args.dtheta} ntry={args.ntry}\n")
            f.write(f"# path length = Euclidean length of the start->goal path "
                    f"extracted by Dijkstra on the Go-Explore tree\n")
            f.write(f"# seeds = {args.seeds}\n\n")
            f.write(f"AVERAGE SHORTEST PATH: {mean:.4f} +/- {std:.4f} "
                    f"(sample std) over {len(lengths)}/{args.seeds} successful "
                    f"seeds\n")
            f.write(f"AVERAGE NODES EXPLORED: {nmean:.2f} +/- {nstd:.2f} "
                    f"(sample std) over all {len(node_counts)}/{args.seeds} seeds "
                    f"(reached or not)\n")
            f.write(f"lengths = {[round(x, 4) for x in lengths]}\n")
            f.write(f"nodes = {node_counts}\n\n")
            for seed, ok, length, n, path_pts in records:
                f.write("-" * 70 + "\n")
                if not ok:
                    f.write(f"seed {seed}: FAILED to reach goal (tree nodes={n})\n")
                    continue
                f.write(f"seed {seed}: length={length:.4f}  tree_nodes={n}  "
                        f"path_waypoints={len(path_pts)}\n")
                pts = ", ".join(f"({x:.3f},{y:.3f})" for x, y in path_pts)
                f.write(f"  path: {pts}\n")
        print(f"  [paths] saved -> {txt}")
        print(f"  [figs]  saved -> {out_dir}/{'seed' if cfg.env == 'layout' else name + '_seed'}*.png")

    summ = os.path.join(args.out, "summary.txt")
    with open(summ, "w") as f:
        f.write("Go-Explore RRT shortest-path length and nodes explored, "
                "mean +/- sample-std across seeds\n")
        f.write("(path length stats: over seeds that reached the goal; "
                "nodes-explored stats: over all seeds, reached or not)\n\n")
        for name, mean, std, k_, tot, nmean, nstd in overall:
            f.write(f"{name:35s}: length {mean:.4f} +/- {std:.4f}  "
                    f"({k_}/{tot} seeds reached)  |  nodes {nmean:.1f} +/- {nstd:.1f}\n")
    print(f"\n[summary] saved -> {summ}")
    print("\n==== SUMMARY (mean +/- std of shortest-path length and nodes explored) ====")
    for name, mean, std, k_, tot, nmean, nstd in overall:
        print(f"  {name:35s}: length {mean:.3f} +/- {std:.3f}  ({k_}/{tot} reached)  "
              f"|  nodes {nmean:.1f} +/- {nstd:.1f}")


if __name__ == "__main__":
    main()

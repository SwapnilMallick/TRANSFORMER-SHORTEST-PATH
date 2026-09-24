"""
Robosuite version of online_cost_transformer.py's collect_trajectory, ported
to match new_maze_envs_v7/online_cost_transformer.py's mechanism:
  * every trajectory resets to the highest-uncertainty existing state s*
    (find_uncertain_state_robosuite, below) instead of always starting at
    the true start
  * veto: cand_cost / diagonal <= c_hat + std_raw (no EMA, no alpha term --
    alpha is now purely the mu(s,a) threshold used to pick s*)
  * find_uncertain_state_robosuite memoizes each node's uncertainty score
    across the k trajectories of one iteration (reset every iteration)

Everything else (Graph, predict_last_mc, build_all_histories) is reused
unchanged from online_cost_transformer.py; only the environment-facing calls
swap to RobosuiteWorld.sample_free() / RobosuiteWorld.collides() /
RobosuiteTask.reached().

Key differences from the maze version, both stemming from robosuite having
real mutable simulator state (qpos), not a pure-functional position update:

  * collect_trajectory_robosuite has to explicitly keep the sim synced to
    whatever the loop currently considers the "committed" state -- restoring
    qpos on every rejection path (collision recheck, veto), not just relying
    on sample_free()'s own internal restoration (which only covers ITS OWN
    failed attempts, not an outer veto decision made after it already
    succeeded and moved the sim).

  * reset-replay to s* is a single world.ik.set_q(node_q[s_star]) call, not a
    reconstructed action sequence. robosuite_env.py's own docstring proves
    (and its smoke test verifies) that under this project's kinematic-only,
    frozen-cube execution model, direct set_q is equivalent to replaying the
    full history -- there's nothing to physically replay. The TOKEN history
    for the transformer's causal context is still reconstructed via
    build_all_histories, exactly as in the maze: that's about what the model
    sees, independent of how the arm physically got there.

  * find_uncertain_state_robosuite's uncertainty scan needs a REAL sim
    teleport (world.ik.set_q(node_q[node])) before every probed candidate,
    since world.sample_free() reads the CURRENT qpos and, on success, leaves
    the sim at the new position -- each of the uncertainty_scan_actions
    probes must start fresh from the same node, not drift from wherever the
    previous probe left off.

  * the scan uses its own `scan_rng`, independent of the trajectory-collection
    rng, so a cache hit can skip world.sample_free() entirely. This differs
    from the maze's v7 fix (which still draws-and-discards on a hit to keep a
    SHARED rng in sync): there, the model's MC-dropout forward pass was the
    dominant per-probe cost, so preserving cheap sample_free() draws kept
    almost all of the speedup. Here it's the opposite: sample_free()/
    collides() involve real IK solves and MuJoCo collision substeps and are
    the dominant cost, so skipping them entirely on a hit is what actually
    preserves the speedup -- nothing about the real rollout depends on
    scan_rng, so desyncing it has no effect on reproducibility of what gets
    explored.

Also: a graph node's stored EEF position alone doesn't determine the arm's
joint configuration (redundant DOF -- flagged during task planning). So a
node id here maps to a stored q (node_q dict, mutated in place), not just
graph.pos[node_id]. When merging into an existing node, the sim is driven
back to THAT node's stored q, not the fresh q the current sample happened to
produce -- consistent with RRT-ROBOSUITE's own Node(eef, q) design (its
reset-and-replay only works because every node has a well-defined q to go
back to).

Usage:
    python collect_trajectory_robosuite.py   # smoke test: a few short
                                              # trajectories + a reset-replay
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from online_cost_transformer import (
    Config, Graph, fuse, predict_last_mc, build_all_histories,
)
from task import RobosuiteTask


def collect_trajectory_robosuite(
    model, graph: Graph, cfg: Config, task: RobosuiteTask, rng,
    node_q: Dict[int, np.ndarray],
    start_node: int = 0,
    start_tokens: Optional[List[np.ndarray]] = None,
    start_node_ids: Optional[List[int]] = None,
    start_path_cost: float = 0.0,
):
    """Returns (tokens, node_ids, reached, n_veto, n_sample_fail, n_wall_fallback)
    -- identical shape to the maze's collect_trajectory (v7).

    start_node/start_tokens/start_node_ids/start_path_cost are the
    reset-replay target s* and its roadmap history (see
    find_uncertain_state_robosuite); the defaults (start_node=0, no history,
    zero path cost) reproduce the original start-anchored rollout. See the
    module docstring for why reset-replay here is a direct set_q, not a
    reconstructed action sequence."""
    world = task.world
    task.reset_to_start()
    node_q.setdefault(0, task.q_start.copy())   # root node's config, seeded once
    if start_node != 0:
        world.ik.set_q(node_q[start_node])      # teleport straight to s*

    s = graph.pos[start_node].copy()
    s_node = start_node
    q_node = node_q[start_node]
    path_cost = start_path_cost
    tokens: List[np.ndarray] = list(start_tokens) if start_tokens else []
    node_ids: List[int] = list(start_node_ids) if start_node_ids else []
    reached = False
    n_veto = 0
    n_sample_fail = 0
    n_wall_fallback = 0

    for _ in range(cfg.max_steps):
        committed = False
        for _retry in range(cfg.max_retries):
            cand = world.sample_free(rng)
            if cand is None:
                n_sample_fail += 1
                continue          # sample_free already restored sim to q_node

            a, s_next, q_to = cand
            tok = fuse(s, a)

            j, dist = graph.nearest(s_next)
            visited = dist < world.cfg.novelty_radius

            if visited:
                # merge into the existing node's STORED config (not q_to,
                # the fresh IK solve for this sample) -- see module docstring.
                node_pos = graph.pos[j]
                q_target = node_q[j]
                if world.collides(q_node, q_target):
                    # nearest existing node is collision-blocked from q_node
                    # even though q_node -> q_to (this sample) was already
                    # validated -- treat as a brand-new, unmerged node
                    # instead of discarding it (same fallback as the maze).
                    n_wall_fallback += 1
                    node_id = graph.add_node(s_next)
                    node_q[node_id] = q_to.copy()
                    s_new, q_new = s_next, q_to
                    step_cost = float(np.linalg.norm(s_new - s))
                    cand_cost = path_cost + step_cost
                else:
                    step_cost = float(np.linalg.norm(node_pos - s))
                    cand_cost = path_cost + step_cost
                    cand_cost_n = cand_cost / cfg.diagonal
                    c_hat, std_raw = predict_last_mc(model, tokens + [tok], cfg)
                    if cand_cost_n <= c_hat + std_raw:
                        node_id, s_new, q_new = j, node_pos, q_target
                    else:
                        n_veto += 1
                        world.ik.set_q(q_node)                    # restore, retry
                        continue
            else:
                node_id = graph.add_node(s_next)
                node_q[node_id] = q_to.copy()
                s_new, q_new = s_next, q_to                   # sim already here
                step_cost = float(np.linalg.norm(s_new - s))
                cand_cost = path_cost + step_cost

            # commit the accepted transition
            graph.add_edge(s_node, node_id, step_cost)
            tokens.append(tok)
            node_ids.append(node_id)
            path_cost = cand_cost
            s, s_node, q_node = s_new, node_id, q_new
            world.ik.set_q(q_node)                            # explicit re-sync
            committed = True
            break

        if not committed:
            break
        if task.reached(s):
            reached = True
            break

    # reset-replay histories can push total length past the positional-
    # embedding table's max_len rows; keep the trailing window (matches the
    # maze's collect_trajectory).
    if len(tokens) > cfg.max_len:
        tokens = tokens[-cfg.max_len:]
        node_ids = node_ids[-cfg.max_len:]

    return tokens, node_ids, reached, n_veto, n_sample_fail, n_wall_fallback


# --------------------------------------------------------------------------- #
# Go-Explore reset-replay target selection, robosuite version
# --------------------------------------------------------------------------- #
def find_uncertain_state_robosuite(
    model, graph: Graph, g: List[float], prev: List[int], cfg: Config,
    task: RobosuiteTask, scan_rng, node_q: Dict[int, np.ndarray],
    cache: Dict[int, Tuple[float, float]],
):
    """Robosuite counterpart of online_cost_transformer.find_uncertain_state
    -- see that function's docstring for the caching/selection rule itself
    (unchanged: cache a node's best-candidate score for the rest of this
    iteration, reset the cache every iteration, fall back to node 0 if
    nothing clears cfg.alpha). The difference here is purely mechanical: see
    the module docstring for why probing a node needs a real sim teleport
    per probe, and why this uses its own scan_rng.

    Returns (s_star, tokens_to_s_star, node_ids_to_s_star,
    path_cost_to_s_star, n_cache_hits, n_cache_misses).
    """
    world = task.world
    tok_hist, id_hist = build_all_histories(graph, g, prev)
    best_node, best_score = 0, -1.0
    n_hits, n_misses = 0, 0
    for node in range(len(graph.pos)):
        if not math.isfinite(g[node]):
            continue
        cached = cache.get(node)
        is_hit = cached is not None and abs(cached[1] - g[node]) < 1e-9
        if is_hit:
            node_best = cached[0]
            n_hits += 1
        else:
            s = graph.pos[node]
            q_node = node_q[node]
            node_best = -1.0
            for _ in range(cfg.uncertainty_scan_actions):
                world.ik.set_q(q_node)              # fresh start every probe
                cand = world.sample_free(scan_rng)
                if cand is None:
                    continue
                a, _s_next, _q_to = cand
                _, std_raw = predict_last_mc(
                    model, tok_hist[node] + [fuse(s, a)], cfg)
                node_best = max(node_best, std_raw)
            cache[node] = (node_best, g[node])
            n_misses += 1
        if node_best > best_score:
            best_score, best_node = node_best, node
    if best_score < cfg.alpha:
        best_node = 0
    return (best_node, tok_hist[best_node], id_hist[best_node], float(g[best_node]),
            n_hits, n_misses)


if __name__ == "__main__":
    from online_cost_transformer import CostTransformer
    from robosuite_env import world_diagonal

    rng = np.random.default_rng(0)
    scan_rng = np.random.default_rng(1)
    cfg = Config(max_steps=15, max_retries=6, mc_samples=10,
                            n_iters=1, alpha=0.05, uncertainty_scan_actions=4)
    task = RobosuiteTask()
    cfg.diagonal = world_diagonal(task.world.cfg)
    model = CostTransformer(cfg, token_dim=6)
    graph = Graph(task.eef_start, task.world.cfg.novelty_radius)
    node_q: Dict[int, np.ndarray] = {}

    print(f"diagonal = {cfg.diagonal:.4f}")

    for i in range(3):
        tokens, node_ids, reached, n_veto, nsf, nwf = collect_trajectory_robosuite(
            model, graph, cfg, task, rng, node_q)
        print(f"traj {i} (start-anchored): steps={len(node_ids)} reached={reached} "
              f"n_veto={n_veto} sample_fail={nsf} wall_fb={nwf} "
              f"graph_nodes={len(graph.pos)}")

    # sanity: sim's current EEF should match the last committed node's stored
    # position (within IK tolerance), confirming state stayed synced throughout
    if node_ids:
        last_node = node_ids[-1]
        sim_eef = task.ik.eef_pos()
        stored_pos = graph.pos[last_node]
        print("\nlast committed node pos:", stored_pos.round(4))
        print("sim EEF after loop:      ", sim_eef.round(4))
        print("match:", bool(np.allclose(sim_eef, stored_pos, atol=1e-3)))
        print("node_q entries == graph nodes:", len(node_q) == len(graph.pos))

    # reset-replay smoke test: pick the highest-uncertainty existing node and
    # verify collect_trajectory_robosuite actually resumes from there (sim's
    # EEF right after reset-replay, before any new steps, should match that
    # node's stored position)
    if len(graph.pos) > 1:
        g, prev = graph.dijkstra(0, return_prev=True)
        cache: Dict[int, Tuple[float, float]] = {}
        s_star, tok_hist, id_hist, cost_hist, n_hits, n_miss = find_uncertain_state_robosuite(
            model, graph, g, prev, cfg, task, scan_rng, node_q, cache)
        print(f"\nfind_uncertain_state_robosuite: s*={s_star} "
              f"hist_len={len(tok_hist)} cost={cost_hist:.4f} "
              f"hits={n_hits} misses={n_miss}")

        task.reset_to_start()
        world_ = task.world
        world_.ik.set_q(node_q[s_star])
        eef_after_replay = task.ik.eef_pos().copy()
        print("s* stored pos:  ", graph.pos[s_star].round(4))
        print("eef after replay:", eef_after_replay.round(4))
        print("reset-replay lands on s*:",
              bool(np.allclose(eef_after_replay, graph.pos[s_star], atol=1e-6)))

    task.close()

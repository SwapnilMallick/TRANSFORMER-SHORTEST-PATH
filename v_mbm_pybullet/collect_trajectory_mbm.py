"""
MBM (MotionBenchMaker, PyBullet) version of the online cost-to-come collection loop, adapted from
legacy_robosuite/collect_trajectory_robosuite.py. The ALGORITHM is unchanged: every trajectory resets to the
highest-uncertainty roadmap node s* (find_uncertain_state_mbm; falls back to node 0 below alpha), the veto is
cand_cost / diagonal <= c_hat + std_raw, the scan memoizes each node's score for one iteration and uses its
own `scan_rng`. Graph, predict_last_mc and build_all_histories are reused from online_cost_transformer.py.

What changed versus the robosuite loop (see also CLAUDE.md):

  * States are 7-D joint configurations: `graph.pos[i]` IS node i's q. Novelty merging, step costs, the veto's
    comparison cost and Dijkstra labels are all joint-space Euclidean. There is no node_q dict and no IK.
  * The environment is a pure function of q: no set_q restore/resync bookkeeping. The only place a PyBullet
    robot must be at a specific q is right before rendering / FK, and those calls set it explicitly.
  * Every NEW node is embedded once, at its own q, AFTER its transition is committed:
        renderer.set_q(q_new); node_emb[id] = encoder.embed(renderer); node_eef[id] = fk_eef(q_new)
    Merged nodes reuse their stored embedding. `node_eef` is for plots only, never a model input.
  * Merge fallback: when the nearest existing node j is within the novelty radius but the STORED edge
    q_node -> graph.pos[j] is not collision-free (valid_motion), the sample becomes a new unmerged node.
  * Goal connection: after each commit, if the joint distance to a goal configuration is <= goal_connect and
    valid_motion(q, q_goal) holds, an exact goal node at q_goal is added (or reused) and that edge is committed
    as one more step (token action = q_goal - q, cost = joint distance). `goal_nodes` (a set, mutated in place)
    tracks goal nodes; the trajectory is `reached` iff its current node is in it.
  * The uncertainty scan calls task.sample_free(graph.pos[node], scan_rng) directly (no teleport).
  * Token width is V*D + 7 (joint-increment action).

Usage:
    python collect_trajectory_mbm.py   # smoke test: 3 short trajectories, bookkeeping + embedding-consistency
                                       # checks and one find_uncertain_state_mbm call
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from online_cost_transformer import (
    Config, Graph, fuse, predict_last_mc, build_all_histories,
)
from robot_mbm import N_DOF
from task_mbm import MBMTask


def state_feat(node_emb: Dict[int, np.ndarray], n: int) -> np.ndarray:
    """Flat (V*D,) state half of a token for node `n`."""
    return node_emb[n].reshape(-1)


def _goal_node_for(graph: Graph, goal_nodes: Set[int], q_goal: np.ndarray) -> Optional[int]:
    for n in goal_nodes:
        if np.array_equal(graph.pos[n], q_goal):
            return n
    return None


def collect_trajectory_mbm(
    model, graph: Graph, cfg: Config, task: MBMTask, rng,
    node_emb: Dict[int, np.ndarray], node_eef: Dict[int, np.ndarray],
    goal_nodes: Set[int], encoder,
    start_node: int = 0,
    start_tokens: Optional[List[np.ndarray]] = None,
    start_node_ids: Optional[List[int]] = None,
    start_path_cost: float = 0.0,
    stats: Optional[dict] = None,
):
    """Returns (tokens, node_ids, reached, n_veto, n_sample_fail, n_wall_fallback) -- the same tuple as the
    robosuite loop. node_emb / node_eef / goal_nodes are mutated in place. `stats` (optional dict) receives
    counters that do not fit the tuple: 'merges' (steps into an existing node), 'goal_connects'."""
    checker, renderer = task.checker, task.renderer
    assert 0 in node_emb and 0 in node_eef, "node 0 must be embedded by the caller before collecting"
    tok_width = encoder.n_views * encoder.emb_dim + N_DOF

    q = np.array(graph.pos[start_node], float)
    s_node = start_node
    path_cost = start_path_cost
    tokens: List[np.ndarray] = list(start_tokens) if start_tokens else []
    node_ids: List[int] = list(start_node_ids) if start_node_ids else []
    reached = False
    n_veto = n_sample_fail = n_wall_fallback = 0
    if stats is None:
        stats = {}

    def commit(u, v_node, tok, step_cost, q_new, new_node):
        nonlocal path_cost, q, s_node
        graph.add_edge(u, v_node, step_cost)
        tokens.append(tok)
        node_ids.append(v_node)
        path_cost += step_cost
        q, s_node = q_new, v_node
        if new_node:
            # embedded only AFTER the commit, at the node's own q (merged nodes reuse their stored embedding)
            renderer.set_q(q_new)
            node_emb[v_node] = encoder.embed(renderer)
            node_eef[v_node] = task.fk_eef(q_new)

    for _ in range(cfg.max_steps):
        committed = False
        for _retry in range(cfg.max_retries):
            cand = task.sample_free(q, rng)
            if cand is None:
                n_sample_fail += 1
                continue

            dq, q_next = cand
            tok = fuse(state_feat(node_emb, s_node), dq)        # image state + joint increment; q never enters
            assert tok.shape[0] == tok_width, f"token width {tok.shape[0]} != {tok_width}"

            j, dist = graph.nearest(q_next)
            visited = dist < task.cfg.novelty_radius
            if visited:
                q_target = graph.pos[j]
                if not checker.valid_motion(q, q_target):
                    # nearest node is collision-blocked from q even though q -> q_next was validated:
                    # keep the sample as a brand-new, unmerged node instead of discarding it
                    n_wall_fallback += 1
                    node_id, q_new, new_node = graph.add_node(q_next), q_next, True
                    step_cost = float(np.linalg.norm(q_new - q))
                else:
                    step_cost = float(np.linalg.norm(q_target - q))
                    cand_cost_n = (path_cost + step_cost) / cfg.diagonal
                    c_hat, std_raw = predict_last_mc(model, tokens + [tok], cfg)
                    if cand_cost_n <= c_hat + std_raw:
                        node_id, q_new, new_node = j, q_target, False
                        stats["merges"] = stats.get("merges", 0) + 1
                    else:
                        n_veto += 1
                        continue                                # VETO -> retry
            else:
                node_id, q_new, new_node = graph.add_node(q_next), q_next, True
                step_cost = float(np.linalg.norm(q_new - q))

            commit(s_node, node_id, tok, step_cost, q_new, new_node)
            committed = True
            break

        if not committed:
            break

        # exact goal connection (after the transition above is fully committed)
        if s_node not in goal_nodes:
            for gi in np.argsort([np.linalg.norm(q - g) for g in task.goals]):
                q_goal = task.goals[int(gi)]
                if (np.linalg.norm(q - q_goal) <= task.cfg.goal_connect
                        and checker.valid_motion(q, q_goal)):
                    gnode = _goal_node_for(graph, goal_nodes, q_goal)
                    is_new = gnode is None
                    if is_new:
                        gnode = graph.add_node(q_goal)
                        goal_nodes.add(gnode)
                    tok = fuse(state_feat(node_emb, s_node), q_goal - q)
                    assert tok.shape[0] == tok_width
                    commit(s_node, gnode, tok, float(np.linalg.norm(q_goal - q)), q_goal.copy(), is_new)
                    stats["goal_connects"] = stats.get("goal_connects", 0) + 1
                    break
        if s_node in goal_nodes:
            reached = True
            break

    # reset-replay histories can push total length past the positional-embedding table's max_len rows;
    # keep the trailing window (matches the maze's collect_trajectory)
    if len(tokens) > cfg.max_len:
        tokens = tokens[-cfg.max_len:]
        node_ids = node_ids[-cfg.max_len:]

    return tokens, node_ids, reached, n_veto, n_sample_fail, n_wall_fallback


# --------------------------------------------------------------------------- #
# Go-Explore reset-replay target selection
# --------------------------------------------------------------------------- #
def find_uncertain_state_mbm(
    model, graph: Graph, g: List[float], prev: List[int], cfg: Config,
    task: MBMTask, scan_rng, node_emb: Dict[int, np.ndarray],
    cache: Dict[int, Tuple[float, float]],
):
    """MBM counterpart of online_cost_transformer.find_uncertain_state (same selection/caching rule: cache a
    node's best-candidate score for the rest of this iteration, reset the cache every iteration, fall back to
    node 0 if nothing clears cfg.alpha). Probes call task.sample_free(graph.pos[node], scan_rng) directly; the
    scan has its own RNG so a cache hit can skip sampling entirely without changing what gets explored.
    Nothing is rendered: a probe token is [node_emb[node], a].

    Returns (s_star, tokens_to_s_star, node_ids_to_s_star, path_cost_to_s_star, n_cache_hits, n_cache_misses)."""
    tok_hist, id_hist = build_all_histories(
        graph, g, prev, state_feat=lambda n: state_feat(node_emb, n))
    tok_width = node_emb[0].size + N_DOF
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
            feat = state_feat(node_emb, node)
            node_best = -1.0
            for _ in range(cfg.uncertainty_scan_actions):
                cand = task.sample_free(graph.pos[node], scan_rng)
                if cand is None:
                    continue
                tok = fuse(feat, cand[0])
                assert tok.shape[0] == tok_width, f"probe token width {tok.shape[0]} != {tok_width}"
                _, std_raw = predict_last_mc(model, tok_hist[node] + [tok], cfg)
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
    import argparse
    from online_cost_transformer import ImageCostTransformer
    from image_encoder import EncoderConfig, ImageEncoder
    from scene_mbm import DEFAULT_SCENES_DIR, scene_path
    from task_mbm import MBMTaskConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="table_pick")
    ap.add_argument("--problem", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    scan_rng = np.random.default_rng(1)
    cfg = Config(max_steps=15, max_retries=6, mc_samples=10, n_iters=1, alpha=0.05,
                 uncertainty_scan_actions=4)
    task = MBMTask(scene_path(DEFAULT_SCENES_DIR, a.scene, a.problem), MBMTaskConfig())
    cfg.diagonal = task.diagonal
    encoder = ImageEncoder(EncoderConfig(n_calib=40, device=a.device))
    encoder.calibrate(task, np.random.default_rng(2))
    task.renderer.set_q(task.q_start)
    node_emb: Dict[int, np.ndarray] = {0: encoder.embed(task.renderer)}
    node_eef: Dict[int, np.ndarray] = {0: task.fk_eef(task.q_start)}
    goal_nodes: Set[int] = set()
    model = ImageCostTransformer(cfg, encoder.n_views, encoder.emb_dim, action_dim=N_DOF,
                                 action_scale=1.0 / task.cfg.step_size)
    graph = Graph(task.q_start, task.cfg.novelty_radius)
    ok_all = True
    print(f"diagonal = {cfg.diagonal:.4f} | token width = {encoder.n_views * encoder.emb_dim + N_DOF}")

    for i in range(3):
        tokens, node_ids, reached, n_veto, nsf, nwf = collect_trajectory_mbm(
            model, graph, cfg, task, rng, node_emb, node_eef, goal_nodes, encoder)
        print(f"traj {i} (start-anchored): steps={len(node_ids)} reached={reached} n_veto={n_veto} "
              f"sample_fail={nsf} wall_fb={nwf} graph_nodes={len(graph.pos)}")

    sync = len(node_emb) == len(node_eef) == len(graph.pos)
    print("len(node_emb) == len(node_eef) == len(graph.pos):", sync)
    ok_all &= sync

    # embeddings must be tied to each node's own q: re-render a few random nodes and compare
    pick = np.random.default_rng(3).choice(len(graph.pos), size=min(3, len(graph.pos)), replace=False)
    for n in pick:
        task.renderer.set_q(graph.pos[int(n)])
        fresh = encoder.embed(task.renderer)
        same = bool(np.allclose(fresh, node_emb[int(n)], atol=1e-4))
        print(f"embedding consistency, node {int(n):4d}: max|diff|="
              f"{np.abs(fresh - node_emb[int(n)]).max():.2e} -> {same}")
        ok_all &= same

    if len(graph.pos) > 1:
        g, prev = graph.dijkstra(0, return_prev=True)
        cache: Dict[int, Tuple[float, float]] = {}
        s_star, tok_hist, id_hist, cost_hist, n_hits, n_miss = find_uncertain_state_mbm(
            model, graph, g, prev, cfg, task, scan_rng, node_emb, cache)
        print(f"\nfind_uncertain_state_mbm: s*={s_star} hist_len={len(tok_hist)} cost={cost_hist:.4f} "
              f"hits={n_hits} misses={n_miss}")
        ok_all &= (0 <= s_star < len(graph.pos)) and len(tok_hist) == len(id_hist)

    print("\nALL SMOKE CHECKS PASSED" if ok_all else "\nSOME SMOKE CHECKS FAILED")
    task.close()

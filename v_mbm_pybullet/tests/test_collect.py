"""Collection loop: goal connection, bookkeeping and embedding consistency (with a cheap stand-in encoder that
pools rendered pixels, so no DINOv2 is needed; the real-encoder smoke test is `python collect_trajectory_mbm.py`)."""
import numpy as np
import pytest

from collect_trajectory_mbm import collect_trajectory_mbm, find_uncertain_state_mbm, state_feat
from conftest import empty_scene, scene_json
from online_cost_transformer import Config, Graph, ImageCostTransformer
from robot_mbm import N_DOF
from task_mbm import MBMTask, MBMTaskConfig


class PoolEncoder:
    """(V, D) embedding = 4x4-pooled grey render of two cameras. Deterministic function of the current q."""
    n_views, emb_dim = 2, 64

    def embed(self, renderer):
        out = []
        for cam in ("front", "wrist"):
            im = renderer.render(cam, 32, 32).astype(np.float32).mean(-1) / 255.0
            out.append(im.reshape(8, 4, 8, 4).mean((1, 3)).reshape(-1))
        return np.stack(out).astype(np.float32)


def _setup(task, **cfg_kw):
    cfg = Config(max_steps=6, max_retries=6, mc_samples=6, n_iters=1, alpha=0.05, uncertainty_scan_actions=3,
                 **cfg_kw)
    cfg.diagonal = task.diagonal
    enc = PoolEncoder()
    task.renderer.set_q(task.q_start)
    node_emb = {0: enc.embed(task.renderer)}
    node_eef = {0: task.fk_eef(task.q_start)}
    model = ImageCostTransformer(cfg, enc.n_views, enc.emb_dim, action_dim=N_DOF,
                                 action_scale=1.0 / task.cfg.step_size)
    graph = Graph(task.q_start, task.cfg.novelty_radius)
    return cfg, enc, node_emb, node_eef, set(), model, graph


def test_goal_connection_makes_exact_goal_node(tp0):
    delta = np.zeros(7)
    delta[0] = 0.3                                              # 0.3 rad from the start: within 0.5 after one step
    sc = empty_scene(tp0, [], start=tp0.start, goals=[tp0.start + delta])
    task = MBMTask(None, MBMTaskConfig(goal_connect=0.5), scene=sc)
    cfg, enc, node_emb, node_eef, goal_nodes, model, graph = _setup(task)
    rng = np.random.default_rng(0)
    tokens, node_ids, reached, *_ = collect_trajectory_mbm(
        model, graph, cfg, task, rng, node_emb, node_eef, goal_nodes, enc)
    assert reached and len(goal_nodes) == 1
    gnode = next(iter(goal_nodes))
    assert node_ids[-1] == gnode and len(node_ids) == 2           # one exploration step + the goal connection
    assert np.array_equal(graph.pos[gnode], sc.goals[0]) and task.reached(graph.pos[gnode])
    prev = node_ids[0]
    a_last = tokens[-1][-N_DOF:]
    np.testing.assert_allclose(a_last, (sc.goals[0] - graph.pos[prev]).astype(np.float32), atol=1e-6)
    assert graph.adj[prev][gnode] == pytest.approx(np.linalg.norm(sc.goals[0] - graph.pos[prev]))
    assert len(node_emb) == len(node_eef) == len(graph.pos)
    # second trajectory reuses the existing goal node instead of adding a duplicate
    n_before = len(graph.pos)
    collect_trajectory_mbm(model, graph, cfg, task, rng, node_emb, node_eef, goal_nodes, enc)
    assert len(goal_nodes) == 1 and len(graph.pos) - n_before <= 1
    task.close()


def test_no_goal_connection_beyond_radius_or_through_wall(tp0):
    far = tp0.start.copy()
    far[0] += 1.5                                               # farther than goal_connect
    task = MBMTask(None, MBMTaskConfig(goal_connect=0.5), scene=empty_scene(tp0, [], goals=[far]))
    cfg, enc, node_emb, node_eef, goal_nodes, model, graph = _setup(task)
    _, _, reached, *_ = collect_trajectory_mbm(model, graph, cfg, task, np.random.default_rng(0),
                                               node_emb, node_eef, goal_nodes, enc)
    assert not reached and not goal_nodes
    task.close()


def test_bookkeeping_and_embedding_consistency(tp0):
    task = MBMTask(scene_json("table_pick", 0), MBMTaskConfig())
    cfg, enc, node_emb, node_eef, goal_nodes, model, graph = _setup(task)
    rng, scan_rng = np.random.default_rng(0), np.random.default_rng(1)
    for _ in range(3):
        toks, ids, reached, nv, nsf, nwf = collect_trajectory_mbm(
            model, graph, cfg, task, rng, node_emb, node_eef, goal_nodes, enc)
        assert all(t.shape[0] == enc.n_views * enc.emb_dim + N_DOF for t in toks)
    assert len(node_emb) == len(node_eef) == len(graph.pos) > 10
    for n in np.random.default_rng(3).choice(len(graph.pos), 3, replace=False):
        task.renderer.set_q(graph.pos[int(n)])
        np.testing.assert_allclose(enc.embed(task.renderer), node_emb[int(n)], atol=1e-4)
        np.testing.assert_allclose(task.fk_eef(graph.pos[int(n)]), node_eef[int(n)], atol=1e-9)
    g, prev = graph.dijkstra(0, return_prev=True)
    s_star, tok_hist, id_hist, cost, hits, misses = find_uncertain_state_mbm(
        model, graph, g, prev, cfg, task, scan_rng, node_emb, {})
    assert 0 <= s_star < len(graph.pos) and len(tok_hist) == len(id_hist) and misses == len(graph.pos)
    # resuming from s* keeps the history prefix
    toks, ids, *_ = collect_trajectory_mbm(model, graph, cfg, task, rng, node_emb, node_eef, goal_nodes, enc,
                                           start_node=s_star, start_tokens=tok_hist, start_node_ids=id_hist,
                                           start_path_cost=cost)
    assert ids[:len(id_hist)] == id_hist or len(toks) == cfg.max_len
    task.close()


def test_same_seed_same_roadmap(tp0):
    roadmaps = []
    for _ in range(2):
        import torch
        torch.manual_seed(0)
        task = MBMTask(scene_json("table_pick", 0), MBMTaskConfig())
        cfg, enc, node_emb, node_eef, goal_nodes, model, graph = _setup(task)
        rng = np.random.default_rng(0)
        for _ in range(2):
            collect_trajectory_mbm(model, graph, cfg, task, rng, node_emb, node_eef, goal_nodes, enc)
        roadmaps.append(np.stack(graph.pos))
        task.close()
    np.testing.assert_array_equal(roadmaps[0], roadmaps[1])

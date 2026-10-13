"""Tests for the processed-layout wiring in online_cost_transformer.py."""
import glob
import json
import math
import os
from dataclasses import replace

import numpy as np
import pytest

import layout_geometry as G
import online_cost_transformer as O

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = sorted(glob.glob(os.path.join(ROOT, "layouts", "processed", "*.json")))
needs_layouts = pytest.mark.skipif(not PROCESSED, reason="no processed layouts")
BUILTINS = [e for e in O.ENVS if e != "layout"]


def layout_path(sub):
    m = [p for p in PROCESSED if sub in os.path.basename(p)]
    assert m, sub
    return m[0]


def random_moves(rng, n, lo, hi, max_len):
    p = rng.uniform(lo, hi, (n, 2))
    ang = rng.uniform(0, 2 * np.pi, n)
    q = p + np.stack([np.cos(ang), np.sin(ang)], 1) * rng.uniform(0, max_len, (n, 1))
    return p, q


# ---- collision -------------------------------------------------------------- #
@pytest.mark.parametrize("env", BUILTINS)
def test_collides_clearance0_matches_original(env):
    cfg = O.apply_env(O.Config(env=env))
    plain = list(O.build_walls(cfg))                       # original list path (python loop)
    walls = O.build_walls(cfg)                             # Walls: vectorized path
    rng = np.random.default_rng(0)
    P, Q = random_moves(rng, 3000, 0, cfg.L, 0.3)
    for p, q in zip(P, Q):
        ref = any(O._seg_intersect(p, q, w0, w1) for w0, w1 in plain)
        assert O.collides(p, q, walls) == ref == O.collides(p, q, plain)


@needs_layouts
@pytest.mark.parametrize("sub,clr", [("corridor_r3_12x5_s1", None), ("mazewalk_9x9_s0", None)])
def test_collides_agrees_with_geometry(sub, clr):
    d = O.load_layout(layout_path(sub))
    walls = O.Walls((np.array(w[:2]), np.array(w[2:])) for w in d["walls"])
    arr = np.array(d["walls"])
    pw = d["passage_width"]
    rng = np.random.default_rng(1)
    n = 100_000 if "corridor" in sub else 30_000
    P, Q = random_moves(rng, n, 0, max(d["domain"]["W"], d["domain"]["H"]), 2 * pw)
    # half of the segments start right next to a wall endpoint, where clearance matters
    k = n // 2
    P[:k] = arr[rng.integers(0, len(arr), k), :2] + rng.normal(0, 0.5 * pw, (k, 2))
    Q[:k] = P[:k] + rng.normal(0, pw, (k, 2))
    for clearance in (0.0, d["clearance_suggested"]):
        ref = G._segments_collide_many(P, Q, arr, clearance)
        mine = np.array([O.collides(p, q, walls, clearance) for p, q in zip(P, Q)])
        assert (ref == mine).all(), (clearance, int((ref != mine).sum()))
        assert ref.any() and not ref.all()


# ---- loading ---------------------------------------------------------------- #
@needs_layouts
@pytest.mark.parametrize("path", PROCESSED)
def test_every_layout_loads(path):
    d = O.load_layout(path)
    cfg = O.apply_env(O.Config(env="layout", layout_file=path))
    assert (cfg.W, cfg.H) == (d["domain"]["W"], d["domain"]["H"])
    # start / goal are away from every wall by at least the clearance
    walls = O.build_walls(cfg)
    for pt in (cfg.start, cfg.goal):
        x = np.array(pt)
        assert not O.collides(x - [1e-6, 0], x + [1e-6, 0], walls, cfg.clearance)
    assert O.run_name(cfg) == O.layout_run_name(path)


@needs_layouts
def test_failed_topology_is_refused(tmp_path):
    d = json.load(open(layout_path("mazewalk_9x9_s0")))
    for key, val in (("reachable", False), ("components_ok", False), ("n_shortcuts", 3)):
        bad = json.loads(json.dumps(d))
        bad["topology"][key] = val
        f = tmp_path / f"bad_{key}.json"
        f.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="topology"):
            O.load_layout(str(f))
    bad = json.loads(json.dumps(d))
    del bad["walls"]
    (tmp_path / "missing.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing"):
        O.load_layout(str(tmp_path / "missing.json"))
    bad = json.loads(json.dumps(d))
    bad["start"] = [bad["domain"]["W"] + 1, 0.1]
    (tmp_path / "outside.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="outside"):
        O.load_layout(str(tmp_path / "outside.json"))


# ---- sampling ---------------------------------------------------------------- #
@needs_layouts
@pytest.mark.parametrize("sub", ["mazewalk_15x15_s0", "corridor_r5_12x5_s0"])
def test_sample_free_stays_inside_and_never_crosses(sub):
    cfg = O.apply_env(O.Config(env="layout", layout_file=layout_path(sub)))
    walls = O.build_walls(cfg)
    arr = np.array(O.load_layout(cfg.layout_file)["walls"])
    rng = np.random.default_rng(0)
    goal = np.array(cfg.goal)
    s = np.array(cfg.start)
    n_ok = 0
    for _ in range(1500):
        out = O.sample_free(s, goal, cfg, walls, rng)
        if out is None:
            s = np.array(cfg.start)
            continue
        a, s2 = out
        assert 0 <= s2[0] <= cfg.W and 0 <= s2[1] <= cfg.H
        assert not G.segment_collides(s, s2, arr, 0.0)
        assert not G.segment_collides(s, s2, arr, cfg.clearance)
        n_ok += 1
        s = s2
    assert n_ok > 500


# ---- auto-scaling ------------------------------------------------------------ #
@needs_layouts
def test_auto_scaling_formulas():
    p = layout_path("corridor_r5_12x5_s0")
    d = O.load_layout(p)
    cfg = O.apply_env(O.Config(env="layout", layout_file=p, step_to_width=0.25))
    pw, ref = d["passage_width"], d["reference_path_length"]
    assert cfg.step_size == pytest.approx(0.25 * pw)
    assert cfg.novelty_radius == pytest.approx(0.2 * cfg.step_size)
    assert cfg.goal_radius == pytest.approx(0.4 * pw)
    assert cfg.clearance == pytest.approx(d["clearance_suggested"])
    assert cfg.max_steps == math.ceil(3.0 * ref / cfg.step_size)
    assert cfg.noise_std == pytest.approx(cfg.step_size * 0.005 / 0.15)
    assert cfg.scale_notes and "passage_width" in cfg.scale_notes[0]


@needs_layouts
def test_explicit_overrides_win():
    p = layout_path("mazewalk_9x9_s0")
    cfg = O.apply_env(O.Config(env="layout", layout_file=p, step_size=0.07, novelty_radius=0.011,
                               goal_radius=0.02, max_steps=33, clearance=0.003, noise_std=0.001,
                               overrides=("step_size", "novelty_radius", "goal_radius", "max_steps",
                                          "noise_std")))
    assert (cfg.step_size, cfg.novelty_radius, cfg.goal_radius, cfg.max_steps) == (0.07, 0.011, 0.02, 33)
    assert cfg.noise_std == 0.001 and cfg.clearance == 0.003


# ---- built-ins are untouched --------------------------------------------------- #
@pytest.mark.parametrize("env", BUILTINS)
def test_builtin_domain(env):
    cfg = O.apply_env(O.Config(env=env))
    assert O.domain_size(cfg) == (cfg.L, cfg.L)
    assert O.domain_diagonal(cfg) == math.sqrt(2) * cfg.L
    assert O.run_name(cfg) == env and cfg.clearance == 0.0 and cfg.W is None
    assert cfg.step_size == 0.15 and cfg.novelty_radius == 0.03


def test_no_minihack_imports():
    src = open(os.path.join(ROOT, "online_cost_transformer.py")).read()
    assert "import minihack" not in src and "import nle" not in src and "from nle" not in src
    assert "import layout_geometry" not in src and "from layout_geometry" not in src


# ---- final path query (history vs free) ------------------------------------------ #
def _tiny_graph(cfg):
    g = O.Graph(cfg.start, cfg.novelty_radius)
    pts = [(0.15, 0.10), (0.20, 0.15), (0.25, 0.20), (0.30, 0.25)]
    for k, p in enumerate(pts):
        j = g.add_node(np.array(p))
        g.add_edge(k, j, float(np.linalg.norm(np.array(p) - g.pos[k])))
    return g


def test_history_field_shape_and_start():
    import torch
    cfg = O.apply_env(O.Config(env="four_room"))
    cfg.diagonal = O.domain_diagonal(cfg)
    torch.manual_seed(0)
    model = O.CostTransformer(cfg)
    graph = _tiny_graph(cfg)
    g, prev = graph.dijkstra(0, return_prev=True)
    chat = O.predict_cost_field_history(model, graph, g, prev, cfg)
    assert chat.shape == (len(graph.pos),) and chat[0] == 0.0 and np.isfinite(chat).all()
    # a node's value equals a direct forward pass on its own history (last token)
    hist, _ = O.build_all_histories(graph, g, prev)
    x = torch.tensor(np.array(hist[3]), dtype=torch.float32)[None]
    model.eval()
    ref = float(model(x, O.causal_mask(x.shape[1], cfg.device), None)[0, -1])
    assert chat[3] == pytest.approx(ref, abs=1e-5)
    seq, ok = O.transformer_path(model, graph, cfg, 3, chat=chat)
    assert ok and seq[0] == 0 and seq[-1] == 3


@pytest.mark.parametrize("mode", ["history", "free"])
def test_path_query_modes_saved(tmp_path, mode):
    cfg = O.Config(env="four_room", n_iters=1, k_trajectories=1, outdir=str(tmp_path), path_query=mode)
    O.main(cfg)
    d = json.load(open(tmp_path / "four_room_paths.json"))
    assert d["path_query"] == mode
    assert d["transformer_path_alt"]["query"] == ("free" if mode == "history" else "history")
    assert d["transformer_path"]["nodes"] and d["transformer_path_alt"]["nodes"]


def test_bad_path_query_rejected(tmp_path):
    cfg = O.Config(env="four_room", n_iters=1, k_trajectories=1, outdir=str(tmp_path), path_query="bogus")
    with pytest.raises(ValueError, match="path_query"):
        O.main(cfg)


# ---- first goal reach record ------------------------------------------------------- #
def test_first_goal_reach_recorded(tmp_path):
    cfg = O.Config(env="four_room", n_iters=2, k_trajectories=2, outdir=str(tmp_path), goal_radius=2.0)
    O.main(cfg)                                    # goal radius covers the domain: reached on episode 1
    d = json.load(open(tmp_path / "four_room_paths.json"))["first_goal_reach"]
    assert (d["iter"], d["traj"], d["episodes"]) == (0, 1, 1)
    assert d["nodes"] >= 2 and d["transitions"] >= 1
    assert "first state within the goal radius" in open(tmp_path / "four_room_run.log").read()


def test_first_goal_reach_null_when_never(tmp_path):
    cfg = O.Config(env="four_room", n_iters=1, k_trajectories=1, outdir=str(tmp_path), goal_radius=1e-6)
    O.main(cfg)
    assert json.load(open(tmp_path / "four_room_paths.json"))["first_goal_reach"] is None
    assert "never reached" in open(tmp_path / "four_room_run.log").read()


# ---- gif: final frame held + saved separately ---------------------------------------- #
def test_final_frame_saved_and_held(tmp_path):
    from PIL import Image
    cfg = O.Config(env="four_room", n_iters=2, k_trajectories=1, outdir=str(tmp_path))
    O.main(cfg)
    png = tmp_path / "four_room_cost_final.png"
    assert png.stat().st_size > 5000
    gif = Image.open(tmp_path / "four_room_cost.gif")
    assert gif.n_frames == 2
    gif.seek(gif.n_frames - 1)
    assert gif.info["duration"] >= 3000
    assert Image.open(png).size[0] > 300

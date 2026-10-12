import itertools
import json

import numpy as np
import pytest

import collision_mbm as C
import robot_mbm as R
import scene_mbm as S
from conftest import SCENES, empty_scene, scene_json


@pytest.mark.parametrize("scene", SCENES)
def test_start_and_goals_valid_first_10(scene):
    for i in range(10):
        sc = S.load_scene(scene_json(scene, i))
        ck = C.PyBulletChecker(sc)
        assert ck.valid(sc.start), f"{scene} p{i} start"
        for k, g in enumerate(sc.goals):
            assert ck.valid(g), f"{scene} p{i} goal {k}"
        ck.close()


def _thin_wall_setup(tp0):
    """Two valid configs (joint1 rotated by 1 rad) with a thin wall at the EEF of the midpoint config,
    thin along the EEF's direction of travel."""
    q0 = tp0.start.copy()
    q1 = q0.copy()
    q1[0] += 1.0
    empty = empty_scene(tp0)
    rb = R.load_robot(R.new_client())
    mid = (q0 + q1) / 2
    p = rb.fk_eef(mid)
    phi = 0.5                                                    # joint1 angle at the midpoint
    yaw = phi + np.pi / 2                                        # box x-axis along the tangent
    quat = [0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]
    wall = {"name": "wall", "type": "box", "position": p.tolist(), "quaternion_xyzw": quat,
            "size": {"size": [0.004, 0.3, 0.3]}}
    return q0, q1, mid, empty_scene(tp0, [wall], start=q0, goals=[q1])


def test_thin_wall_caught_by_fine_res_missed_by_coarse_res(tp0):
    q0, q1, mid, sc = _thin_wall_setup(tp0)
    fine = C.PyBulletChecker(sc, motion_res=0.02)
    assert fine.valid(q0) and fine.valid(q1)
    assert not fine.valid(mid), "wall should be at the midpoint configuration"
    assert not fine.valid_motion(q0, q1)
    coarse = C.PyBulletChecker(sc, motion_res=5.0)               # n = 1: only the endpoint is checked
    assert coarse.valid_motion(q0, q1), "coarse resolution must miss the wall (proves resolution matters)"


@pytest.mark.parametrize("strategy", ["closest", "contact"])
def test_thin_wall_both_strategies(tp0, strategy):
    q0, q1, mid, sc = _thin_wall_setup(tp0)
    ck = C.PyBulletChecker(sc, motion_res=0.02, strategy=strategy)
    assert not ck.valid_motion(q0, q1)


def test_strategies_agree_on_random_configs(tp0):
    ck = {s: C.PyBulletChecker(tp0, strategy=s) for s in ("closest", "contact")}
    rng = np.random.default_rng(1)
    lo, hi = ck["closest"].q_low, ck["closest"].q_high
    n = 10000
    qs = rng.uniform(lo, hi, size=(n, 7))
    dis = sum(ck["closest"].valid(q) != ck["contact"].valid(q) for q in qs)
    print(f"strategy disagreement: {dis}/{n}")
    assert dis <= n * 0.001


def test_self_collision_pairs_exclude_adjacent():
    d = json.load(open(C.PAIRS_PATH))
    checked = {frozenset(p) for p in d["checked"]}
    chain = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5",
             "panda_link6", "panda_link7", "panda_hand"]
    for a, b in zip(chain[:-1], chain[1:]):
        assert frozenset((a, b)) not in checked, f"adjacent pair {a}-{b} is checked"
    for f in ("panda_leftfinger", "panda_rightfinger"):
        assert frozenset((f, "panda_hand")) not in checked
    assert frozenset(("panda_leftfinger", "panda_rightfinger")) not in checked
    assert len(checked) > 30
    dis = {frozenset(p) for p in d["always_in_contact_disabled"]}
    assert not (dis & checked)


def test_folded_arm_is_invalid_by_self_collision(tp0):
    ck = C.PyBulletChecker(empty_scene(tp0))
    fold = np.array([0, 1.7, 0, -3.1, 0, 3.7, 0])
    assert ck.in_limits(fold)
    assert not ck.valid(fold)
    assert ck.valid(tp0.start)


def test_limits_and_counters(tp0):
    ck = C.PyBulletChecker(tp0)
    q = tp0.start.copy()
    q[3] = 1.0                                                   # above joint-4's upper limit (0.0873)
    assert not ck.valid(q)
    assert ck.n_state_checks == 1
    assert ck.valid_motion(tp0.start, tp0.start + 0.035)
    assert ck.n_motion_checks == 1 and ck.n_state_checks == 1 + 2   # ceil(0.035 / 0.02) = 2 interpolated states


def test_margin_makes_checker_more_conservative(tp0):
    ck0 = C.PyBulletChecker(tp0, margin=0.0)
    ck1 = C.PyBulletChecker(tp0, margin=0.05, strategy="closest")
    rng = np.random.default_rng(3)
    qs = rng.uniform(ck0.q_low, ck0.q_high, size=(300, 7))
    v0 = np.array([ck0.valid(q) for q in qs])
    v1 = np.array([ck1.valid(q) for q in qs])
    assert not np.any(~v0 & v1) and v1.sum() < v0.sum()

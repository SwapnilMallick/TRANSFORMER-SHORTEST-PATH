"""Collision checking for the MBM port, behind a minimal backend-agnostic interface.

    class CollisionChecker(Protocol):
        def valid(q) -> bool
        def valid_motion(q_from, q_to) -> bool
        n_state_checks, n_motion_checks          # counters for logging

`PyBulletChecker` is the only implementation (a VAMP checker can be dropped in later behind the same
interface). It is the single source of truth for collisions: everything in the pipeline (sampler, merge
re-check, goal connection, RRT-Connect baseline) goes through it.

Semantics of the counters: `n_state_checks` counts every configuration evaluated (including the interpolated
ones inside a motion check), `n_motion_checks` counts `valid_motion` calls.

The checker moves ITS OWN robot (own DIRECT client); never rely on it leaving the robot anywhere.

Two validity strategies are implemented (`strategy=`):
  "closest"  per obstacle `getClosestPoints(robot, obstacle, margin)` plus one query per enabled self pair
  "contact"  `performCollisionDetection()` + `getContactPoints(robot)`, with obstacle/robot collision groups
             and `setCollisionFilterPair` switching off the disabled self pairs
Both report a collision when a contact point has distance <= margin. See `DEFAULT_STRATEGY` (chosen by
benchmarking + agreement over random configurations, see tests/test_checker.py).

Self-collision matrix (MoveIt-style): adjacent links (parent-child, collapsing the geometry-less link8 so
link7-hand is adjacent) and finger-finger / hand-finger pairs are never checked; additionally 2,000 random
configurations within limits are sampled and any pair in contact in EVERY sample is disabled. The resulting
pair list is cached in assets/panda/self_collision_pairs.json.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np
import pybullet as pb

import robot_mbm as R
from scene_mbm import Scene, add_obstacles

PAIRS_PATH = R.ASSET_DIR / "self_collision_pairs.json"
DEFAULT_STRATEGY = "contact"
N_MATRIX_SAMPLES = 2000
LIMIT_TOL = 1e-9

GROUP_ROBOT, GROUP_OBSTACLE = 1, 2


class CollisionChecker(Protocol):
    n_state_checks: int
    n_motion_checks: int

    def valid(self, q) -> bool: ...

    def valid_motion(self, q_from, q_to) -> bool: ...


def _has_geometry(client, body, link) -> bool:
    return len(client.getCollisionShapeData(body, link)) > 0


def robot_link_structure(client, robot: R.Robot):
    """(geometry links, geometry-parent map). link index -1 is the base."""
    n = client.getNumJoints(robot.body)
    geo = [l for l in range(-1, n) if _has_geometry(client, robot.body, l)]
    parent = {-1: None}
    for j in range(n):
        parent[j] = client.getJointInfo(robot.body, j)[16]
    geo_parent = {}
    for l in geo:
        p = parent[l]
        while p is not None and p not in geo:      # skip geometry-less links (link8)
            p = parent[p]
        geo_parent[l] = p
    return geo, geo_parent


def structural_excluded_pairs(client, robot: R.Robot):
    """Pairs never checked: adjacent (parent-child) links and finger-finger / hand-finger pairs."""
    geo, geo_parent = robot_link_structure(client, robot)
    names = robot.link_names
    excluded = set()
    for a, b in itertools.combinations(geo, 2):
        if geo_parent[a] == b or geo_parent[b] == a:
            excluded.add((a, b))
        elif names[a].endswith("finger") and names[b].endswith("finger"):
            excluded.add((a, b))
        elif "panda_hand" in (names[a], names[b]) and (names[a].endswith("finger") or names[b].endswith("finger")):
            excluded.add((a, b))
    return geo, excluded


def _random_q(rng, lo, hi):
    return rng.uniform(lo, hi)


def build_self_collision_pairs(n_samples: int = N_MATRIX_SAMPLES, seed: int = 0, margin: float = 0.0,
                               log=print) -> dict:
    """Sample the allowed-collision matrix and return the JSON-serialisable description."""
    client = R.new_client()
    robot = R.load_robot(client, self_collision=True)
    geo, excluded = structural_excluded_pairs(client, robot)
    cand = [(a, b) for a, b in itertools.combinations(geo, 2) if (a, b) not in excluded]
    rng = np.random.default_rng(seed)
    hits = {p: 0 for p in cand}
    for _ in range(n_samples):
        robot.set_q(_random_q(rng, robot.q_low, robot.q_high))
        for a, b in cand:
            pts = client.getClosestPoints(robot.body, robot.body, margin, linkIndexA=a, linkIndexB=b)
            if any(p[8] <= margin for p in pts):
                hits[(a, b)] += 1
    always = [p for p in cand if hits[p] == n_samples]
    checked = [p for p in cand if p not in always]
    names = robot.link_names
    out = {
        "n_samples": n_samples, "seed": seed, "margin": margin,
        "geometry_links": [names[l] for l in geo],
        "excluded_structural": [[names[a], names[b]] for a, b in sorted(excluded)],
        "always_in_contact_disabled": [[names[a], names[b]] for a, b in always],
        "checked": [[names[a], names[b]] for a, b in checked],
        "contact_fraction": {f"{names[a]}|{names[b]}": hits[(a, b)] / n_samples for a, b in cand},
    }
    client.disconnect()
    log(f"[self-collision] {len(geo)} geometry links, {len(excluded)} structural exclusions, "
        f"{len(always)} always-in-contact pairs disabled, {len(checked)} pairs checked")
    return out


def load_self_collision_pairs(path: Path = PAIRS_PATH, log=None) -> dict:
    if not Path(path).exists():
        data = build_self_collision_pairs(log=log or (lambda *_: None))
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
    with open(path) as f:
        return json.load(f)


class PyBulletChecker:
    """PyBullet collision checker (meshes vs primitives, mesh-hull vs mesh-hull self collision)."""

    def __init__(self, scene: Scene, motion_res: float = 0.02, margin: float = 0.0,
                 strategy: str = DEFAULT_STRATEGY):
        if strategy not in ("closest", "contact"):
            raise ValueError(f"strategy must be 'closest' or 'contact', got {strategy!r}")
        self.scene, self.motion_res, self.margin, self.strategy = scene, float(motion_res), float(margin), strategy
        self.n_state_checks = 0
        self.n_motion_checks = 0
        c = self.client = R.new_client()
        self.robot = R.load_robot(c, self_collision=True)
        self.obstacle_ids = add_obstacles(c, scene.obstacles, visual=False)
        self.q_low, self.q_high = self.robot.q_low, self.robot.q_high
        if margin > 0.02:                                   # contact points beyond the default breaking threshold
            c.setPhysicsEngineParameter(contactBreakingThreshold=margin * 1.5)

        # --- self-collision matrix -> explicit filter for EVERY link pair
        pairs = load_self_collision_pairs()
        by_name = {v: k for k, v in self.robot.link_names.items()}
        self.self_pairs: List[Tuple[int, int]] = [(by_name[a], by_name[b]) for a, b in pairs["checked"]]
        enabled = set(self.self_pairs) | {(b, a) for a, b in self.self_pairs}
        links = range(-1, c.getNumJoints(self.robot.body))
        for a, b in itertools.combinations(links, 2):
            c.setCollisionFilterPair(self.robot.body, self.robot.body, a, b, int((a, b) in enabled))
        for l in links:
            c.setCollisionFilterGroupMask(self.robot.body, l, GROUP_ROBOT, GROUP_ROBOT | GROUP_OBSTACLE)
        for ob in self.obstacle_ids:
            c.setCollisionFilterGroupMask(ob, -1, GROUP_OBSTACLE, GROUP_ROBOT)

        # the base is fixed: a base-obstacle overlap would make every configuration invalid
        self.robot.set_q(np.zeros(R.N_DOF))
        for ob in self.obstacle_ids:
            pts = c.getClosestPoints(self.robot.body, ob, margin, linkIndexA=-1)
            if any(p[8] <= margin for p in pts):
                raise ValueError(f"{scene.name}: the robot base overlaps obstacle {ob}")

    # ------------------------------------------------------------------ validity
    def in_limits(self, q) -> bool:
        return bool(np.all(q >= self.q_low - LIMIT_TOL) and np.all(q <= self.q_high + LIMIT_TOL))

    def valid(self, q) -> bool:
        self.n_state_checks += 1
        return self._valid(np.asarray(q, float))

    def _valid(self, q) -> bool:
        if not self.in_limits(q):
            return False
        c, body, m = self.client, self.robot.body, self.margin
        self.robot.set_q(q)
        if self.strategy == "contact":
            c.performCollisionDetection()
            for p in c.getContactPoints(bodyA=body):
                if p[8] <= m:
                    return False
            return True
        for ob in self.obstacle_ids:
            for p in c.getClosestPoints(body, ob, m):
                if p[8] <= m:
                    return False
        for a, b in self.self_pairs:
            for p in c.getClosestPoints(body, body, m, linkIndexA=a, linkIndexB=b):
                if p[8] <= m:
                    return False
        return True

    def valid_motion(self, q_from, q_to) -> bool:
        """Straight joint-space segment: n = ceil(max|dq| / motion_res) steps, every intermediate
        configuration including q_to is checked; early exit on the first collision. q_from is not re-checked
        (callers pass a configuration that is already valid)."""
        self.n_motion_checks += 1
        q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
        n = max(1, int(math.ceil(float(np.max(np.abs(q_to - q_from))) / self.motion_res)))
        for k in range(1, n + 1):
            self.n_state_checks += 1
            if not self._valid(q_from + (q_to - q_from) * (k / n)):
                return False
        return True

    def close(self):
        try:
            self.client.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------ utilities
    def benchmark(self, n: int = 1000, seed: int = 0, step: float = 0.15) -> dict:
        """State checks/sec on n random configurations and motion checks/sec on n random short motions
        (from a valid random configuration, |dq| <= step, direction uniform). Restores the counters."""
        rng = np.random.default_rng(seed)
        saved = (self.n_state_checks, self.n_motion_checks)
        qs = [rng.uniform(self.q_low, self.q_high) for _ in range(n)]
        t0 = time.perf_counter()
        ok = [self._valid(q) for q in qs]
        t_state = time.perf_counter() - t0
        starts = [q for q, v in zip(qs, ok) if v]
        motions = []
        for i in range(n):
            q0 = starts[i % len(starts)] if starts else self.scene.start
            d = rng.normal(size=R.N_DOF)
            motions.append((q0, q0 + d / np.linalg.norm(d) * step * rng.random() ** (1 / R.N_DOF)))
        m0 = self.n_state_checks
        t0 = time.perf_counter()
        mok = [self.valid_motion(a, b) for a, b in motions]
        t_motion = time.perf_counter() - t0
        per_motion = (self.n_state_checks - m0) / n
        out = {"state_checks_per_sec": n / t_state, "motion_checks_per_sec": n / t_motion,
               "valid_fraction_random_q": float(np.mean(ok)), "valid_fraction_motions": float(np.mean(mok)),
               "states_per_motion": per_motion, "strategy": self.strategy}
        self.n_state_checks, self.n_motion_checks = saved
        return out


if __name__ == "__main__":
    import argparse
    import scene_mbm as S
    ap = argparse.ArgumentParser(description="Benchmark + agreement of the two collision strategies.")
    ap.add_argument("--scene", default="table_pick")
    ap.add_argument("--problem", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000, help="configurations for the agreement check")
    a = ap.parse_args()
    sc = S.load_scene(S.scene_path(S.DEFAULT_SCENES_DIR, a.scene, a.problem))
    ck = {s: PyBulletChecker(sc, strategy=s) for s in ("closest", "contact")}
    for s, c in ck.items():
        print(s, {k: round(v, 3) if isinstance(v, float) else v for k, v in c.benchmark().items()})
    rng = np.random.default_rng(1)
    qs = [rng.uniform(ck["closest"].q_low, ck["closest"].q_high) for _ in range(a.n)]
    va = [ck["closest"].valid(q) for q in qs]
    vb = [ck["contact"].valid(q) for q in qs]
    dis = sum(x != y for x, y in zip(va, vb))
    print(f"agreement on {a.n} random configs: {a.n - dis}/{a.n}  (valid fraction {np.mean(va):.3f})")

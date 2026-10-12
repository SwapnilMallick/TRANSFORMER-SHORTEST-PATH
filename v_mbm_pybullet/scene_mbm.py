"""MBM problem JSON loading and PyBullet obstacle construction, shared by the checker and the renderer.

The JSON format is documented in ../mbm_vamp_render/FORMAT.md (written by tools/export_scenes.py).
Obstacles are true primitives (box / cylinder / sphere) with their stored world poses; note VAMP itself
collision-checks cylinders as capsules (cuboids in `box`), so PyBullet verdicts are less conservative there.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pybullet as pb

ROOT = Path(__file__).resolve().parent
DEFAULT_SCENES_DIR = ROOT / "scenes" / "panda"
OBSTACLE_RGBA = (0.55, 0.55, 0.58, 1.0)


@dataclass
class Scene:
    scene: str
    index: int                       # 0-based problem index
    joint_names: List[str]
    start: np.ndarray                # (7,)
    goals: List[np.ndarray]          # list of (7,); MBM has exactly one
    obstacles: List[dict]            # {name, type, position, quaternion_xyzw, size}
    extra: dict = field(default_factory=dict)
    path: Optional[str] = None

    @property
    def name(self) -> str:
        return f"{self.scene}_p{self.index}"

    def bbox(self):
        """Axis-aligned (lo, hi) of all obstacles' oriented shapes."""
        pts = []
        for o in self.obstacles:
            R = _quat_to_R(o["quaternion_xyzw"])
            c = np.asarray(o["position"], float)
            s = o["size"]
            if o["type"] == "box":
                ext = np.abs(R) @ (np.asarray(s["size"], float) / 2)
            elif o["type"] == "cylinder":
                ax = R[:, 2]
                ext = np.abs(ax) * s["length"] / 2 + s["radius"] * np.sqrt(np.clip(1 - ax ** 2, 0, None))
            else:
                ext = np.full(3, s["radius"])
            pts += [c - ext, c + ext]
        return np.min(pts, axis=0), np.max(pts, axis=0)


def _quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_scene(path) -> Scene:
    with open(path) as f:
        d = json.load(f)
    assert d["robot"] == "panda", f"only Panda scenes are supported, got {d['robot']!r}"
    return Scene(d["scene"], d["index"], d["joint_names"], np.array(d["start"], float),
                 [np.array(g, float) for g in d["goals"]], d["obstacles"], d.get("extra", {}), str(path))


def scene_path(scenes_dir, scene: str, problem: int) -> Path:
    return Path(scenes_dir) / scene / f"p{problem:03d}.json"


def add_obstacles(client, obstacles, rgba=OBSTACLE_RGBA, visual=True) -> List[int]:
    """Static bodies with the stored poses/sizes. Returns their body ids (same order as `obstacles`)."""
    ids = []
    for o in obstacles:
        s = o["size"]
        if o["type"] == "box":
            he = [x / 2 for x in s["size"]]
            col = client.createCollisionShape(pb.GEOM_BOX, halfExtents=he)
            vis = client.createVisualShape(pb.GEOM_BOX, halfExtents=he, rgbaColor=rgba) if visual else -1
        elif o["type"] == "cylinder":
            col = client.createCollisionShape(pb.GEOM_CYLINDER, radius=s["radius"], height=s["length"])
            vis = (client.createVisualShape(pb.GEOM_CYLINDER, radius=s["radius"], length=s["length"],
                                            rgbaColor=rgba) if visual else -1)
        elif o["type"] == "sphere":
            col = client.createCollisionShape(pb.GEOM_SPHERE, radius=s["radius"])
            vis = client.createVisualShape(pb.GEOM_SPHERE, radius=s["radius"], rgbaColor=rgba) if visual else -1
        else:
            raise ValueError(f"unknown obstacle type {o['type']!r}")
        ids.append(client.createMultiBody(0, col, vis, o["position"], o["quaternion_xyzw"]))
    return ids

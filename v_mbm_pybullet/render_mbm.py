"""PyBullet (TinyRenderer) rendering of a Panda in an MBM scene, in its OWN DIRECT client so collision checks
and rendering never interfere. Same URDF, obstacles and base pose as the checker; obstacles grey, robot with
its URDF visual materials, no physics stepping (`set_q` = resetJointState).

Interface used by image_encoder.ImageEncoder:
    renderer.set_q(q)                                  place the arm
    renderer.render(camera, height, width) -> uint8    HxWx3 of the CURRENT q
    renderer.camera_names() -> list[str]

Cameras are computed ONCE from the scene bounding box (identical for every q of a run):
    front  from +x looking at the scene centre (elevated 0.25)      side  from -y
    top    straight down (image-up = +x)                            iso   from (+x, -y, +z)
    wrist  attached to the hand: eye at the grasp target, looking along the hand's approach (+z) axis;
           its view matrix is recomputed from the hand link pose on EVERY render.
Framing box = obstacle bbox clamped to the arm's reach box, united with the robot base region and the start/goal
end-effector positions (a 2 m table would otherwise shrink the arm to a few pixels), padded like the render test.

Output is deterministic: identical q gives identical pixels (tested in tests/test_render.py).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pybullet as pb

import robot_mbm as R
from scene_mbm import Scene, add_obstacles

FIXED_VIEWS = {"front": ((1.0, 0.0, 0.25), (0, 0, 1)),
               "side": ((0.0, -1.0, 0.25), (0, 0, 1)),
               "top": ((0.0, 0.0, 1.0), (1, 0, 0)),
               "iso": ((0.8, -0.8, 0.65), (0, 0, 1))}
CAMERA_NAMES = ["front", "side", "top", "iso", "wrist"]
FOV = 45.0
WRIST_FOV = 75.0
NEAR, FAR = 0.02, 20.0
REACH_LO = np.array([-0.95, -0.95, -0.3])      # arm reach box (Panda reach ~0.855 m)
REACH_HI = np.array([0.95, 0.95, 1.3])
BASE_LO, BASE_HI = np.array([-0.25, -0.25, 0.0]), np.array([0.25, 0.25, 0.6])


def framing_box(scene: Scene, robot: R.Robot):
    lo, hi = [BASE_LO], [BASE_HI]
    if scene.obstacles:
        olo, ohi = scene.bbox()
        lo.append(np.maximum(olo, REACH_LO))
        hi.append(np.minimum(ohi, REACH_HI))
    for q in [scene.start] + list(scene.goals):
        p = robot.fk_eef(q)
        lo.append(p - 0.1)
        hi.append(p + 0.1)
    return np.min(lo, axis=0), np.max(hi, axis=0)


def _fit_view(lo, hi, direction, up, aspect, fov=FOV):
    """Look at the box centre from `direction`, at the smallest distance whose projection fits (92% of NDC)."""
    ctr = (lo + hi) / 2
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = 1.0 / np.tan(np.radians(fov) / 2)
    x_axis = np.cross(up, d)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(d, x_axis)
    rel = pts - ctr

    def fits(dist):
        depth = dist - rel @ d
        if np.any(depth <= 0.1):
            return False
        return max(np.abs(f / aspect * (rel @ x_axis) / depth).max(),
                   np.abs(f * (rel @ y_axis) / depth).max()) <= 0.92

    a, b = 0.2, 20.0
    for _ in range(40):
        mid = (a + b) / 2
        a, b = (a, mid) if fits(mid) else (mid, b)
    return ctr + d * b, ctr


class PyBulletRenderer:
    def __init__(self, scene: Scene, cameras: List[str] = None):
        self.scene = scene
        self.client = R.new_client()
        self.robot = R.load_robot(self.client)                 # URDF visual materials, no recolouring
        self.obstacle_ids = add_obstacles(self.client, scene.obstacles)
        self.lo, self.hi = framing_box(scene, self.robot)
        self.robot.set_q(scene.start)
        self._views: Dict[str, list] = {}
        for name, (direction, up) in FIXED_VIEWS.items():
            eye, ctr = _fit_view(self.lo, self.hi, direction, np.asarray(up, float), aspect=1.0)
            self._views[name] = self.client.computeViewMatrix(eye.tolist(), ctr.tolist(), list(up))
        self._proj: Dict[tuple, list] = {}
        self.n_renders = 0

    # ------------------------------------------------------------------ interface
    def camera_names(self) -> List[str]:
        return list(CAMERA_NAMES)

    def set_q(self, q) -> None:
        self.robot.set_q(q)

    def _wrist_view(self):
        pos, Rm = self.robot.link_pose(self.robot.hand_link)
        eye = pos + Rm @ np.array([0.0, 0.0, 0.105])          # grasp target (between the fingertips)
        target = eye + Rm @ np.array([0.0, 0.0, 1.0])
        up = Rm @ np.array([1.0, 0.0, 0.0])
        return self.client.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())

    def render(self, camera: str, height: int, width: int) -> np.ndarray:
        if camera not in CAMERA_NAMES:
            raise ValueError(f"unknown camera {camera!r}; available: {CAMERA_NAMES}")
        view = self._wrist_view() if camera == "wrist" else self._views[camera]
        fov = WRIST_FOV if camera == "wrist" else FOV
        key = (fov, height, width)
        if key not in self._proj:
            self._proj[key] = self.client.computeProjectionMatrixFOV(fov, width / height, NEAR, FAR)
        _, _, rgb, _, _ = self.client.getCameraImage(width, height, view, self._proj[key],
                                                     renderer=pb.ER_TINY_RENDERER,
                                                     flags=pb.ER_NO_SEGMENTATION_MASK, shadow=0)
        self.n_renders += 1
        return np.reshape(np.asarray(rgb, dtype=np.uint8), (height, width, 4))[:, :, :3].copy()

    def close(self):
        try:
            self.client.disconnect()
        except Exception:
            pass

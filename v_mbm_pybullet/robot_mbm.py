"""Panda model for the MBM port: URDF loading, joint order/limits, and forward kinematics.

Uses the URDF the MBM problems were defined with (assets/README.md). The 7 arm joints are looked up BY NAME
so their order matches the JSON `q` vectors; the fingers are fixed joints in this URDF (nothing to set).
No IK anywhere: FK only, for end-effector (EEF) positions in plots and the frame checks.
"""
from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pybullet as pb
from pybullet_utils.bullet_client import BulletClient

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "panda"
URDF_PATH = ASSET_DIR / "panda.urdf"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
EEF_LINK = "panda_grasptarget"          # frame VAMP's eefk() uses
HAND_LINK = "panda_hand"                # its +z axis is the approach direction (wrist camera)
N_DOF = 7


@contextlib.contextmanager
def quiet():
    """Silence C-level stdout/stderr (PyBullet's mesh loader is chatty)."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in saved + [null]:
            os.close(fd)


def new_client() -> BulletClient:
    return BulletClient(connection_mode=pb.DIRECT)


@dataclass
class Robot:
    client: BulletClient
    body: int
    joints: List[int]                   # pybullet joint index of each arm joint, in ARM_JOINTS order
    q_low: np.ndarray
    q_high: np.ndarray
    eef_link: int
    hand_link: int
    link_names: dict                    # link index -> name (base link is -1 -> 'panda_link0')

    def set_q(self, q) -> None:
        for j, v in zip(self.joints, q):
            self.client.resetJointState(self.body, j, float(v), targetVelocity=0.0)

    def fk_eef(self, q) -> np.ndarray:
        """World XYZ of the grasp-target frame at configuration q (moves this robot to q)."""
        self.set_q(q)
        st = self.client.getLinkState(self.body, self.eef_link, computeForwardKinematics=True)
        return np.array(st[4], float)   # link frame position (URDF link origin, not the CoM)

    def link_pose(self, link: int):
        """(position, 3x3 rotation) of a link's URDF frame at the CURRENT configuration."""
        st = self.client.getLinkState(self.body, link, computeForwardKinematics=True)
        R = np.array(self.client.getMatrixFromQuaternion(st[5])).reshape(3, 3)
        return np.array(st[4], float), R


def load_robot(client: BulletClient, rgb=None, self_collision: bool = False) -> Robot:
    """Load the Panda fixed at the world origin (the base pose the MBM problems assume)."""
    flags = pb.URDF_MAINTAIN_LINK_ORDER
    if self_collision:
        flags |= pb.URDF_USE_SELF_COLLISION
    client.setAdditionalSearchPath(str(ASSET_DIR))
    with quiet():
        body = client.loadURDF(str(URDF_PATH), basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
                               useFixedBase=True, flags=flags)
    by_joint = {}
    link_names = {-1: client.getBodyInfo(body)[0].decode()}
    for j in range(client.getNumJoints(body)):
        info = client.getJointInfo(body, j)
        by_joint[info[1].decode()] = j
        link_names[j] = info[12].decode()
    base = client.getBodyInfo(body)[0].decode()
    link_names[-1] = base
    joints = [by_joint[n] for n in ARM_JOINTS]
    lo = np.array([client.getJointInfo(body, j)[8] for j in joints])
    hi = np.array([client.getJointInfo(body, j)[9] for j in joints])
    if rgb is not None:
        for link in range(-1, client.getNumJoints(body)):
            client.changeVisualShape(body, link, rgbaColor=(*rgb, 1.0))
    by_link = {v: k for k, v in link_names.items()}
    return Robot(client, body, joints, lo, hi, by_link[EEF_LINK], by_link[HAND_LINK], link_names)

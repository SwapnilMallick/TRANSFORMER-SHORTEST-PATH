"""Robot model / frame checks: joint order + limits match the JSON, FK agrees with VAMP's recorded EEF positions."""
import numpy as np
import pytest

import robot_mbm as R
import scene_mbm as S
from conftest import SCENES, scene_json


@pytest.fixture(scope="module")
def robot():
    return R.load_robot(R.new_client())


def test_joint_order_and_limits_match_json(robot, tp0):
    assert tp0.joint_names == R.ARM_JOINTS
    lo, hi = tp0.extra["vamp_joint_limits"]
    np.testing.assert_allclose(robot.q_low, lo, atol=1e-4)     # VAMP stores float32
    np.testing.assert_allclose(robot.q_high, hi, atol=1e-4)
    # PyBullet joint indices follow the arm order
    names = [robot.client.getJointInfo(robot.body, j)[1].decode() for j in robot.joints]
    assert names == R.ARM_JOINTS


def test_fingers_are_fixed_and_eef_is_grasptarget(robot):
    assert robot.link_names[robot.eef_link] == "panda_grasptarget"
    for j in range(robot.client.getNumJoints(robot.body)):
        info = robot.client.getJointInfo(robot.body, j)
        if "finger" in info[1].decode():
            assert info[2] == 4                                   # JOINT_FIXED


@pytest.mark.parametrize("scene", SCENES)
def test_fk_matches_vamp_eef_all_problems(robot, scene):
    worst = 0.0
    for i in range(100):
        sc = S.load_scene(scene_json(scene, i))
        np.testing.assert_allclose(robot.fk_eef(sc.start), sc.extra["vamp_eef_start"], atol=1e-4)
        for g, ref in zip(sc.goals, sc.extra["vamp_eef_goals"]):
            err = np.abs(robot.fk_eef(g) - ref).max()
            worst = max(worst, err)
            assert err < 1e-4
    print(f"{scene}: worst goal FK error vs VAMP = {worst:.2e} m")


def test_base_at_origin(robot):
    # zero configuration: arm straight up, flange offset 0.088 in x, tool pointing down. Closed form from the
    # Panda DH parameters: z = 0.333 + 0.316 + 0.384 - 0.107 - 0.105 (grasp target) = 0.821 above the base frame
    np.testing.assert_allclose(robot.fk_eef(np.zeros(7)), [0.088, 0.0, 0.821], atol=1e-6)

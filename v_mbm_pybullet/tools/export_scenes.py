"""Export MBM Panda problems to the neutral JSON format (see ../../mbm_vamp_render/FORMAT.md).

The render test ships `load_mbm.export_scene_json` but never wrote the files, so this script does. It
needs VAMP (for joint names, limits and reference EEF positions), so run it with the render test's venv:

    ../mbm_vamp_render/.venv/bin/python tools/export_scenes.py [--scenes ...] [--out scenes/panda]

Beyond the FORMAT.md fields, `extra` gets three VAMP-computed reference entries used by the frame tests:
    vamp_eef_start, vamp_eef_goals   XYZ of VAMP's `end_effector()` frame (panda_grasptarget) at start/goals
    vamp_joint_limits                [lower, upper] in VAMP joint order
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RENDER = HERE.parents[1] / "mbm_vamp_render"
sys.path.insert(0, str(RENDER))
import load_mbm as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=L.SCENES)
    ap.add_argument("--out", default=str(HERE.parent / "scenes" / "panda"))
    a = ap.parse_args()
    import vamp
    lo, hi = L.joint_limits("panda")
    n = 0
    for scene in a.scenes:
        for i in range(L.n_problems("panda", scene)):
            p = L.load_problem("panda", scene, i)
            d = L.problem_to_dict(p)
            eef = lambda q: np.asarray(vamp.panda.eefk(q))[:3, 3].astype(float).tolist()
            d["extra"] = dict(d["extra"], vamp_eef_start=eef(p.start),
                              vamp_eef_goals=[eef(g) for g in p.goals],
                              vamp_joint_limits=[lo.tolist(), hi.tolist()])
            path = Path(a.out) / scene / f"p{i:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(d, f, indent=1)
            n += 1
        print(f"{scene}: exported {L.n_problems('panda', scene)}", flush=True)
    print(f"{n} problems -> {a.out}")


if __name__ == "__main__":
    main()

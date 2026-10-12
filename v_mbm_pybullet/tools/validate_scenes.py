"""Start/goal validity of every exported problem under the PyBullet checker (also reports the straight-line
start->goal motion, which shows how many problems are trivial in joint space).

    .venv/bin/python tools/validate_scenes.py [--scenes ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collision_mbm as C  # noqa: E402
import scene_mbm as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", nargs="+", default=["table_pick", "table_under_pick", "box", "bookshelf_small",
                                                "bookshelf_tall", "bookshelf_thin", "cage"])
a = ap.parse_args()
for scene in a.scenes:
    bad, straight = [], 0
    for i in range(100):
        sc = S.load_scene(S.scene_path(S.DEFAULT_SCENES_DIR, scene, i))
        ck = C.PyBulletChecker(sc)
        ok_s, ok_g = ck.valid(sc.start), all(ck.valid(g) for g in sc.goals)
        if not (ok_s and ok_g):
            bad.append((i, "start" if not ok_s else "goal"))
        elif ck.valid_motion(sc.start, sc.goals[0]):
            straight += 1
        ck.close()
    print(f"{scene:18s} invalid under PyBullet: {len(bad)}/100 {bad}  |  straight start->goal free: {straight}/100",
          flush=True)

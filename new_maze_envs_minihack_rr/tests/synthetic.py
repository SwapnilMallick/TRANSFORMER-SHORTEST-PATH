"""Synthetic layouts for tests (own recursive backtracker; no NLE)."""
import numpy as np

from layout_geometry import grid_to_rows


def backtracker_grid(nr, nc, seed=0):
    """nr x nc logical maze on a lattice grid (2*nr+1) x (2*nc+1); nodes at odd coords."""
    rng = np.random.default_rng(seed)
    free = np.zeros((2 * nr + 1, 2 * nc + 1), bool)
    seen = {(0, 0)}
    free[1, 1] = True
    stack = [(0, 0)]
    while stack:
        i, j = stack[-1]
        nb = [(i + a, j + b) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= i + a < nr and 0 <= j + b < nc and (i + a, j + b) not in seen]
        if not nb:
            stack.pop()
            continue
        u, v = nb[rng.integers(len(nb))]
        seen.add((u, v))
        free[2 * u + 1, 2 * v + 1] = True
        free[i + u + 1, j + v + 1] = True       # connector between (i,j) and (u,v)
        stack.append((u, v))
    return free


def corridor_grid():
    """Two rooms joined by a 1-wide corridor with a bend, plus a parallel spur."""
    g = np.zeros((14, 30), bool)
    g[2:6, 2:8] = True          # room A
    g[8:12, 20:27] = True       # room B
    g[3, 8:24] = True           # corridor east
    g[3:9, 23] = True           # corridor south into room B
    g[5, 8:16] = True           # parallel spur (2 rows below, wall of 1 between)
    return g


def raw_dict(free, name, family, start, goal, size="syn", seed=0):
    return {"name": name, "family": family, "size": size, "seed": seed, "env_id": "synthetic",
            "grid": grid_to_rows(free), "start": list(start), "goal": list(goal),
            "reveal_method": "synthetic", "explorer_steps": 0, "versions": {}, "timestamp": "n/a"}

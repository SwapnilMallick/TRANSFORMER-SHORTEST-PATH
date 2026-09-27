"""Derive a two-room MazeWalk layout: a maze plus its left-right mirror image, joined by a one-cell bridge.

Reads layouts/raw/<base>.json (a MazeWalk lattice grid), writes layouts/raw/<name>.json in the same
raw format, so layout_geometry.py / render_layouts.py / online_cost_transformer.py treat it like any
other layout.  No NLE needed.

Raw lattice layout (odd rows/cols are logical cells, even ones are walls/connectors):
    room A = base grid (cols 0..W-1) | bridge cell column (col W+1) | room B = mirror(base) (cols W+2..)
The bridge is one logical cell in row `bridge_row`, linked by connector cells to A's right edge and to
B's left edge (which is A's right edge mirrored), so it is exactly one passage wide.
Start and goal are set in logical cells in mazewalk_endpoints.json (the raw start/goal below are defaults).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import layout_geometry as G

HERE = Path(__file__).resolve().parent


def mirror_two_room(free: np.ndarray, bridge_row: int):
    """free: base lattice grid (2*nr+1 x 2*nc+1). Returns the two-room grid and the bridge's raw row."""
    n, w = free.shape
    nr = (n - 1) // 2
    if not 0 <= bridge_row < nr:
        raise ValueError(f"bridge_row must be in 0..{nr - 1}")
    out = np.zeros((n, 2 * w + 1), bool)
    out[:, :w] = free
    out[:, w + 1:] = free[:, ::-1]                    # mirror image, shifted right of the bridge column
    r = 2 * bridge_row + 1
    if not (free[r, w - 2] and out[r, w + 2]):
        raise ValueError(f"bridge row {bridge_row}: the edge cell of the maze is not free")
    out[r, w - 1] = out[r, w] = out[r, w + 1] = True  # connector, bridge cell, connector
    return out, r


def logical_path_len(free, start, goal):
    adj, idx = G.grid_graph(free)
    d = G.bfs(adj, int(idx[tuple(start)]))[int(idx[tuple(goal)])]
    return int(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="mazewalk_9x9_s0")
    ap.add_argument("--start", nargs=2, type=int, default=[0, 0], metavar=("ROW", "COL"),
                    help="start logical cell in room A (default 0 0)")
    ap.add_argument("--goal", nargs=2, type=int, default=[2, 0], metavar=("ROW", "COL"),
                    help="goal as a logical cell of the BASE maze; it is placed at its mirror image in room B")
    ap.add_argument("--bridge-row", type=int, default=None,
                    help="logical row of the bridge (default: the row giving the longest start->goal path)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--raw-dir", default=str(HERE / "layouts" / "raw"))
    a = ap.parse_args(argv)
    raw_dir = Path(a.raw_dir)
    base = json.loads((raw_dir / f"{a.base}.json").read_text())
    free = G.parse_grid(base["grid"])
    n, w = free.shape
    nr, nc = (n - 1) // 2, (w - 1) // 2
    start = (2 * a.start[0] + 1, 2 * a.start[1] + 1)
    gb = (2 * a.goal[0] + 1, 2 * a.goal[1] + 1)
    goal = (gb[0], 2 * w - gb[1])                       # mirror image of the base goal, in room B
    goal_logical = [a.goal[0], 2 * nc - a.goal[1]]
    rows = [a.bridge_row] if a.bridge_row is not None else range(nr)
    best = None
    for br in rows:
        try:
            grid, r = mirror_two_room(free, br)
        except ValueError:
            continue
        d = logical_path_len(grid, start, goal)
        if d >= 0 and (best is None or d > best[0]):
            best = (d, br, grid)
    if best is None:
        raise SystemExit("no valid bridge row (start and goal not connected)")
    d, br, grid = best
    name = a.name or f"{a.base.rsplit('_s', 1)[0]}_mirror2_s{base['seed']}"
    rec = dict(base, name=name, size=f"{base['size']}_mirror2", grid=G.grid_to_rows(grid),
               start=list(start), goal=list(goal), env_id=f"derived:{base['env_id']} (mirrored two-room)",
               reveal_method="derived_mirror", derived_from=a.base, bridge_row=br,
               endpoints="default_corner_to_mirrored_goal", explorer_steps=0)
    rec.pop("native_start", None)
    rec.pop("native_goal", None)
    (raw_dir / f"{name}.json").write_text(json.dumps(rec))
    print(f"{name}: grid {grid.shape[0]}x{grid.shape[1]}, logical {nr}x{2 * nc + 1}, bridge row {br}, "
          f"start {a.start} -> goal {goal_logical} ({d // 2} logical steps)")
    print(f"config entry: \"{name}\": {{\"start\": {a.start}, \"goal\": {goal_logical}}}")


if __name__ == "__main__":
    main()

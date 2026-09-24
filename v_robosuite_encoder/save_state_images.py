"""Re-render the 224x224 camera images the encoder saw for chosen graph nodes.

Reads robosuite_node_embeddings.npz from a finished run (it holds every node's
EEF position and joint configuration), puts the arm in each node's joint
configuration with set_q(), and renders the same cameras at the same size the
encoder uses. Nothing is re-run and the experiment's files are not modified.

Outputs (under --out, default <run>/state_images):
  node_<id>_<camera>.png   exact 224x224 renders, one file per node and view
  overview.png             grid: one row per node, one column per view

Usage:
  python3 save_state_images.py --run robosuite_encoder_results/alpha_0.05
  python3 save_state_images.py --run ... --nodes 0 10 500 --check-embedding
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

from image_encoder import EncoderConfig, ImageEncoder
from task import RobosuiteTask


def pick_nodes(pos, n, nodes, task):
    """Explicit ids if given; else node 0, the node closest to the cube, and
    n-2 nodes spread evenly over the ids (a spread over exploration order)."""
    if nodes:
        return list(nodes)
    cube = task.ik.cube_pos()
    closest = int(np.argmin(np.linalg.norm(pos - cube, axis=1)))
    picks = [0, closest]
    for i in np.linspace(0, len(pos) - 1, max(n - 2, 0)).astype(int):
        if int(i) not in picks:
            picks.append(int(i))
    return picks[:max(n, 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="finished run folder")
    p.add_argument("--seed", type=int, default=0, help="seed the run used")
    p.add_argument("--nodes", type=int, nargs="*", help="explicit node ids")
    p.add_argument("--n", type=int, default=6, help="how many nodes if --nodes not given")
    p.add_argument("--cameras", nargs="+", default=["agentview", "sideview", "frontview"])
    p.add_argument("--image-size", type=int, default=224, dest="image_size")
    p.add_argument("--out", default=None)
    p.add_argument("--check-embedding", action="store_true", dest="check",
                   help="also embed the re-rendered images with DINOv2 + the run's "
                        "saved stats and compare to the stored embeddings")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    d = np.load(os.path.join(a.run, "robosuite_node_embeddings.npz"))
    pos, q = d["pos"], d["q"]
    out = a.out or os.path.join(a.run, "state_images")
    os.makedirs(out, exist_ok=True)

    task = RobosuiteTask(seed=a.seed, enable_render=True)
    ids = pick_nodes(pos, a.n, a.nodes, task)
    S = a.image_size

    rows = []
    frames = {}
    for nid in ids:
        task.ik.set_q(q[nid])
        eef = task.ik.eef_pos()
        assert np.allclose(eef, pos[nid], atol=2e-3), (
            f"node {nid}: re-set EEF {eef} != stored {pos[nid]}")
        imgs = []
        for cam in a.cameras:
            img = np.ascontiguousarray(task.ik.render(cam, S, S))
            Image.fromarray(img).save(os.path.join(out, f"node_{nid:05d}_{cam}.png"))
            imgs.append(img)
        frames[nid] = np.stack(imgs)
        rows.append(np.concatenate(imgs, axis=1))

    # overview grid with a label strip per row
    label_h = 18
    W = rows[0].shape[1]
    head_h = 18
    sheet = Image.new("RGB", (W, head_h + len(rows) * (S + label_h)), "white")
    dr = ImageDraw.Draw(sheet)
    for r, (nid, row) in enumerate(zip(ids, rows)):
        y = head_h + r * (S + label_h)
        dr.text((4, y + 3), f"node {nid}   eef={np.round(pos[nid], 3).tolist()}", fill="black")
        sheet.paste(Image.fromarray(row), (0, y + label_h))
    for c, cam in enumerate(a.cameras):
        dr.text((c * S + 4, 3), cam, fill="black")
    sheet.save(os.path.join(out, "overview.png"))
    print(f"saved {len(ids) * len(a.cameras)} images + overview.png -> {out}")
    print("nodes:", ids)

    if a.check:
        enc = ImageEncoder(EncoderConfig(cameras=tuple(a.cameras), image_size=S,
                                         device=a.device))
        enc.load_stats(os.path.join(a.run, "encoder_stats.npz"))
        worst = 0.0
        for nid in ids:
            task.ik.set_q(q[nid])
            worst = max(worst, float(np.abs(enc.embed(task.ik) - d["emb"][nid]).max()))
        print(f"max |re-embedded - stored| over {len(ids)} nodes: {worst:.2e}")


if __name__ == "__main__":
    main()

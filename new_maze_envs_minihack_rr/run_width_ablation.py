"""Width ablation: step_to_width in {1.0, 0.5, 0.25} x seeds on chosen layouts.

Runs online_cost_transformer.py once per (layout, step_to_width, seed) in parallel worker
processes (a few torch threads each), then writes <out>/summary.md and summary.csv.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYOUTS = ["layouts/processed/mazewalk_15x15_s0_thinwall.json",
           "layouts/processed/corridor_r5_12x5_s0_dilate_k4m1.json"]


def run_dir(out, layout, w, seed):
    return Path(out) / Path(layout).stem / f"w{w:g}" / f"seed{seed}"


def launch(args):
    layout, w, seed, out, threads, extra = args
    d = run_dir(out, layout, w, seed)
    d.mkdir(parents=True, exist_ok=True)
    if list(d.glob("*_metrics.csv")):
        return d                                             # already finished
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    cmd = [sys.executable, "-W", "ignore", str(HERE / "online_cost_transformer.py"), "--layout", str(HERE / layout),
           "--step-to-width", str(w), "--seed", str(seed), "--out", str(d)] + extra
    with open(d / "stdout.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    return d


def summarize(out, layouts, widths, seeds):
    rows = []
    for layout in layouts:
        for w in widths:
            for seed in seeds:
                d = run_dir(out, layout, w, seed)
                mets = list(d.glob("*_metrics.csv"))
                paths = list(d.glob("*_paths.json"))
                if not mets:
                    rows.append(dict(layout=Path(layout).stem, w=w, seed=seed, done=False))
                    continue
                it = list(csv.DictReader(open(mets[0])))
                succ = [int(r["succ"]) for r in it]
                first = next((i for i, s in enumerate(succ) if s > 0), None)
                ktraj = 8                                                      # default k_trajectories
                rows.append(dict(
                    layout=Path(layout).stem, w=w, seed=seed, done=True,
                    sample_fails=sum(int(r["sample_fails"]) for r in it),
                    nodes_first_reach=int(it[first]["nodes"]) if first is not None else None,
                    first_reach_iter=first,
                    success_rate=sum(succ) / (ktraj * len(it)),
                    final_goal_reached=json.load(open(paths[0]))["goal_reached_by_collection"],
                    final_nodes=int(it[-1]["nodes"]),
                    seconds=sum(float(r["iter_sec"]) for r in it)))
    keys = ["layout", "w", "seed", "done", "sample_fails", "nodes_first_reach", "first_reach_iter",
            "success_rate", "final_goal_reached", "final_nodes", "seconds"]
    with open(Path(out) / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "_ablation"))
    ap.add_argument("--layouts", nargs="*", default=LAYOUTS)
    ap.add_argument("--widths", nargs="*", type=float, default=[1.0, 0.5, 0.25])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra args for online_cost_transformer.py, after --")
    a = ap.parse_args()
    jobs = [(l, w, s, a.out, a.threads, a.extra) for l in a.layouts for w in a.widths for s in a.seeds]
    if not a.summary_only:
        with ThreadPoolExecutor(a.workers) as ex:
            for d in ex.map(launch, jobs):
                print("finished", d, flush=True)
    rows = summarize(a.out, a.layouts, a.widths, a.seeds)
    print(f"summary -> {a.out}/summary.csv ({sum(r['done'] for r in rows)}/{len(rows)} runs done)")


if __name__ == "__main__":
    main()

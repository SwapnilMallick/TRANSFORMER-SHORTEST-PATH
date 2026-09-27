"""Diagnose why the transformer's greedy path is worse than Dijkstra even at ~0 training loss.

Re-runs one layout with the given settings (same as the CLI), then on the trained model compares
  (a) the history-free query used by transformer_path (token = [node_pos, 0, 0], length-1 sequence),
  (b) the training-style query (each node's real token history from the start along the Dijkstra tree),
  (c) the true Dijkstra costs (sanity check of the graph and the greedy walk itself),
by cost error, agreement of neighbour ordering, and the cost of the greedy path each one produces.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

import online_cost_transformer as O


def batched_last_pred(model, seqs, cfg, bs=64):
    """Model prediction at the last token of each (tokens) sequence."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            chunk = [(np.array(t), np.zeros(len(t))) for t in seqs[i:i + bs]]
            x, _, pad = O.collate(chunk, cfg)
            c = model(x, O.causal_mask(x.shape[1], cfg.device), pad)
            for b, (t, _) in enumerate(chunk):
                out.append(float(c[b, len(t) - 1]))
    return np.array(out)


def report(name, chat, gn, graph, cfg, goal_node, g, log):
    nbr = graph.undirected_neighbors()
    fin = np.isfinite(gn)
    mae = float(np.abs(chat[fin] - gn[fin]).mean())
    agree = tot = close_agree = close_tot = 0
    for u in range(len(gn)):
        for v in nbr[u]:
            if v <= u or not (fin[u] and fin[v]) or abs(gn[u] - gn[v]) < 1e-9:
                continue
            ok = (chat[u] - chat[v]) * (gn[u] - gn[v]) > 0
            tot += 1
            agree += ok
            if abs(gn[u] - gn[v]) < 0.01:
                close_tot += 1
                close_agree += ok
    seq, reached = O.transformer_path(model_holder["m"], graph, cfg, goal_node, chat=chat)
    cost = O.path_length(graph, seq)
    log(f"{name:34s} MAE={mae:.4f} (raw {mae * cfg.diagonal:.3f})  neighbour-order agreement "
        f"{agree / max(tot, 1):.1%} (pairs closer than 0.01: {close_agree / max(close_tot, 1):.1%}, n={close_tot})  "
        f"greedy path: {len(seq)} nodes, cost {cost:.3f}, ratio {cost / g[goal_node]:.2f}, reached={reached}")
    return dict(name=name, mae=mae, order_agree=agree / max(tot, 1), close_agree=close_agree / max(close_tot, 1),
                path_nodes=len(seq), path_cost=cost, ratio=cost / g[goal_node], reached=bool(reached))


model_holder = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = O.Config(env="layout", layout_file=a.layout, alpha=a.alpha, seed=a.seed, n_iters=a.iters,
                   k_trajectories=a.k, outdir=a.out)
    ov = []
    if a.max_steps is not None:
        cfg.max_steps = a.max_steps
        ov.append("max_steps")
    cfg.overrides = tuple(ov)
    model, graph, _ = O.main(cfg)
    model_holder["m"] = model
    g, prev = graph.dijkstra(0, return_prev=True)
    g = np.array(g)
    gn = g / cfg.diagonal
    goal_node = graph.nearest(np.array(cfg.goal))[0]
    torch.save({"state": model.state_dict(), "pos": np.array(graph.pos), "g": g, "prev": prev,
                "adj": {u: dict(v) for u, v in graph.adj.items()}, "cfg": cfg.__dict__}, os.path.join(a.out, "diag_model.pt"))
    lines = []

    def log(m):
        print(m)
        lines.append(m)
    log(f"\n=== path-field diagnosis: {len(g)} nodes, goal node {goal_node} (g={g[goal_node]:.3f}) ===")
    tok_hist, _ = O.build_all_histories(graph, list(g), prev)
    chat0 = O.predict_cost_field(model, graph, cfg)                       # what transformer_path uses
    chat_h = np.zeros(len(g))
    idx = [i for i in range(len(g)) if i != 0 and tok_hist[i] and math.isfinite(g[i])]
    chat_h[idx] = batched_last_pred(model, [tok_hist[i] for i in idx], cfg)
    res = [report("history-free query (as used)", chat0, gn, graph, cfg, goal_node, g, log),
           report("history-conditioned (as trained)", chat_h, gn, graph, cfg, goal_node, g, log),
           report("true Dijkstra costs (sanity)", gn.copy(), gn, graph, cfg, goal_node, g, log)]
    with open(os.path.join(a.out, "diagnosis.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

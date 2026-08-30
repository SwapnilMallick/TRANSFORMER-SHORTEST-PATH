"""
Online Transformer Training Loop for cost-to-come estimation with a
cost-based veto in a *continuous* four-room maze.

Implements the approach in "Approach Discussed On August 6"
(S. Mallick, UBC #68686534, 2026).

MDP
---
    state   s = [x, y]  in R^2                 (2-D position)
    action  a = [ax, ay] in R^2                (intended displacement)
    trans.  s' = s + a + eps,  eps ~ N(0, Sigma)   (Gaussian process noise)
    reward  1 if ||s - goal|| <= goal_radius else 0

Algorithm 1 (Online Transformer Training Loop)
----------------------------------------------
    D <- {}
    for n iterations:
        collect k trajectories with the current transformer
            * novelty-radius merging of proposed next states
            * an action reaching an *already-visited* state is VETOED unless
                  c_{s'}  <  c_hat_{s'} + (mu_{s'} - alpha)
              i.e. the current path is (near-)competitive with the best known
              cost-to-come for that state.  Rejected actions get a fixed
              number of retries.
        D <- D U (new transitions)
        build a graph over merged states; Dijkstra from start -> cost-to-come
            (these Dijkstra costs are the training labels)
        train the transformer on the labelled buffer

Transformer  T_cost(c_hat | h_t, a_t)
--------------------------------------
    * state/action fused 4-d tokens -> Linear(4, 64)
    * learned positional embeddings (added, not concatenated)
    * 2 x nn.TransformerEncoderLayer, d_model=64, nhead=2, GELU, dropout=0.1
    * causal mask (upper-triangular -inf above the diagonal)
    * single linear head: cost c_hat, trained with plain Huber loss

Uncertainty (v3): epistemic only, via MC dropout
-------------------------------------------------
    At veto time, T stochastic forward passes (dropout left ON) are run over
    the same candidate sequence; their spread gives an epistemic-uncertainty
    estimate for that state:
        c_hat, std_raw   = mean / std of the T passes' last-token predictions
    std_raw (not variance) is used so the estimate stays in cost units.
    std_raw is noisy at small T, so it is blended into a per-node running
    estimate with an EMA (mu_ema[node] <- beta*mu_ema[node] + (1-beta)*std_raw),
    keyed by destination node id -- nodes visited often get a stable estimate,
    rarely-visited nodes track their few observations closely.
    veto rule (v3, fixed alpha):
        cand_cost < c_hat + (mu_ema[node] - alpha)
    v4 will replace the fixed alpha with a quantile tracked online over the
    mu_ema population, so the threshold self-calibrates to whatever scale the
    MC-dropout estimates land at and adapts as it shrinks over training.
"""

from __future__ import annotations

import math
import heapq
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- maze / task ---
    env: str = "four_room"                      # see ENVS registry below
    L: float = 1.0                              # square domain [0, L]^2
    start: Tuple[float, float] = (0.10, 0.10)   # bottom-left room
    goal: Tuple[float, float] = (0.90, 0.90)    # top-right room
    goal_radius: float = 0.05
    door: float = 0.16                          # doorway width

    # --- dynamics ---
    step_size: float = 0.15                     # max |a|
    noise_std: float = 0.005                    # Sigma = (noise_std^2) I
    goal_bias: float = 0.0                      # RRT-style goal biasing (0 = pure exploration)

    # --- graph / merging ---
    novelty_radius: float = 0.03                # merge radius for states

    # --- online loop ---
    n_iters: int = 15
    k_trajectories: int = 8
    max_steps: int = 60
    max_retries: int = 6                        # veto retries per step
    max_sample_tries: int = 25                  # collision-free resamples
    alpha: float = 0.05                         # fixed uncertainty discount in veto (v3)

    # --- transformer ---
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 2
    dim_ff: int = 128                           # FFN width (free; d_model fixed by spec)
    dropout: float = 0.1                        # also drives MC-dropout at veto time
    max_len: int = 64                           # >= max_steps

    # --- MC-dropout epistemic uncertainty (veto) ---
    mc_samples: int = 25                        # stochastic forward passes per veto query
    mc_ema_beta: float = 0.6                    # per-node EMA weight on the running estimate

    # --- loss / optimisation ---
    huber_delta: float = 1.0
    lr: float = 1e-3
    train_epochs: int = 25
    batch_size: int = 32

    seed: int = 0
    device: str = "cpu"
    outdir: str = "."                           # where {env}_* outputs are written


# --------------------------------------------------------------------------- #
# Maze environments: walls + collision checking
#
# Grid environments are an (n_cols, n_rows) grid of rooms joined by one doorway
# per shared interior wall (2x2 = four-room, 4x2 = eight-room). The "decoy"
# environment is a custom "bait chamber": a tunnel aimed straight at the goal
# dead-ends just short of it, while the real entrance to the goal chamber is
# hidden around the top -- goal-greedy motion is lured into the dead end.
# Each ENVS entry also fixes the start/goal that suit its geometry.
# --------------------------------------------------------------------------- #
def build_grid_walls(cfg: Config, n_cols: int, n_rows: int):
    """Walls for an n_cols x n_rows grid of rooms. Interior walls carry one
    doorway per adjacent room pair (centred on each shared room span); the outer
    boundary is included so out-of-bounds moves are blocked too.

    At (2, 2) this reproduces the original four-room layout exactly."""
    L, d = cfg.L, cfg.door / 2.0
    rw, rh = L / n_cols, L / n_rows
    walls = [
        ((0, 0), (L, 0)), ((L, 0), (L, L)), ((L, L), (0, L)), ((0, L), (0, 0)),
    ]

    def segmented(fixed, doors, horizontal):
        """Emit wall segments along a line, skipping [c-d, c+d] gaps at each
        doorway centre in `doors`. `fixed` is the constant coordinate."""
        prev = 0.0
        for c in sorted(doors):
            a, b = c - d, c + d
            if a > prev:
                if horizontal:
                    walls.append(((prev, fixed), (a, fixed)))
                else:
                    walls.append(((fixed, prev), (fixed, a)))
            prev = b
        if prev < L:
            if horizontal:
                walls.append(((prev, fixed), (L, fixed)))
            else:
                walls.append(((fixed, prev), (fixed, L)))

    # vertical interior walls (doorways centred on each row span)
    for i in range(1, n_cols):
        segmented(i * rw, [(r + 0.5) * rh for r in range(n_rows)], horizontal=False)
    # horizontal interior walls (doorways centred on each column span)
    for j in range(1, n_rows):
        segmented(j * rh, [(c + 0.5) * rw for c in range(n_cols)], horizontal=True)

    return [((np.array(a, float), np.array(b, float))) for a, b in walls]


def build_decoy_walls(cfg: Config):
    """Bait-chamber decoy (start=(0.1,0.5), goal=(0.9,0.5)).

    A tunnel centred on the start->goal line runs east and dead-ends at a cap at
    x=0.78, one step short of the goal. The goal sits in a chamber (x>0.78) whose
    only doorway is up at y in [0.72, 0.90]. A goal-greedy agent drives straight
    into the tunnel and stalls against the cap; the goal is only reachable by
    detouring up and around to the chamber door."""
    L = cfg.L
    walls = [
        # outer boundary
        ((0, 0), (L, 0)), ((L, 0), (L, L)), ((L, L), (0, L)), ((0, L), (0, 0)),
        # decoy tunnel on the start->goal line, open to the west, capped at x=0.78
        ((0.40, 0.56), (0.78, 0.56)),      # tunnel top
        ((0.40, 0.44), (0.78, 0.44)),      # tunnel bottom
        ((0.78, 0.44), (0.78, 0.56)),      # dead-end cap (just before the goal)
        # west wall of the goal chamber; true doorway is the gap y in [0.72, 0.90]
        ((0.78, 0.00), (0.78, 0.44)),
        ((0.78, 0.56), (0.78, 0.72)),
        ((0.78, 0.90), (0.78, 1.00)),
    ]
    return [((np.array(a, float), np.array(b, float))) for a, b in walls]


# env name -> spec. Grid envs give {"grid": (cols, rows)}; custom envs give a
# {"builder": fn}. Every entry fixes the start/goal for that geometry.
ENVS = {
    "four_room":  {"grid": (2, 2), "start": (0.10, 0.10), "goal": (0.90, 0.90)},
    "eight_room": {"grid": (4, 2), "start": (0.10, 0.10), "goal": (0.90, 0.90)},
    "decoy":      {"builder": build_decoy_walls,
                   "start": (0.10, 0.50), "goal": (0.90, 0.50)},
}


def env_label(cfg: Config) -> str:
    spec = ENVS[cfg.env]
    if "grid" in spec:
        c, r = spec["grid"]
        return f"{c}x{r} rooms"
    return "bait-chamber decoy"


def build_walls(cfg: Config):
    """Dispatch to the layout selected by cfg.env."""
    if cfg.env not in ENVS:
        raise ValueError(f"unknown env '{cfg.env}'; choose from {list(ENVS)}")
    spec = ENVS[cfg.env]
    if "grid" in spec:
        return build_grid_walls(cfg, *spec["grid"])
    return spec["builder"](cfg)


def apply_env(cfg: Config) -> Config:
    """Set the env's start/goal on cfg (geometry-appropriate)."""
    spec = ENVS[cfg.env]
    cfg.start = spec["start"]
    cfg.goal = spec["goal"]
    return cfg


def _ccw(a, b, c) -> float:
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_intersect(p1, p2, p3, p4) -> bool:
    d1, d2 = _ccw(p3, p4, p1), _ccw(p3, p4, p2)
    d3, d4 = _ccw(p1, p2, p3), _ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def collides(s, s2, walls) -> bool:
    return any(_seg_intersect(s, s2, w0, w1) for (w0, w1) in walls)


# --------------------------------------------------------------------------- #
# Roadmap graph over merged states (novelty-radius merging + Dijkstra)
# --------------------------------------------------------------------------- #
class Graph:
    def __init__(self, start, novelty_radius: float):
        self.pos: List[np.ndarray] = [np.array(start, float)]  # node 0 = start
        self.adj = defaultdict(dict)                           # u -> {v: w}
        self.nr = novelty_radius
        self._stack = np.array(self.pos)
        self._dirty = False

    def _ensure(self):
        if self._dirty:
            self._stack = np.array(self.pos)
            self._dirty = False

    def nearest(self, p) -> Tuple[int, float]:
        self._ensure()
        d = np.linalg.norm(self._stack - p, axis=1)
        j = int(d.argmin())
        return j, float(d[j])

    def add_node(self, p) -> int:
        self.pos.append(np.array(p, float))
        self._dirty = True
        return len(self.pos) - 1

    def add_edge(self, u: int, v: int, w: float):
        if u == v:
            return
        if v not in self.adj[u] or w < self.adj[u][v]:
            self.adj[u][v] = w

    def dijkstra(self, src: int = 0, return_prev: bool = False):
        n = len(self.pos)
        g = [math.inf] * n
        prev = [-1] * n
        g[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > g[u]:
                continue
            for v, w in self.adj[u].items():
                nd = d + w
                if nd < g[v]:
                    g[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return (g, prev) if return_prev else g

    def undirected_neighbors(self):
        """Adjacency treated as undirected (both traversal directions), for
        path extraction over the roadmap."""
        nbr = defaultdict(set)
        for u, d in self.adj.items():
            for v in d:
                nbr[u].add(v)
                nbr[v].add(u)
        return nbr


# --------------------------------------------------------------------------- #
# Transformer  T_cost(c_hat, mu | h_t, a_t)
# --------------------------------------------------------------------------- #
def causal_mask(T: int, device) -> torch.Tensor:
    # boolean mask: True = position is NOT allowed to attend (strictly-upper
    # triangle). Equivalent to placing -inf above the diagonal, but matches the
    # boolean key-padding mask dtype used during batched training.
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)


class CostTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.max_len = cfg.max_len
        self.proj = nn.Linear(4, cfg.d_model)                 # fused token -> 64-d
        self.pos = nn.Parameter(torch.randn(cfg.max_len, cfg.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.cost_head = nn.Linear(cfg.d_model, 1)

    def forward(self, x, attn_mask, pad_mask=None):
        # x: (B, T, 4)
        T = x.shape[1]
        h = self.proj(x) + self.pos[:T].unsqueeze(0)          # add learned pos emb
        h = self.encoder(h, mask=attn_mask, src_key_padding_mask=pad_mask)
        return self.cost_head(h).squeeze(-1)                  # (B, T)


@torch.no_grad()
def predict_last_mc(model: CostTransformer, tokens: List[np.ndarray], cfg: Config):
    """MC-dropout (c_hat, std) for the *last* token of a single sequence, used
    at veto time. cfg.mc_samples stochastic passes (dropout ON) are batched as
    replicas of the same sequence -> one forward call, not a python loop."""
    toks = tokens[-cfg.max_len:]
    x = torch.tensor(np.array(toks), dtype=torch.float32, device=cfg.device)
    x = x.unsqueeze(0).repeat(cfg.mc_samples, 1, 1)            # (T, seq_len, 4)
    model.train()                                              # keep dropout active
    c = model(x, causal_mask(x.shape[1], cfg.device), None)    # (T, seq_len)
    model.eval()
    last = c[:, -1]
    return float(last.mean()), float(last.std(unbiased=True))


# --------------------------------------------------------------------------- #
# Huber loss (point estimate only; uncertainty comes from MC dropout instead)
# --------------------------------------------------------------------------- #
def huber_loss(c, tgt, pad, cfg: Config):
    valid = ~pad
    err = (tgt - c)[valid]
    ae = err.abs()
    d = cfg.huber_delta
    huber = torch.where(ae <= d, 0.5 * err ** 2, d * (ae - 0.5 * d))
    return huber.mean()


# --------------------------------------------------------------------------- #
# Trajectory collection with the current transformer (+ veto)
# --------------------------------------------------------------------------- #
def fuse(s: np.ndarray, a: np.ndarray) -> np.ndarray:
    return np.concatenate([s, a]).astype(np.float32)          # [sx, sy, ax, ay]


def sample_free(s, goal, cfg, walls, rng) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Sample a collision-free (a, s_next).  RRT-style with goal biasing."""
    for _ in range(cfg.max_sample_tries):
        if rng.random() < cfg.goal_bias:
            v = goal - s
            n = np.linalg.norm(v)
            dirv = v / n if n > 1e-9 else rng.normal(size=2)
            mag = cfg.step_size
        else:
            ang = rng.uniform(0, 2 * math.pi)
            dirv = np.array([math.cos(ang), math.sin(ang)])
            mag = cfg.step_size * math.sqrt(rng.random())     # uniform in disk
        a = dirv * mag
        eps = rng.normal(0, cfg.noise_std, size=2)
        s_next = s + a + eps
        if not (0.0 <= s_next[0] <= cfg.L and 0.0 <= s_next[1] <= cfg.L):
            continue
        if collides(s, s_next, walls):
            continue
        return a.astype(float), s_next.astype(float)
    return None


def collect_trajectory(model, graph, cfg, walls, rng, mu_ema: Dict[int, float]):
    """Returns (tokens, node_ids, reached).  tokens[t] = fuse(s_t, a_t);
    node_ids[t] = merged-graph id of the resulting state s_{t+1}.

    mu_ema: node id -> running EMA of MC-dropout std at that node, persisted
    and updated across the whole run (mutated in place)."""
    goal = np.array(cfg.goal, float)
    s = np.array(cfg.start, float)
    s_node = 0
    path_cost = 0.0
    tokens: List[np.ndarray] = []
    node_ids: List[int] = []
    reached = False
    n_veto = 0
    ema_log: List[Tuple[int, float, float]] = []     # (node, std_raw, mu_ema) per update

    for _ in range(cfg.max_steps):
        committed = False
        for _retry in range(cfg.max_retries):
            cand = sample_free(s, goal, cfg, walls, rng)
            if cand is None:
                continue
            a, s_next = cand
            tok = fuse(s, a)

            j, dist = graph.nearest(s_next)
            visited = dist < cfg.novelty_radius

            if visited:
                # merge into the existing node; apply the cost-to-come veto.
                # The sampler checked s -> s_next, but the STORED edge is
                # s -> pos[j] (the merged node), which can differ by up to
                # novelty_radius and cross a wall the sampled segment did not.
                # Re-check the actual edge and reject if so.
                node_pos = graph.pos[j]
                if collides(s, node_pos, walls):
                    continue                                   # reject, retry
                step_cost = float(np.linalg.norm(node_pos - s))
                cand_cost = path_cost + step_cost
                c_hat, std_raw = predict_last_mc(model, tokens + [tok], cfg)
                prev = mu_ema.get(j)
                mu_smoothed = std_raw if prev is None else (
                    cfg.mc_ema_beta * prev + (1 - cfg.mc_ema_beta) * std_raw)
                mu_ema[j] = mu_smoothed
                ema_log.append((j, std_raw, mu_smoothed))
                # take the action iff current path is competitive with best-known
                if cand_cost < c_hat + (mu_smoothed - cfg.alpha):
                    node_id, s_new = j, node_pos
                else:
                    n_veto += 1
                    continue                                   # VETO -> retry
            else:
                node_id = graph.add_node(s_next)
                s_new = s_next
                step_cost = float(np.linalg.norm(s_new - s))
                cand_cost = path_cost + step_cost

            # commit the accepted transition
            graph.add_edge(s_node, node_id, step_cost)
            tokens.append(tok)
            node_ids.append(node_id)
            path_cost = cand_cost
            s, s_node = s_new, node_id
            committed = True
            break

        if not committed:                                      # retries exhausted
            break
        if np.linalg.norm(s - goal) < cfg.goal_radius:
            reached = True
            break

    return tokens, node_ids, reached, n_veto, ema_log


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def collate(batch, cfg):
    B = len(batch)
    T = max(len(t[1]) for t in batch)
    x = torch.zeros(B, T, 4)
    tgt = torch.zeros(B, T)
    pad = torch.ones(B, T, dtype=torch.bool)
    for b, (toks, tgts) in enumerate(batch):
        L = len(tgts)
        x[b, :L] = torch.tensor(np.array(toks), dtype=torch.float32)
        tgt[b, :L] = torch.tensor(tgts, dtype=torch.float32)
        pad[b, :L] = False
    return (x.to(cfg.device), tgt.to(cfg.device), pad.to(cfg.device))


def build_sequences(D, g):
    """Turn the replay buffer into (tokens, dijkstra-cost targets) sequences."""
    seqs = []
    for tokens, node_ids, _ in D:
        if not node_ids:
            continue
        tgts = np.array([g[i] for i in node_ids], float)
        if not np.all(np.isfinite(tgts)):          # keep only start-connected
            continue
        seqs.append((np.array(tokens), tgts))
    return seqs


def train_transformer(model, D, g, cfg, rng):
    seqs = build_sequences(D, g)
    if not seqs:
        return float("nan")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    order = np.arange(len(seqs))
    last = float("nan")
    for _ in range(cfg.train_epochs):
        model.train()
        rng.shuffle(order)
        total, nb = 0.0, 0
        for i in range(0, len(order), cfg.batch_size):
            batch = [seqs[k] for k in order[i:i + cfg.batch_size]]
            x, tgt, pad = collate(batch, cfg)
            c = model(x, causal_mask(x.shape[1], cfg.device), pad)
            loss = huber_loss(c, tgt, pad, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        last = total / max(nb, 1)
    return last


@torch.no_grad()
def field_mae(model, D, g, cfg):
    """Mean |c_hat - g| of the transformer's cost prediction vs. the Dijkstra
    label, over all transitions in the buffer (how well the field is learned)."""
    model.eval()
    seqs = build_sequences(D, g)
    if not seqs:
        return float("nan")
    err, n = 0.0, 0
    for i in range(0, len(seqs), cfg.batch_size):
        x, tgt, pad = collate(seqs[i:i + cfg.batch_size], cfg)
        c = model(x, causal_mask(x.shape[1], cfg.device), pad)
        v = ~pad
        err += (c[v] - tgt[v]).abs().sum().item()
        n += int(v.sum().item())
    return err / max(n, 1)


def goal_coverage(graph, g, cfg, radius=0.15):
    """Nodes near the goal, the nearest node to the goal, and its cost-to-come.
    Radius-based so it works for any layout (corner or interior goals)."""
    gpos = np.array(cfg.goal)
    P = np.array(graph.pos)
    gv = np.array(g)
    d = np.linalg.norm(P - gpos, axis=1)
    near = int(np.sum(d <= radius))
    j = int(d.argmin())
    return near, float(d[j]), float(gv[j])


# --------------------------------------------------------------------------- #
# Path extraction: actual shortest path (Dijkstra) and the transformer's path
# --------------------------------------------------------------------------- #
def path_length(graph, nodes):
    """Geometric length of a node sequence (sum of consecutive positions).
    Consecutive roadmap nodes are edge-connected, so this equals the path cost."""
    if len(nodes) < 2:
        return 0.0
    P = [graph.pos[i] for i in nodes]
    return float(sum(np.linalg.norm(P[k + 1] - P[k]) for k in range(len(P) - 1)))


def dijkstra_path(graph, prev, target):
    """Back-track prev[] from `target` to the start (node 0). Returns the node
    sequence in start -> target order (empty if target is unreachable)."""
    if target != 0 and prev[target] == -1:
        return []
    seq = [target]
    while seq[-1] != 0:
        p = prev[seq[-1]]
        if p == -1:
            return []                       # not connected to start
        seq.append(p)
    seq.reverse()
    return seq


@torch.no_grad()
def predict_cost_field(model, graph, cfg):
    """Transformer's predicted cost-to-come at every node, queried consistently
    as the token (node_pos, zero-action) -> c_hat of that node. History-free, so
    it is a single well-defined scalar field over the roadmap."""
    model.eval()
    P = np.asarray(graph.pos, dtype=np.float32)
    toks = np.concatenate([P, np.zeros_like(P)], axis=1)      # [x, y, 0, 0]
    chat = np.empty(len(P), dtype=np.float64)
    bs = 512
    for i in range(0, len(P), bs):
        x = torch.tensor(toks[i:i + bs, None, :], device=cfg.device)  # (B,1,4)
        c = model(x, causal_mask(1, cfg.device), None)
        chat[i:i + bs] = c[:, 0].cpu().numpy()
    return chat


@torch.no_grad()
def transformer_path(model, graph, cfg, target, chat=None):
    """Greedy descent on the transformer's predicted cost-to-come. From the
    goal's nearest node, repeatedly step to the roadmap neighbour with the lowest
    predicted c_hat (i.e. the one the transformer thinks is closest to the
    start), backtracking out of dead-ends, until the start (node 0) is reached.

    Backtracking means the walk completes whenever the roadmap is connected, so
    the informative quantity is not *whether* it arrives but how much longer the
    greedy path is than Dijkstra's optimum (the cost ratio) -- that gap measures
    how the learned field's local bumps deflect a purely greedy planner."""
    if chat is None:
        chat = predict_cost_field(model, graph, cfg)
    nbr = graph.undirected_neighbors()
    visited = {target}
    stack = [target]                                   # current path goal->...
    budget = 50 * len(graph.pos)
    steps = 0
    while stack and stack[-1] != 0 and steps < budget:
        steps += 1
        cur = stack[-1]
        cands = sorted((v for v in nbr[cur] if v not in visited),
                       key=lambda z: chat[z])          # greedy: lowest predicted cost
        if cands:
            nxt = cands[0]
            visited.add(nxt)
            stack.append(nxt)
        else:
            stack.pop()                                # dead end -> backtrack
    reached = bool(stack and stack[-1] == 0)
    seq = stack[::-1] if reached else stack[:]         # start .. target
    return seq, reached


def extract_and_save_paths(model, graph, g, prev, cfg, log):
    """Compute the roadmap Dijkstra path and the transformer's greedy path, log
    a comparison, and save them to {env}_paths.json."""
    import json
    goal_node = graph.nearest(np.array(cfg.goal))[0]

    d_nodes = dijkstra_path(graph, prev, goal_node)
    d_cost = g[goal_node] if d_nodes else float("inf")

    t_nodes, t_reached = transformer_path(model, graph, cfg, goal_node)
    t_cost = path_length(graph, t_nodes)

    d_goal = float(np.linalg.norm(graph.pos[goal_node] - np.array(cfg.goal)))
    covered = d_goal <= cfg.goal_radius
    opt = (t_cost / d_cost) if (d_nodes and math.isfinite(d_cost) and d_cost > 0) else float("nan")

    log("")
    log(f"Nearest reached node to goal: {goal_node} at "
        f"{tuple(round(float(x), 3) for x in graph.pos[goal_node])} "
        f"(d={d_goal:.3f} from true goal)")
    if not covered:
        log(f"  NOTE: collection never reached the goal (nearest node is "
            f"{d_goal:.3f} away) -- the roadmap has no nodes at the goal, so the")
        log(f"  paths below only reach this dead-end, not the true goal. This is "
            f"a coverage failure of the sampler.")
    log(f"Roadmap Dijkstra path:    {len(d_nodes):>3} nodes, cost = {d_cost:.3f} "
        f"(to nearest reached node)")
    log(f"Transformer greedy path:  {len(t_nodes):>3} nodes, cost = {t_cost:.3f}, "
        f"reached_start = {t_reached} (to same node)")
    if math.isfinite(opt):
        log(f"Transformer/roadmap-Dijkstra cost ratio: {opt:.3f}  (1.0 = matches roadmap optimum)")

    payload = {
        "env": cfg.env,
        "start": list(cfg.start),
        "goal": list(cfg.goal),
        "goal_reached_by_collection": bool(covered),
        "roadmap_dijkstra_path": {                   # over collected nodes only
            "target_node": int(goal_node),
            "target_dist_from_goal": round(d_goal, 4),
            "nodes": [int(i) for i in d_nodes],
            "positions": [graph.pos[i].round(4).tolist() for i in d_nodes],
            "cost": float(d_cost) if math.isfinite(d_cost) else None,
        },
        "transformer_path": {                        # greedy on predicted field
            "nodes": [int(i) for i in t_nodes],
            "positions": [graph.pos[i].round(4).tolist() for i in t_nodes],
            "cost": float(t_cost),
            "reached_start": bool(t_reached),
        },
    }
    out = os.path.join(cfg.outdir, f"{cfg.env}_paths.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log(f"[paths] saved -> {out}")
    d_xy = np.array([graph.pos[i] for i in d_nodes]) if d_nodes else None
    t_xy = np.array([graph.pos[i] for i in t_nodes]) if len(t_nodes) > 1 else None
    return d_xy, t_xy


# --------------------------------------------------------------------------- #
# Plot (maze, learned cost-to-come field, a sample successful trajectory)
# --------------------------------------------------------------------------- #
def plot_results(graph, g, walls, cfg, d_xy=None, t_xy=None,
                 path="four_room_cost.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                     # pragma: no cover
        print(f"[plot] skipped: {e}")
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    for (w0, w1) in walls:
        ax.plot([w0[0], w1[0]], [w0[1], w1[1]], color="k", lw=2)

    P = np.array(graph.pos)
    gv = np.array(g)
    finite = np.isfinite(gv)
    sc = ax.scatter(P[finite, 0], P[finite, 1], c=gv[finite], s=14,
                    cmap="viridis", zorder=3)
    fig.colorbar(sc, ax=ax, label="Dijkstra cost-to-come  g(s)")

    # roadmap Dijkstra vs. transformer greedy descent
    if d_xy is not None and len(d_xy):
        ax.plot(d_xy[:, 0], d_xy[:, 1], color="crimson", lw=2.0, alpha=0.95,
                zorder=5, label="roadmap Dijkstra (to nearest node)")
    if t_xy is not None and len(t_xy):
        ax.plot(t_xy[:, 0], t_xy[:, 1], color="deepskyblue", lw=1.7, ls="--",
                alpha=0.95, zorder=6, label="transformer greedy path")

    ax.scatter(*cfg.start, c="white", edgecolors="k", s=90, zorder=5, label="start")
    ax.scatter(*cfg.goal, marker="*", c="gold", edgecolors="k", s=240,
               zorder=5, label="goal")
    ax.add_patch(plt.Circle(cfg.goal, cfg.goal_radius, color="gold",
                            alpha=0.25, zorder=2))
    ax.set_xlim(-0.02, cfg.L + 0.02)
    ax.set_ylim(-0.02, cfg.L + 0.02)
    ax.set_aspect("equal")
    ax.set_title(f"{cfg.env} ({env_label(cfg)}): cost-to-come + shortest paths")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Main online loop  (Algorithm 1)
# --------------------------------------------------------------------------- #
def main(cfg: Config = Config()):
    import csv
    apply_env(cfg)                                           # set env start/goal
    # learned positional-embedding table must have a row for every position, so
    # it must cover the trajectory horizon. d_model is unaffected (model width
    # and sequence length are independent axes).
    cfg.max_len = max(cfg.max_len, cfg.max_steps)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # tee every printed line to {outdir}/{env}_run.log
    os.makedirs(cfg.outdir, exist_ok=True)
    log_path = os.path.join(cfg.outdir, f"{cfg.env}_run.log")
    logf = open(log_path, "w")

    def log(msg=""):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    # evolving per-node MC-dropout EMA: one row per veto-time update
    mu_ema_path = os.path.join(cfg.outdir, f"{cfg.env}_mu_ema.csv")
    mu_ema_f = open(mu_ema_path, "w", newline="")
    mu_ema_writer = csv.writer(mu_ema_f)
    mu_ema_writer.writerow(["iter", "node", "std_raw", "mu_ema"])

    walls = build_walls(cfg)
    model = CostTransformer(cfg).to(cfg.device)
    graph = Graph(cfg.start, cfg.novelty_radius)
    D = []                                                    # replay buffer
    rows = []                                                 # per-iter metrics
    mu_ema: Dict[int, float] = {}                             # node id -> EMA(std) (MC dropout)

    log(f"env = {cfg.env}  ({env_label(cfg)})  "
        f"start={cfg.start} goal={cfg.goal}  seed={cfg.seed}")
    log(f"horizon (max_steps) = {cfg.max_steps}  |  pos-emb rows (max_len) = "
        f"{cfg.max_len}  |  d_model = {cfg.d_model}  |  alpha = {cfg.alpha}  "
        f"|  goal_bias = {cfg.goal_bias}")
    hdr = f"{'iter':>4} {'nodes':>6} {'trans':>7} {'succ':>5} {'len':>5} {'veto':>5} {'loss':>8} {'fieldMAE':>9}"
    log(hdr)
    for it in range(cfg.n_iters):
        # 3-4: collect k trajectories with the current model; append to D
        succ, vetoes, steps = 0, 0, 0
        for _ in range(cfg.k_trajectories):
            tokens, node_ids, reached, nv, ema_log = collect_trajectory(
                model, graph, cfg, walls, rng, mu_ema)
            D.append((tokens, node_ids, reached))
            succ += int(reached)
            vetoes += nv
            steps += len(node_ids)
            for node_id, std_raw, mu_val in ema_log:
                mu_ema_writer.writerow([it, node_id, f"{std_raw:.6f}", f"{mu_val:.6f}"])
        mu_ema_f.flush()
        # 5-6: build graph (already incremental) + Dijkstra labels
        g = graph.dijkstra(0)
        # 7: train the transformer on the labelled buffer
        loss = train_transformer(model, D, g, cfg, rng)
        mae = field_mae(model, D, g, cfg)

        n_trans = sum(len(t[1]) for t in D)
        avg_len = steps / cfg.k_trajectories
        log(f"{it:>4} {len(graph.pos):>6} {n_trans:>7} "
            f"{succ:>2}/{cfg.k_trajectories:<2} {avg_len:>5.1f} {vetoes:>5} "
            f"{loss:>8.3f} {mae:>9.4f}")
        rows.append({
            "iter": it, "nodes": len(graph.pos), "trans": n_trans,
            "succ": succ, "avg_len": round(avg_len, 3), "vetoes": vetoes,
            "loss": round(loss, 6), "field_mae": round(mae, 6),
        })

    # write per-iteration metrics table
    csv_path = os.path.join(cfg.outdir, f"{cfg.env}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # final summary + path extraction
    g, prev = graph.dijkstra(0, return_prev=True)
    n_in, d_goal, g_goal = goal_coverage(graph, g, cfg)
    log(f"\nNodes within 0.15 of goal: {n_in} | nearest node to goal: "
        f"d={d_goal:.3f}, cost-to-come g={g_goal:.3f}")
    log(f"Field cost-to-come MAE (transformer vs Dijkstra): "
        f"{field_mae(model, D, g, cfg):.4f}")

    d_xy, t_xy = extract_and_save_paths(model, graph, g, prev, cfg, log)

    out = plot_results(graph, g, walls, cfg, d_xy=d_xy, t_xy=t_xy,
                       path=os.path.join(cfg.outdir, f"{cfg.env}_cost.png"))
    if out:
        log(f"[plot]    saved -> {out}")
    log(f"[log]     saved -> {log_path}")
    log(f"[metrics] saved -> {csv_path}")
    log(f"[mu_ema]  saved -> {mu_ema_path}")
    logf.close()
    mu_ema_f.close()
    return model, graph, g


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Online transformer cost-to-come training in a maze.")
    p.add_argument("--env", choices=list(ENVS), default="four_room",
                   help="maze environment (default: four_room)")
    p.add_argument("--iters", type=int, default=None, help="override n_iters")
    p.add_argument("--k", type=int, default=None,
                   help="override trajectories collected per iteration")
    p.add_argument("--seed", type=int, default=None, help="override random seed")
    p.add_argument("--max-steps", type=int, default=None, dest="max_steps",
                   help="trajectory horizon; positional-embedding table is "
                        "auto-sized to cover it (default: 60)")
    p.add_argument("--alpha", type=float, default=None,
                   help="veto uncertainty discount (constant > 0); lower = looser "
                        "veto, more exploration (default: 0.05)")
    p.add_argument("--alpha-sweep", type=str, default=None,
                   help="comma-separated alpha values (e.g. 0.0,0.01,0.02,0.03,0.04,0.05); "
                        "runs the full loop once per value with everything else held "
                        "fixed (same seed), writing outputs to <out>/alpha_<value>/. "
                        "Overrides --alpha.")
    p.add_argument("--out", type=str, default=None,
                   help="output directory for {env}_* files (default: current dir)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(env=args.env)
    if args.iters is not None:
        cfg.n_iters = args.iters
    if args.k is not None:
        cfg.k_trajectories = args.k
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.out is not None:
        cfg.outdir = args.out

    if args.alpha_sweep:
        base_out = cfg.outdir
        for a in (float(v) for v in args.alpha_sweep.split(",")):
            run_cfg = replace(cfg, alpha=a, outdir=os.path.join(base_out, f"alpha_{a:g}"))
            main(run_cfg)
    else:
        main(cfg)

#!/usr/bin/env python3
"""G10 analysis: critic estimate vs realised discounted return, per evaluated checkpoint."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

TERMINAL = {"collision", "off_road", "route_completed"}


def load(path):
    eps = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            eps.setdefault(int(row["episode"]), []).append(row)
    out = []
    for e, rows in sorted(eps.items()):
        rows.sort(key=lambda r: int(r["step"]))
        r = np.array([float(x["reward"]) for x in rows])
        last = rows[-1]
        if "terminated" in last and last["terminated"] != "":
            term = bool(int(last["terminated"]))
        else:
            term = last.get("termination", "") in TERMINAL
        has_q = "q1" in last and last["q1"] != ""
        out.append({
            "episode": e, "reward": r, "terminated": term,
            "reason": last.get("termination", ""),
            "q1": np.array([float(x["q1"]) for x in rows]) if has_q else None,
            "q2": np.array([float(x["q2"]) for x in rows]) if has_q else None,
            "speed": np.array([float(x["speed_kmh"]) for x in rows]),
            "thr": np.array([float(x["throttle_brake"]) for x in rows]),
            "obst": np.array([float(x["dist_obstacle_m"]) if x.get("dist_obstacle_m") not in (None, "")
                              else math.inf for x in rows]),
        })
    return out


def returns(r, gamma):
    g = np.zeros_like(r)
    acc = 0.0
    for t in range(len(r) - 1, -1, -1):
        acc = r[t] + gamma * acc
        g[t] = acc
    return g


def cluster_boot(per_ep_vals, n=2000, seed=0):
    per_ep_vals = [v for v in per_ep_vals if len(v)]
    if not per_ep_vals:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    k = len(per_ep_vals)
    point = float(np.mean(np.concatenate(per_ep_vals)))
    boots = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        boots.append(np.mean(np.concatenate([per_ep_vals[i] for i in idx])))
    return point, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def analyse(path, gamma, H):
    eps = load(path)
    res = {"file": Path(path).name, "episodes": len(eps),
           "terminated": sum(e["terminated"] for e in eps),
           "completions": sum(e["reason"] == "route_completed" for e in eps)}
    G_keep, Q1_keep, Qmin_keep, bias1, biasmin, deltas = [], [], [], [], [], []
    park_G, park_Q1, n_states, n_excl = [], [], 0, 0
    term_check = []
    for e in eps:
        G = returns(e["reward"], gamma)
        T = len(G)
        keep = np.ones(T, bool)
        if not e["terminated"]:
            keep[max(0, T - H):] = False
        n_states += int(keep.sum()); n_excl += int((~keep).sum())
        if e["terminated"]:
            term_check.append(abs(G[-1] - e["reward"][-1]))
        G_keep.append(G[keep])
        if e["q1"] is not None:
            q1, q2 = e["q1"], e["q2"]
            qmin = np.minimum(q1, q2)
            Q1_keep.append(q1[keep]); Qmin_keep.append(qmin[keep])
            bias1.append((q1 - G)[keep]); biasmin.append((qmin - G)[keep])
            nxt = np.append(q1[1:], 0.0)
            not_term = np.ones(T); not_term[-1] = 0.0 if e["terminated"] else 1.0
            d = e["reward"] + gamma * not_term * nxt - q1
            deltas.append(d[:-1] if not e["terminated"] else d)
            parked = (e["speed"] < 1.0) & (e["thr"] <= -0.9) & np.isinf(e["obst"]) & keep
            park_G.append(G[parked]); park_Q1.append(q1[parked])
    res["states_used"] = n_states
    res["states_excluded_near_truncation"] = n_excl
    res["terminal_return_equals_reward_maxabs"] = float(max(term_check)) if term_check else None
    res["G_mean"], res["G_ci95"] = cluster_boot(G_keep)
    if Q1_keep:
        res["Q1_mean"], res["Q1_ci95"] = cluster_boot(Q1_keep)
        res["Qmin_mean"], res["Qmin_ci95"] = cluster_boot(Qmin_keep)
        res["bias_Q1_minus_G"], res["bias_Q1_ci95"] = cluster_boot(bias1)
        res["bias_Qmin_minus_G"], res["bias_Qmin_ci95"] = cluster_boot(biasmin)
        allg = np.concatenate([g for g in G_keep if len(g)])
        allq = np.concatenate([q for q in Q1_keep if len(q)])
        res["corr_Q1_G"] = float(np.corrcoef(allq, allg)[0, 1]) if len(allg) > 2 else None
        D = np.concatenate(deltas)
        res["bellman_delta_mean"] = float(D.mean())
        res["bellman_delta_mean_abs"] = float(np.abs(D).mean())
        res["reward_mean_abs"] = float(np.mean(np.abs(np.concatenate([e["reward"] for e in eps]))))
        n_park = int(sum(len(p) for p in park_G))
        res["parked_states"] = n_park
        if n_park:
            res["parked_G_mean"] = float(np.mean(np.concatenate(park_G)))
            res["parked_Q1_mean"] = float(np.mean(np.concatenate(park_Q1)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--H", type=int, default=300)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = [analyse(f, a.gamma, a.H) for f in a.files]
    for r in out:
        print(json.dumps(r, indent=None, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())

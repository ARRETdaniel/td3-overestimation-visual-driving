#!/usr/bin/env python3
import argparse, itertools, json, math, re, sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-09-23" / "g10"))
from g10_analyse import analyse, load

D2_STEPS = (500000, 525000, 550000, 575000, 600000)
FINAL = 600000
H, GAMMA = 300, 0.99
SPAWN_STEPS = 10
CKPT_RE = re.compile(r"^(?:\w+?_)?s(\d+)_step(\d+)_telemetry\.csv$")
FRESH_RE = re.compile(r"^\w+?_s(\d+)_step(\d+)_fresh_telemetry\.csv$")


def exact_perm_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pooled = np.concatenate([x, y]); n, k = len(pooled), len(x)
    obs = abs(y.mean() - x.mean()); total = hits = 0
    for idx in itertools.combinations(range(n), k):
        m = np.zeros(n, bool); m[list(idx)] = True
        d = abs(pooled[~m].mean() - pooled[m].mean())
        hits += d >= obs - 1e-12; total += 1
    return hits / total, total


def welch_ci(x, y, level=0.95):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else float("inf")
    t = stats.t.ppf(0.5 + level / 2, df); d = y.mean() - x.mean()
    return d - t * se, d + t * se


def boot_diff_ci(x, y, n=20000, seed=0):
    rng = np.random.default_rng(seed); x, y = np.asarray(x, float), np.asarray(y, float)
    d = [rng.choice(y, len(y)).mean() - rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def power_two_sample(d, n1, n2, alpha=0.05):
    if not np.isfinite(d) or d == 0:
        return float("nan")
    df = n1 + n2 - 2; nc = abs(d) * math.sqrt(n1 * n2 / (n1 + n2)); tc = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


def n_for_power(d, target=0.80):
    for n in range(2, 2001):
        if power_two_sample(d, n, n) >= target:
            return n
    return None


def compare(x, y, label):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = dict(label=label, n=[len(x), len(y)], mean=[float(x.mean()), float(y.mean())],
               per_seed=[list(map(float, x)), list(map(float, y))], diff=float(y.mean() - x.mean()))
    if len(x) < 2 or len(y) < 2:
        out["note"] = "fewer than 2 runs in an arm - no test"
        return out
    w = stats.ttest_ind(y, x, equal_var=False)
    mw = stats.mannwhitneyu(y, x, alternative="two-sided", method="exact")
    pp, nperm = exact_perm_p(x, y)
    sp = math.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    d = (y.mean() - x.mean()) / sp if sp > 0 else float("nan")
    out.update(welch_t=float(w.statistic), welch_p=float(w.pvalue), mw_p=float(mw.pvalue),
               perm_p=float(pp), perm_relabellings=nperm, ci95_welch=welch_ci(x, y), ci95_boot=boot_diff_ci(x, y),
               cohen_d=float(d), power_at_observed_d=power_two_sample(d, len(x), len(y)),
               n_per_arm_for_80pct=n_for_power(d))
    return out


def spawn_collisions(path):
    return sum(1 for e in load(path) if e["reason"] == "collision" and len(e["reward"]) <= SPAWN_STEPS)


def arm_runs(dirs):
    runs = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*_telemetry.csv")):
            if f.name.endswith("_fresh_telemetry.csv"):
                continue
            m = CKPT_RE.match(f.name)
            if not m or int(m.group(2)) not in D2_STEPS:
                continue
            seed, step = int(m.group(1)), int(m.group(2))
            if step in runs.get(seed, {}):
                sys.exit(f"duplicate checkpoint s{seed} step {step} in {dirs} - resolve before analysing")
            r = analyse(str(f), GAMMA, H); r["spawn_collisions"] = spawn_collisions(str(f)); r["path"] = str(f)
            runs.setdefault(seed, {})[step] = r
    return runs


def fresh_runs(dirs):
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*_fresh_telemetry.csv")):
            m = FRESH_RE.match(f.name)
            if m:
                r = analyse(str(f), GAMMA, H)
                out[int(m.group(1))] = dict(step=int(m.group(2)), completions=r["completions"],
                                            episodes=r["episodes"], spawn_collisions=spawn_collisions(str(f)))
    return out


def per_seed(runs):
    comp, bias, comp_final, bias_final, detail, incomplete = {}, {}, {}, {}, {}, {}
    for seed, st in sorted(runs.items()):
        missing = [s for s in D2_STEPS if s not in st]
        if missing:
            incomplete[seed] = missing
            continue
        eps = sum(st[s]["episodes"] for s in D2_STEPS)
        comp[seed] = sum(st[s]["completions"] for s in D2_STEPS) / eps
        bias[seed] = float(np.mean([st[s]["bias_Q1_minus_G"] for s in D2_STEPS]))
        comp_final[seed] = st[FINAL]["completions"] / st[FINAL]["episodes"]
        bias_final[seed] = st[FINAL]["bias_Q1_minus_G"]
        detail[seed] = {s: dict(completions=st[s]["completions"], episodes=st[s]["episodes"],
                                bias=st[s]["bias_Q1_minus_G"], spawn_collisions=st[s]["spawn_collisions"])
                        for s in D2_STEPS}
    return comp, bias, comp_final, bias_final, detail, incomplete


def selftest():
    x, y = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    p, n = exact_perm_p(x, y); assert n == 252 and abs(p - 2 / 252) < 1e-12, (p, n)
    assert abs(stats.mannwhitneyu(y, x, alternative="two-sided", method="exact").pvalue - 2 / 252) < 1e-12
    p, n = exact_perm_p([0, 0, 1], [0, 1, 0]); assert n == 20 and p == 1.0
    assert abs(power_two_sample(0.81, 5, 5) - 0.21) < 0.02
    assert n_for_power(0.81) in (25, 26)
    lo, hi = welch_ci(x, y); assert lo < 5 < hi
    print("selftest: exact permutation, Mann-Whitney, power (0.21 @ n=5, d=0.81) and n for 80% all as expected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], help="NAME=DIR[,DIR...]; first arm is the reference")
    ap.add_argument("--fresh", action="append", default=[], help="NAME=DIR[,DIR...]")
    ap.add_argument("--json", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    arms = [(s.split("=", 1)[0], s.split("=", 1)[1].split(",")) for s in a.arm]
    fresh = {s.split("=", 1)[0]: s.split("=", 1)[1].split(",") for s in a.fresh}
    res = dict(rule="PREREG_D2_CHECKPOINT_RULE.md (2026-09-24 09:40)", steps=D2_STEPS, H=H, gamma=GAMMA, arms={})
    for name, dirs in arms:
        comp, bias, cf, bf, detail, incomplete = per_seed(arm_runs(dirs))
        fr = fresh_runs(fresh[name]) if name in fresh else {}
        res["arms"][name] = dict(dirs=dirs, completion=comp, bias=bias, completion_final=cf, bias_final=bf,
                                 fresh=fr, detail=detail, incomplete_runs=incomplete)
        print(f"\n== {name}  ({', '.join(dirs)})")
        if incomplete:
            print(f"   INCOMPLETE runs (not in the primary): {incomplete}")
        print("   seed | completions per ckpt 500/525/550/575/600k | D2 completion | D2 bias (Q1-G) | 600k: comp, bias | fresh")
        for s in comp:
            per = " ".join(f"{detail[s][st]['completions']:>2}" for st in D2_STEPS)
            spw = sum(detail[s][st]["spawn_collisions"] for st in D2_STEPS)
            f = fr.get(s); fs = f"{f['completions']}/{f['episodes']} @{f['step'] // 1000}k" if f else "-"
            print(f"   s{s} | {per} | {comp[s]:.2f} | {bias[s]:+7.1f} | {cf[s]:.1f}, {bf[s]:+7.1f} | {fs}"
                  + (f"  [spawn collisions {spw}]" if spw else ""))
        if comp:
            print(f"   arm: completion {np.mean(list(comp.values())):.3f}  bias {np.mean(list(bias.values())):+.1f}  (n={len(comp)})")
    if len(arms) == 2:
        (A, _), (B, _) = arms
        RA, RB = res["arms"][A], res["arms"][B]
        tests = [compare(list(RA["completion"].values()), list(RB["completion"].values()), "PRIMARY completion (D2)"),
                 compare(list(RA["bias"].values()), list(RB["bias"].values()), "PRIMARY G10 bias (D2)"),
                 compare(list(RA["completion_final"].values()), list(RB["completion_final"].values()), "secondary: completion @600k"),
                 compare(list(RA["bias_final"].values()), list(RB["bias_final"].values()), "secondary: G10 bias @600k")]
        if RA["fresh"] and RB["fresh"]:
            tests.append(compare([v["completions"] / v["episodes"] for v in RA["fresh"].values()],
                                 [v["completions"] / v["episodes"] for v in RB["fresh"].values()],
                                 "secondary: select/report fresh completion"))
        res["tests"] = tests
        print(f"\n== {B} - {A}, unpaired, seed = unit")
        for t in tests:
            if "welch_p" not in t:
                print(f"   {t['label']}: {t.get('note')}"); continue
            print(f"   {t['label']}: {A} {t['mean'][0]:.3f} (n={t['n'][0]}) vs {B} {t['mean'][1]:.3f} (n={t['n'][1]}); "
                  f"diff {t['diff']:+.3f}, 95% CI Welch [{t['ci95_welch'][0]:+.3f}, {t['ci95_welch'][1]:+.3f}] "
                  f"boot [{t['ci95_boot'][0]:+.3f}, {t['ci95_boot'][1]:+.3f}]; Welch p={t['welch_p']:.3f}, "
                  f"exact perm p={t['perm_p']:.3f} ({t['perm_relabellings']}), MW exact p={t['mw_p']:.3f}; "
                  f"d={t['cohen_d']:+.2f}, power@observed d={t['power_at_observed_d']:.2f} (optimistic), "
                  f"n/arm for 80%={t['n_per_arm_for_80pct']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Recompute the per-wave contrasts of the paper from results/per_run.csv.

Each wave is a set of runs trained together, five per arm. For every pair of arms compared in the paper, this prints
the difference of the arm means, the Welch 95% interval, the exact two-sided permutation p over all relabellings,
the exact Mann-Whitney p and Cohen's d, computed by d2_analyse.compare (the module used for every reported test).
Usage: python3 analysis/reproduce_statistics.py   (from the repository root; needs numpy and scipy)
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d2_analyse as d2   # noqa: E402

ROWS = list(csv.DictReader(open(HERE.parent / "results" / "per_run.csv", newline="")))

# (wave, reference arm, compared arm, measure); the printed difference is compared arm minus reference arm
CONTRASTS = [
    ("earlier", "TD3", "AHE", "bias_q1_npc20"), ("earlier", "TD3", "AHE", "completion_npc20"),
    ("single", "TD3", "AHE", "bias_q1_npc20"), ("single", "TD3", "AHE", "completion_npc20"),
    ("single", "TD3", "AHE", "completion_npc50"),
    ("single_ahe2h", "AHE-2H", "AHE (fresh)", "bias_q1_npc20"),
    ("multi", "TD3", "AHE", "bias_q1_npc20"), ("multi", "TD3", "AHE", "completion_npc20"),
    ("multi", "TD3", "AHE", "completion_npc50"), ("multi", "TD3", "AHE", "heldout_rc_npc20"),
    ("multi", "TD3", "AHE", "town02_rc_npc20"),
    ("multi_caps", "TD3+CAPS", "AHE+CAPS", "bias_q1_npc20"), ("multi_caps", "TD3+CAPS", "AHE+CAPS", "completion_npc20"),
    ("multi_cdq", "AHE+CDQ", "AHE (fresh)", "bias_q1_npc20"), ("multi_cdq", "AHE+CDQ", "AHE (fresh)", "completion_npc20"),
]
for ref, alt in (("AHE", "AHE_CDQ"), ("AHE", "AHE_TPS"), ("AHE", "AHE_DP"), ("TD3", "TD3_NOCDQ"), ("TD3", "TD3_NOTPS"), ("TD3", "TD3_NODP")):
    CONTRASTS.append(("ablation", ref, alt, "bias_q1_npc20"))


def values(wave, arm, measure):
    return [float(r[measure]) for r in ROWS if r["wave"] == wave and r["arm"] == arm and r[measure] != ""]


print(f"{'wave':13s} {'compared arm - reference arm':34s} {'measure':17s} {'diff':>9s}  {'Welch 95%':>20s}  {'perm p':>7s}  {'MW p':>6s}  {'d':>6s}")
for wave, ref, alt, measure in CONTRASTS:
    x, y = values(wave, ref, measure), values(wave, alt, measure)
    if len(x) < 2 or len(y) < 2:
        continue
    c = d2.compare(x, y, measure)
    lo, hi = c["ci95_welch"]
    print(f"{wave:13s} {alt + ' - ' + ref:34s} {measure:17s} {c['diff']:+9.3f}  [{lo:+8.3f}, {hi:+8.3f}]  {c['perm_p']:7.4f}  {c['mw_p']:6.4f}  {c['cohen_d']:+6.2f}")

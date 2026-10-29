"""Summary of experiments/expM_mri_us1.py (Google's MRI-US1 prompt), written 2026-10-27.

Direct-answer readout only (log P(" yes") - log P(" no") after "Final Answer:"); expM was run
without --generate. Run: python src/analyze_expM.py [--model medgemma]
"""
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import mcnemar_exact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "medgemma"
d = pd.concat(pd.read_csv(f) for f in glob.glob(str(ROOT / "results" / "expM_mri_us1" / f"{model}_*.csv")))
# 'gt', 'hist' and 'empty' clash with DataFrame attributes; rename, and keep "" histories as "empty"
d = d.rename(columns={"gt": "truth", "hist": "hh"})
d["hh"] = d["hh"].fillna("empty")
d["em_side"] = d["em_side"].fillna("-")
H = ["empty", "knee", "pain", "suspected", "known_em"]

em = d[d.label == "em"]
print("=== P(yes) > P(no) for 'an ovarian endometrioma' (direct answer) ===")
for abl, truth in [("real", "negative or cyst"), ("real", "endometrioma"), ("blank", "all")]:
    g = em[em.abl == abl]
    if truth == "endometrioma":
        g = g[g.truth == "endometrioma"]
    elif truth != "all":
        g = g[g.truth != "endometrioma"]
    t = g.pivot_table(index="layout", columns="hh", values="logit_yes_no",
                      aggfunc=lambda x: f"{(x > 0).sum()}/{len(x)}")
    print(f"-- {abl} volumes, {truth}")
    print(t[H].to_string())

neg = em[(em.abl == "real") & (em.truth != "endometrioma")]
print("\n=== em-negative real volumes: mean logit ===")
print(neg.pivot_table(index="layout", columns="hh", values="logit_yes_no").round(2)[H].to_string())
print("\n=== known history vs no history / knee, em-negative real volumes (paired) ===")
for lay, g in neg.groupby("layout"):
    w = g.pivot_table(index="subject", columns="hh", values="logit_yes_no")
    e, k, kn = w["empty"], w["known_em"], w["knee"]
    b, c = int(((e <= 0) & (k > 0)).sum()), int(((e > 0) & (k <= 0)).sum())
    bk, ck = int(((kn <= 0) & (k > 0)).sum()), int(((kn > 0) & (k <= 0)).sum())
    print(f"  {lay:8s} logit up with known {int((k > e).sum())}/{len(w)} (mean {(k - e).mean():+.2f}); "
          f"no->yes {b} of {int((e <= 0).sum())}, yes->no {c}, McNemar p={mcnemar_exact(b, c):.2g}; "
          f"known vs knee discordant {bk} vs {ck}, p={mcnemar_exact(bk, ck):.2g}")

print("\n=== side: 'a left ...' vs 'a right ovarian endometrioma' (real volumes) ===")
s = d[(d.abl == "real") & d.label.isin(["em_left", "em_right"])].pivot_table(
    index=["layout", "hh", "subject", "truth", "em_side"], columns="label", values="logit_yes_no").reset_index()
s["lr"] = s.em_left - s.em_right
for lay, g in s.groupby("layout"):
    e = g[g.truth == "endometrioma"]
    print(f"  {lay:8s} logit(left) > logit(right) in {int((g.lr > 0).sum())}/{len(g)} reads; "
          f"'yes' to both sides {int(((g.em_left > 0) & (g.em_right > 0)).sum())}/{len(g)}; "
          f"endometrioma cases, mean left-right by true side: "
          + ", ".join(f"{k} {v.lr.mean():+.2f} (n={len(v)})" for k, v in e.groupby("em_side")))

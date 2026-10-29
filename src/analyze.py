"""Aggregate experiment shards and print the summary numbers used in FINDINGS.md / the write-up.

Run: python src/analyze.py [--model medgemma|qwen] [exp ...]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stats import bootstrap_ci, cluster_bootstrap_prop, fmt_prop, mcnemar_exact, wilson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_orient import EXCLUDED_SUBJECTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


MODEL = "medgemma"  # set by --model; every experiment loader reads only this model's CSVs


# Input identity of one row per experiment (PATCH.md, P11). Sharded, unsharded and reverse-order
# workers write the same rows; rows are de-duplicated on this key, and a duplicate key whose other
# columns differ is reported (never silently kept or dropped).
KEYS = {
    "exp00_baseline": ["subject", "seq"],
    "expC_laterality": ["cohort", "subject", "seq", "geom", "conv"],
    "expC2_synthetic_side": ["subject", "cond"],
    "expC3_side_prompt": ["subject", "source", "gt_side", "variant"],
    "expC4_psychometric": ["subject", "radius_mm", "contrast", "dist_mm", "side"],
    "expR_freetext_report": ["cohort", "subject", "geom", "hist"],
    "expB_history": ["subject", "seq", "abl", "hist"],
    "expB2_history_dose": ["cohort", "subject", "seq", "abl", "hist"],
    "expB3_history_x_paraphrase": ["subject", "para", "hist"],
    "expB4_history_controls": ["subject", "part", "image", "hist", "q"],
    "expP_positive_controls": ["part", "subject", "cond"],
    "expA3_type_order": ["subject", "seq", "rot"],
    "expX_cxr_laterality": ["image", "source", "cond", "fmt", "truth"],
}
# Columns whose value legitimately varies between identical inputs (timing, provenance)
VOLATILE = {"secs"}


def load(exp_dir: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Rows of one experiment for the selected MODEL, without EXCLUDED_SUBJECTS (P01)."""
    # every pattern names the model as "medgemma_*" or "medgemma.csv"; substitute the selected
    # model (the expX pattern used to be "medgemma*", which never matched another model, P12)
    pattern = pattern.replace("medgemma_", f"{MODEL}_").replace("medgemma.csv", f"{MODEL}.csv")
    files = sorted(glob.glob(str(RES / exp_dir / pattern)))
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "subject" in d:
        d = d[~d.subject.isin(EXCLUDED_SUBJECTS)]
    key = [k for k in KEYS.get(exp_dir, []) if k in d]
    if not key:
        return d.drop_duplicates().reset_index(drop=True)
    cols = [c for c in d.columns if c not in VOLATILE]
    full = d.drop_duplicates(subset=cols)
    conflict = full[full.duplicated(subset=key, keep=False)]
    if len(conflict):
        print(f"  WARNING {exp_dir}: {conflict[key].drop_duplicates().shape[0]} input keys have rows that differ; "
              f"keeping the first. Example: {conflict[key].iloc[0].to_dict()}")
    return full.drop_duplicates(subset=key).reset_index(drop=True)


def exp00():
    d = load("exp00_baseline", "medgemma_*.csv")
    if d.empty:
        return print("exp00: no data")
    print("\n=== exp00 baseline (MedGemma, reference pipeline, D2) ===")
    print(f"  side answers 'left' overall: {fmt_prop(int((d.side == 'left').sum()), len(d))} "
          f"({d.subject.nunique()} patients)")
    print(f"  score vs generated agreement: {fmt_prop(int(((d.logit_yes_no > 0) == (d.endo == 'yes')).sum()), len(d))}")
    for seq, g in d.groupby("seq"):
        print(f"-- {seq}")
        for lab, gg in g.groupby("label"):
            k = (gg.endo == "yes").sum()
            print(f"   endometrioma=yes | {lab:13s}: {fmt_prop(k, len(gg))}  unparsed={int((gg.endo == 'unparsed').sum())}")
        print("   side answers:", g.side.value_counts().to_dict())
        em = g[g.label == "endometrioma"]
        if len(em):
            print("   side vs mask (em+):", list(zip(em.subject, em.em_side, em.side)))


def expA():
    d = load("expA_sequence", "medgemma_*.csv")
    if d.empty:
        return print("expA: no data")
    print("\n=== expA sequence: pixels x name ===")
    print(d.pivot_table(index=["label"], columns=["pix", "name"], values="logit_yes_no", aggfunc="mean").round(2))
    if "endo" in d:
        print(d.assign(yes=(d.endo == "yes")).pivot_table(index="label", columns=["pix", "name"], values="yes",
                                                           aggfunc="mean").round(2))
        print(d.assign(em=(d.lesion_type == "endometrioma")).pivot_table(index="label", columns=["pix", "name"],
                                                                          values="em", aggfunc="mean").round(2))
        # Rationale faithfulness: which sequence does the stated "low signal on Tx" criterion name?
        d["low_on"] = d.endo_raw.astype(str).str.extract(r"low signal intensity on (T1|T2)-weighted")[0]
        d["named"] = d.described_as.str.extract(r"^(T1|T2)")[0]
        x = d[d.low_on.notna() & d.named.notna()]
        print(f"  rationale 'low signal on Tx': Tx == named sequence {fmt_prop(int((x.low_on == x.named).sum()), len(x))}; "
              f"Tx == shown pixels {fmt_prop(int((x.low_on == x.pix.str[:2]).sum()), len(x))}")
        print("lesion_type counts by label/pix/name:")
        print(d.groupby(["label", "pix", "name"]).lesion_type.value_counts().unstack(fill_value=0))


def expB():
    d = load("expB_history", "medgemma_*.csv")
    if d.empty:
        return print("expB: no data")
    print("\n=== expB history: mean logit(yes)-logit(no) and P(yes>no) ===")
    d["yes"] = d.logit_yes_no > 0
    d["neg"] = np.where(d.label == "endometrioma", "em+", "em-")
    print(d.pivot_table(index=["abl", "neg"], columns="hist", values="logit_yes_no", aggfunc="mean").round(2))
    print(d.pivot_table(index=["abl", "neg"], columns="hist", values="yes", aggfunc="mean").round(2))
    for abl in ["real", "blank", "noise", "text"]:
        g = d[(d.abl == abl) & (d["hist"] == "known_em")]
        print(f"  known_em + {abl:5s}: direct 'yes' {fmt_prop(int(g.yes.sum()), len(g))}")
    # paired flip vs no-history on real images, em-negative cases
    r = d[(d.abl == "real") & (d.neg == "em-")].pivot_table(index="subject", columns="hist", values="yes")
    for h in r.columns:
        if h == "none":
            continue
        b = int(((r["none"] == 0) & (r[h] == 1)).sum())
        c = int(((r["none"] == 1) & (r[h] == 0)).sum())
        print(f"  em- real: none->{h}: answers changed {int(b + c)}/{len(r)}; FP {int(r['none'].sum())}->{int(r[h].sum())} of {len(r)}; "
              f"new FP={b}, lost FP={c}, McNemar p={mcnemar_exact(b, c):.2g}")


def expC():
    d = load("expC_laterality", "medgemma_*.csv")
    if d.empty:
        return print("expC: no data")
    if "cohort" not in d:
        d["cohort"] = "D2"
    for cohort, dc in d.groupby(d.cohort.fillna("D2")):
        print(f"\n=== expC laterality, cohort {cohort} ===")
        _expC(dc)


def _expC(d):
    col = "side" if "side" in d else "side_argmax"
    print(f"  all answers 'left': {fmt_prop(int((d[col] == 'left').sum()), len(d))}; unparsed {int((d[col] == 'unparsed').sum())}"
          + (f" (truncated {int(d.truncated.fillna(False).astype(bool).sum())})" if "truncated" in d else ""))
    o = d[(d.geom == "orig") & (d.gt_side == "right")].drop_duplicates("subject")
    print(f"  right-sided lesions (orig) called left: {int((o[col] == 'left').sum())}/{len(o)}")
    col = "side" if "side" in d else "side_argmax"
    for c in [col, "side_argmax"] if col != "side_argmax" else [col]:
        print(f"-- answer source: {c}")
        print(d.groupby(["geom", "conv"])[c].value_counts().unstack(fill_value=0))
        w = d.pivot_table(index=["subject", "conv"], columns="geom", values=c, aggfunc="first")
        flip = {"left": "right", "right": "left"}
        lat = w[w["orig"].isin(["left", "right"])]
        eq = (lat["lr"] == lat["orig"].map(flip))
        same = (lat["lr"] == lat["orig"])
        unp = (lat["lr"] == "unparsed")
        print(f"   LR equivariance (orig answer lateral): flipped correctly {fmt_prop(int(eq.sum()), len(lat))}; "
              f"unchanged {fmt_prop(int(same.sum()), len(lat))}; mirrored answer unparsed {int(unp.sum())}"
              + (f" -> excluding those: flipped {fmt_prop(int(eq.sum()), int((~unp).sum()))}" if unp.any() else ""))
        if "ap" in w:
            apsame = (lat["ap"] == lat["orig"])
            print(f"   AP control (should be unchanged): {fmt_prop(int(apsame.sum()), len(lat))}")
        gt = d[(d.geom == "orig") & d.gt_side.isin(["left", "right"])]
        for conv, g in gt.groupby("conv"):
            print(f"   accuracy vs mask side, orig, conv={conv}: {fmt_prop(int((g[c] == g.gt_side).sum()), len(g))}")
        gtl = d[(d.geom == "lr") & d.gt_side.isin(["left", "right"])]
        for conv, g in gtl.groupby("conv"):
            exp_side = g.gt_side.map(flip)
            print(f"   accuracy vs mask side, LR-mirrored, conv={conv}: {fmt_prop(int((g[c] == exp_side).sum()), len(g))}")


def expD():
    d = load("expD_slices", "medgemma_*.csv")
    if d.empty:
        return print("expD: no data")
    print("\n=== expD slices: mean logit yes-no ===")
    print(d.pivot_table(index="label", columns="cond", values="logit_yes_no", aggfunc="mean").round(2))
    d["side"] = d[["lp_left", "lp_right", "lp_bilateral", "lp_none"]].idxmax(axis=1).str[3:]
    print(d.groupby("cond").side.value_counts().unstack(fill_value=0))


def expE():
    d = load("expE_invariance", "medgemma_*.csv")
    if d.empty:
        return print("expE: no data")
    print("\n=== expE invariance: mean logit yes-no, and % yes ===")
    print(d.pivot_table(index="label", columns="cond", values="logit_yes_no", aggfunc="mean").round(2))
    print(d.assign(yes=d.logit_yes_no > 0).pivot_table(index="label", columns="cond", values="yes", aggfunc="mean").round(2))
    w = d.pivot_table(index="subject", columns="cond", values="logit_yes_no")
    ref = w["ref"] > 0
    for c in w.columns:
        if c != "ref":
            print(f"   flips vs ref: {c:18s} {fmt_prop(int(((w[c] > 0) != ref).sum()), len(w))}")


def expF():
    d = load("expF_pushback", "medgemma_*.csv")
    if d.empty:
        return print("expF: no data")
    print("\n=== expF pushback ===")
    pb = d[d.kind == "pushback"]
    print(f"  pushback overall: flipped {fmt_prop(int((pb.turn2 != pb.turn1).sum()), len(pb))}")
    pb = pb.assign(truth=np.where(pb.label == "endometrioma", "yes", "no"))
    ok = pb[pb.turn1 == pb.truth]
    bad = pb[pb.turn1 != pb.truth]
    print(f"  correct answers abandoned {fmt_prop(int((ok.turn2 != ok.turn1).sum()), len(ok))}; "
          f"wrong answers corrected {fmt_prop(int((bad.turn2 != bad.turn1).sum()), len(bad))}")
    for (kind, t1), g in d.dropna(subset=["kind"]).groupby(["kind", "turn1"]):
        print(f"  {kind:8s} turn1={t1}: flipped {fmt_prop(int((g.turn2 != g.turn1).sum()), len(g))}")


def expN():
    d = load("expN_noise", "medgemma_*.csv")
    if d.empty:
        return print("expN: no data")
    print("\n=== expN noise floors ===")
    p = d[d.kind == "para_endo"].pivot_table(index="subject", columns="idx", values="logit_yes_no")
    yes = p > 0
    print("  endo paraphrase: % yes per paraphrase", yes.mean().round(2).to_dict())
    print("  endo paraphrase: fraction of cases whose answer differs from paraphrase 0, per paraphrase",
          {k: round(float((yes[k] != yes[0]).mean()), 2) for k in yes.columns if k != 0})
    s = d[d.kind == "para_side"].pivot_table(index="subject", columns="idx", values="side_argmax", aggfunc="first")
    for k in s.columns:
        print(f"  side paraphrase {k}: 'left' {fmt_prop(int((s[k] == 'left').sum()), len(s))}")
    print("  side paraphrase answers:", {k: s[k].value_counts().to_dict() for k in s.columns})
    for kind in ["sample_endo", "sample_side"]:
        g = d[d.kind == kind]
        if len(g):
            agree = g.groupby("subject").answer.agg(lambda a: a.value_counts().iloc[0] / len(a))
            dis = g.groupby("subject").answer.agg(lambda a: a.nunique() > 1)
            if kind == "sample_side":
                print(f"  sample_side: answers 'left' {int((g.answer == 'left').sum())}/{len(g)}")
            print(f"  {kind}: cases with any disagreement across seeds {fmt_prop(int(dis.sum()), len(dis))}; "
                  f"answers {g.answer.value_counts().to_dict()} (n={len(g)})")
            print(f"  {kind}: mean modal-answer agreement across seeds = {agree.mean():.2f} (n={len(agree)} cases)")


def expA2():
    d = load("expA2_contrast", "medgemma_*.csv")
    if d.empty:
        return print("expA2: no data")
    print("\n=== expA2 SYNTHETIC lesion contrast (T1FS) ===")
    print(d.pivot_table(index="label", columns="cond", values="logit_yes_no", aggfunc="mean").round(2))
    w = d.pivot_table(index=["subject", "label"], columns="cond", values="logit_yes_no").reset_index()
    diff = (w.lesion_bright - w.lesion_dark).values
    print(f"  paired logit(bright) - logit(dark): mean {diff.mean():.2f}, median {np.median(diff):.2f}, "
          f"range [{diff.min():.2f}, {diff.max():.2f}], n={len(diff)} lesions")
    if "endo" in d:
        y = d.assign(yes=d.endo == "yes").pivot_table(index="subject", columns="cond", values="yes")
        b = int(((y.lesion_dark == 0) & (y.lesion_bright == 1)).sum())
        c = int(((y.lesion_dark == 1) & (y.lesion_bright == 0)).sum())
        print(f"  generated yes: bright {fmt_prop(int(y.lesion_bright.sum()), len(y))}; dark {fmt_prop(int(y.lesion_dark.sum()), len(y))}; "
              f"ref {fmt_prop(int(y.ref.sum()), len(y))}; filled {fmt_prop(int(y.lesion_filled.sum()), len(y))}")
        print(f"  discordant dark->bright: became yes {b}, became no {c}, McNemar p={mcnemar_exact(b, c):.2g}")
        print("  lesion_type by cond:", d.groupby("cond").lesion_type.value_counts().unstack(fill_value=0).to_dict("index"))


def expA3():
    d = load("expA3_type_order", "medgemma_*.csv")
    if d.empty:
        return print("expA3: no data")
    print("\n=== expA3 lesion-type option rotation ===")
    print(pd.crosstab([d.seq, d.first_option], d.answer))
    k = int((d.answer_position == 0).sum())
    print(f"  answer == first-listed option: {fmt_prop(k, len(d))}")
    opts = ["endometrioma", "simple cyst", "dermoid", "hemorrhagic cyst", "none"]
    fnn = d.rot.apply(lambda r: [o for o in opts[r:] + opts[:r] if o != "none"][0])
    print(f"  answer == first non-'none' option: {fmt_prop(int((d.answer == fnn).sum()), len(d))}")
    k = int((d.answer == "endometrioma").sum())
    print(f"  answer == endometrioma (any position): {fmt_prop(k, len(d))}")


def expB2(cohort: str = "D2"):
    pattern = "medgemma_T2FS_*.csv" if cohort == "D2" else f"medgemma_{cohort}_*.csv"
    d = load("expB2_history_dose", pattern)
    if d.empty:
        return print(f"expB2 {cohort}: no data")
    print(f"\n=== expB2 history dose / paraphrase / generated confirmation, cohort {cohort} ===")
    d["grp"] = np.where(d.label == "endometrioma", "em+", "em-")
    order = ["none", "possible", "suspected", "history_endo", "known_em", "known_large_bilat",
             "known_em_p1", "known_em_p2", "known_em_p3", "resected"]
    print(d.pivot_table(index=["abl", "grp"], columns="hist", values="logit_yes_no", aggfunc="mean")[
        [h for h in order if h in set(d["hist"])]].round(2))
    g = d[(d.abl == "real") & d.endo.notna()]
    print("  generated 'yes' rate (real image):")
    for (grp, h), gg in g.groupby(["grp", "hist"]):
        print(f"    {grp} {h:12s} {fmt_prop(int((gg.endo == 'yes').sum()), len(gg))}  unparsed={int((gg.endo == 'unparsed').sum())}")
    w = g[g.grp == "em-"].pivot_table(index="subject", columns="hist", values="endo", aggfunc="first")
    for h in [c for c in w.columns if c != "none"]:
        b = int(((w["none"] == "no") & (w[h] == "yes")).sum())
        c = int(((w["none"] == "yes") & (w[h] == "no")).sum())
        n_no = int((w["none"] == "no").sum())
        print(f"    em- paired none->{h}: of {n_no} 'no' at baseline, {b} flipped to yes "
              f"({fmt_prop(b, n_no)}); reverse flips {c}; McNemar p={mcnemar_exact(b, c):.2g}")


def expC2():
    d = load("expC2_synthetic_side", "medgemma_*.csv")
    if d.empty:
        return print("expC2: no data")
    print("\n=== expC2 SYNTHETIC cyst with known side ===")
    col = "side" if "side" in d else "side_argmax"
    print(pd.crosstab(d.cond, d[col]))
    d["gap"] = d.lp_right - d.lp_left
    print(d.groupby("cond")[["gap", "cyst_logit"]].agg(["mean", "std"]).round(2))
    w = d.pivot_table(index="subject", columns="cond", values="gap")
    for c in ["left", "right", "bilateral"]:
        x = (w[c] - w["none"]).values
        print(f"  paired gap shift {c} - none: mean {x.mean():.3f} (sd {x.std():.3f}), n={len(x)}")
    r = d[d.cond == "right"]
    print(f"  right-inserted cyst answered 'right': {fmt_prop(int((r[col] == 'right').sum()), len(r))}; "
          f"unparsed {int((r[col] == 'unparsed').sum())} (excluding unparsed: "
          f"{fmt_prop(int((r[col] == 'right').sum()), int((r[col] != 'unparsed').sum()))})")


def expC3():
    d = load("expC3_side_prompt", "medgemma_*.csv")
    if d.empty:
        return print("expC3: no data")
    print("\n=== expC3 side mechanism (prompt variants) ===")
    print(pd.crosstab([d.variant, d.gt_side], d.answer))
    for v, g in d[~d.variant.str.startswith("yn_")].groupby("variant"):
        print(f"  {v:12s} correct side: {fmt_prop(int((g.answer == g.gt_side).sum()), len(g))} reads "
              f"({g.subject.nunique()} volumes); unparsed {int((g.answer == 'unparsed').sum())}")
    o = d[d.variant == "open"]
    if len(o) and {"lp_left", "lp_right"} <= set(o.columns):
        mass = np.exp(o.lp_left) + np.exp(o.lp_right)
        print(f"  open: P(left)+P(right) after 'Final Answer:' (case-summed if scored after P09): median {mass.median():.3f}")
    from scipy.stats import fisher_exact
    for src, g in [("real", d[(d.variant == "open") & (d.source != "synthetic")]),
                   ("synthetic", d[(d.variant == "open") & (d.source == "synthetic")])]:
        t = [[int(((g.gt_side == "right") & (g.answer == "right")).sum()), int(((g.gt_side == "right") & (g.answer != "right")).sum())],
             [int(((g.gt_side == "left") & (g.answer == "right")).sum()), int(((g.gt_side == "left") & (g.answer != "right")).sum())]]
        print(f"  open format, {src}: correct {fmt_prop(int((g.answer == g.gt_side).sum()), len(g))}; "
              f"right-sided called right {t[0][0]}/{sum(t[0])}; Fisher p={fisher_exact(t)[1]:.2g}")
    for v, g in d[d.variant.str.startswith("yn_")].groupby("variant"):
        side = v[3:]
        truth = np.where(g.gt_side == side, "yes", "no")
        print(f"  {v:12s} 'yes' rate {fmt_prop(int((g.answer == 'yes').sum()), len(g))}; "
              f"correct {fmt_prop(int((g.answer == truth).sum()), len(g))}")


def expD2():
    d = load("expD2_shuffle_confirm", "medgemma_*.csv")
    if d.empty:
        return print("expD2: no data")
    print("\n=== expD2 shuffled slices, generated answers ===")
    w = d.pivot_table(index=["subject", "label"], columns="cond", values="endo", aggfunc="first").reset_index()
    for s in [c for c in w.columns if str(c).startswith("shuffled")]:
        for lab, g in [("all", w), ("lesion-free", w[w.label == "negative"])]:
            b = int(((g.ref == "no") & (g[s] == "yes")).sum())
            c = int(((g.ref == "yes") & (g[s] == "no")).sum())
            print(f"  {s} {lab:11s}: yes ref {fmt_prop(int((g.ref == 'yes').sum()), len(g))} -> "
                  f"{fmt_prop(int((g[s] == 'yes').sum()), len(g))}; McNemar p={mcnemar_exact(b, c):.2g}")
    g = d[(d.label == "negative") & d.cond.isin(["ref", "short_block"])]
    for c, gg in g.groupby("cond"):
        print(f"  lesion-free {c:11s}: generated yes {fmt_prop(int((gg.endo == 'yes').sum()), len(gg))}; "
              f"logit>0 {fmt_prop(int((gg.logit_yes_no > 0).sum()), len(gg))}; mean logit {gg.logit_yes_no.mean():.2f}")


def expB3():
    d = load("expB3_history_x_paraphrase", "medgemma_*.csv")
    if d.empty:
        return print("expB3: no data")
    print("\n=== expB3 history x question paraphrase (em-negative volumes) ===")
    n = d[d.label != "endometrioma"]
    print("  direct-answer 'yes' rate:")
    print(n.assign(yes=n.logit_yes_no > 0).pivot_table(index="para", columns="hist", values="yes", aggfunc="mean").round(2))
    w = n.pivot_table(index=["subject", "para"], columns="hist", values="logit_yes_no").reset_index()
    for k, g in w.groupby("para"):
        up = int((g.known_em > g["none"]).sum())
        print(f"  para {k}: logit(known_em) > logit(none) in {fmt_prop(up, len(g))}; mean shift "
              f"{(g.known_em - g['none']).mean():+.2f}; resected shift {(g.resected - g['none']).mean():+.2f}")
    print(f"  pooled over paraphrases: logit(known_em) > logit(none) in {fmt_prop(int((w.known_em > w['none']).sum()), len(w))}")
    if "endo" in d:
        gen = n[n.endo.notna()].pivot_table(index=["subject", "para"], columns="hist", values="endo", aggfunc="first").reset_index()
        pooled_b = pooled_n = 0
        for k, g in gen.groupby("para"):
            base_no = g[g["none"] == "no"]
            b = int((base_no.known_em == "yes").sum())
            c = int(((g["none"] == "yes") & (g.known_em == "no")).sum())
            pooled_b += b
            pooled_n += len(base_no)
            print(f"  para {k} generated: of {len(base_no)} 'no' without history, {fmt_prop(b, len(base_no))} flip to yes "
                  f"with known_em; reverse flips {c}; McNemar p={mcnemar_exact(b, c):.2g}")
        print(f"  pooled generated (paraphrases 1-4): {fmt_prop(pooled_b, pooled_n)} baseline-'no' flip to yes")


def inventory():
    d = pd.read_csv(ROOT / "derived" / "inventory.csv")
    print("\n=== inventory ===")
    exc = d[d.excluded.notna() & (d.excluded != "")] if "excluded" in d else d.iloc[0:0]
    for _, r in exc.iterrows():
        print(f"  EXCLUDED {r.subject}: {r.excluded}")
    d = d[~d.subject.isin(exc.subject)]
    for cohort, g in d.groupby("cohort"):
        em = g[g.em_present == True]
        print(f"  {cohort}: subjects {len(g)}; endometrioma+ {len(em)} (sides, largest lesion: {em.em_side.value_counts().to_dict()}); "
              f"cyst+ {int((g.get('cy_present') == True).sum())}; no images {int((g.sequences.fillna('') == '').sum())}")
        if cohort == "D1":
            print(f"    D1 endometrioma+ by rule: union of raters {int((g.em_present_union == True).sum())}; "
                  f"majority of raters {int((g.em_present_majority == True).sum())} (primary rule: union)")
        for seq in ["T2FS", "T2"]:
            s = g[g.sequences.fillna("").str.split(";").apply(lambda x: seq in x)]
            em_, cy_ = s.em_present == True, s.get("cy_present", pd.Series(False, index=s.index)) == True
            print(f"    with {seq}: {len(s)} = endometrioma {int(em_.sum())} (of which also cyst {int((em_ & cy_).sum())}) "
                  f"+ cyst only {int((cy_ & ~em_).sum())} + neither {int((~em_ & ~cy_).sum())}")
    em = d[(d.cohort == "D2") & (d.em_present == True)]
    lat = em[em.em_side.isin(["left", "right"])]
    print(f"  D2 right-sided among lateralised endometriomas: {int((lat.em_side == 'right').sum())}/{len(lat)}")
    for s_ in ["T1", "T1FS", "T2", "T2FS"]:
        col = f"{s_}_identical_to"
        if col in d:
            x = d[d[col].fillna("") != ""]
            if len(x):
                print(f"  identical voxel data, {s_}: " + "; ".join(f"{a}={b}" for a, b in zip(x.subject, x[col])))
    tilt = d.filter(like="_tilt_deg").max(axis=1)
    print(f"  series tilted >20 deg from axial (model sees an oblique stack): {', '.join(d.subject[tilt > 20])}")


def parser():
    p = RES / "parser_audit_checked.csv"
    if not p.exists():
        return print("parser audit: not checked yet")
    a = pd.read_csv(p)
    print("\n=== parser audit ===")
    print(f"  hand label == parsed: {fmt_prop(int((a.hand_label == a.parsed).sum()), len(a))} "
          f"(checked by: {a.checked_by.iloc[0]})")


def _cl(label, success, cluster):
    p, lo, hi, n, k = cluster_bootstrap_prop(success, cluster)
    if p in (0.0, 1.0):  # bootstrap CI is degenerate at 0% / 100%: use Wilson over patients instead
        per = pd.Series(np.asarray(success, bool)).groupby(np.asarray(cluster)).all()
        print(f"  {label}: {fmt_prop(int(per.sum()), len(per))} patients (Wilson over patients; {n} reads)")
        return
    print(f"  {label}: {p:.0%} [patient-cluster bootstrap 95% CI {lo:.0%}-{hi:.0%}] ({n} reads, {k} patients)")


def patientlevel():
    """Headline proportions restated with one unit per patient (Wilson) or a patient-cluster bootstrap."""
    print("\n=== patient-level restatements (no pseudo-replication) ===")
    e = load("exp00_baseline", "medgemma_*.csv")
    if not e.empty:
        per = e.groupby("subject").side.agg(lambda a: (a == "left").all())
        print(f"  exp00 patients whose every read is 'left': {fmt_prop(int(per.sum()), len(per))}")
    c = load("expC_laterality", "medgemma_*.csv")
    if not c.empty:
        c = c[c.get("cohort", pd.Series("D2", index=c.index)).fillna("D2") == "D2"] if "cohort" in c else c
        w = c.pivot_table(index=["subject", "conv"], columns="geom", values="side", aggfunc="first").reset_index()
        w["equiv"] = w.lr == w.orig.map({"left": "right", "right": "left"})
        per = w.groupby("subject").equiv.any()
        print(f"  expC patients with an L<->R-equivariant answer under ANY convention text: {fmt_prop(int(per.sum()), len(per))}")
    c3 = load("expC3_side_prompt", "medgemma_*.csv")
    if not c3.empty:
        for v in ["ref", "order_rl", "open"]:
            g = c3[c3.variant == v]
            _cl(f"expC3 {v} correct side", g.answer == g.gt_side, g.subject)
        g = c3[c3.variant == "order_rl"]
        _cl("expC3 order_rl answer == 'right'", g.answer == "right", g.subject)
    a3 = load("expA3_type_order", "medgemma_*.csv")
    if not a3.empty:
        opts = ["endometrioma", "simple cyst", "dermoid", "hemorrhagic cyst", "none"]
        fnn = a3.rot.apply(lambda r: [o for o in opts[r:] + opts[:r] if o != "none"][0])
        _cl("expA3 answer == first non-'none' option", a3.answer == fnn, a3.subject)
    b3 = load("expB3_history_x_paraphrase", "medgemma_*.csv")
    if not b3.empty:
        n = b3[b3.label != "endometrioma"]
        w = n.pivot_table(index=["subject", "para"], columns="hist", values="logit_yes_no").reset_index()
        per = w.groupby("subject").apply(lambda g: (g.known_em > g["none"]).all())
        print(f"  expB3 em- patients with logit(known) > logit(none) under ALL 5 paraphrases: {fmt_prop(int(per.sum()), len(per))}")


def expP():
    d = load("expP_positive_controls", "medgemma_*.csv")
    if d.empty:
        return print("expP: no data")
    print("\n=== expP positive controls (open format unless stated) ===")
    for part, g in d.groupby("part"):
        if part == "P3":
            continue
        for cond, gg in g.groupby("cond"):
            print(f"  {part} {cond:12s}: correct {fmt_prop(int((gg.answer == gg.truth).sum()), len(gg))}; "
                  f"answers {gg.answer.value_counts().to_dict()}")
    p3 = d[d.part == "P3"]
    if len(p3):
        for cond, gg in p3.groupby("cond"):
            print(f"  P3 {cond:18s}: 'yes' {fmt_prop(int((gg.answer == 'yes').sum()), len(gg))}; "
                  f"correct {fmt_prop(int((gg.answer == gg.truth).sum()), len(gg))}")
        w = p3.assign(obj=p3.cond.str.split("_").str[0], q=p3.cond.str.split("_").str[1])
        w = w.pivot_table(index="subject", columns=["obj", "q"], values="answer", aggfunc="first")
        both = ((w[("present", "askpresent")] == "yes") & (w[("present", "askabsent")] == "no") &
                (w[("absent", "askpresent")] == "no") & (w[("absent", "askabsent")] == "yes"))
        print(f"  P3 volumes answered correctly in all 4 cells (both polarities, object present and absent): "
              f"{fmt_prop(int(both.sum()), len(both))}")
    p2 = d[d.part == "P2"]
    if len(p2):
        w = p2.pivot_table(index="subject", columns="cond", values="answer", aggfunc="first")
        flip = {"left": "right", "right": "left"}
        for fmt in ["open", "ref"]:
            o, l = w[f"{fmt}_orig"], w[f"{fmt}_lr"]
            lat = o.isin(["left", "right"])
            eq = int((l[lat] == o[lat].map(flip)).sum())
            print(f"  P2 {fmt}: L<->R equivariant {fmt_prop(eq, int(lat.sum()))} (of patients with a lateral original answer)")


def expX():
    d = load("expX_cxr_laterality", "medgemma.csv")
    if d.empty:
        return print("expX: no data")
    print("\n=== expX chest X-ray laterality (single PA image) ===")
    for (src, fmt), g in d.groupby(["source", "fmt"]):
        print(f"  {src:9s} {fmt:4s}: correct side {fmt_prop(int((g.answer == g.truth).sum()), len(g))}; "
              f"truth=right called right {fmt_prop(int(((g.truth == 'right') & (g.answer == 'right')).sum()), int((g.truth == 'right').sum()))}; "
              f"answers {g.answer.value_counts().to_dict()}")
    o = d[(d.source == "real") & (d.fmt == "open") & (d.cond == "orig")]
    print(f"  real, open, ORIGINAL films only (one read per finding): correct {fmt_prop(int((o.answer == o.truth).sum()), len(o))} "
          f"({o.image.nunique()} films)")
    from scipy.stats import fisher_exact
    t = [[int(((o.truth == "right") & (o.answer == "right")).sum()), int(((o.truth == "right") & (o.answer != "right")).sum())],
         [int(((o.truth == "left") & (o.answer == "right")).sum()), int(((o.truth == "left") & (o.answer != "right")).sum())]]
    print(f"  real, open, original: answer 'right' when truth right {t[0][0]}/{sum(t[0])} vs truth left {t[1][0]}/{sum(t[1])}; "
          f"Fisher p={fisher_exact(t)[1]:.2g}")
    r = d[d.source == "real"].copy()
    if len(r):
        # one row per FINDING (a film can have findings on both sides): identify a finding by its
        # side on the original film (P14)
        flipd = {"left": "right", "right": "left"}
        r["finding_side"] = np.where(r.cond == "orig", r.truth, r.truth.map(flipd))
        for fmt, g in r.groupby("fmt"):
            w = g.pivot_table(index=["image", "finding_side"], columns="cond", values="answer", aggfunc="first")
            lat = w["orig"].isin(["left", "right"])
            eq = int((w.lr[lat] == w.orig[lat].map(flipd)).sum())
            print(f"  real {fmt}: mirror-equivariant {fmt_prop(eq, int(lat.sum()))} findings "
                  f"({w.index.get_level_values(0)[lat].nunique()} films)")


def expR():
    d = load("expR_freetext_report", "medgemma_D2_*.csv")
    if d.empty:
        return print("expR: no data")
    print("\n=== expR free-text FINDINGS section (parsed) ===")
    import re as _re
    def max_repeat(t):
        sents = [x.strip() for x in _re.split(r"(?<=[.!?])\s+", str(t)) if x.strip()]
        return max(pd.Series(sents).value_counts()) if sents else 0
    d["degenerate"] = d.raw.apply(max_repeat) >= 5
    d["lesion_reported"] = d.lesion_sides != "none"
    if "truncated" not in d:  # pre-patch CSVs: no token count was recorded
        d["truncated"] = np.nan
    print(f"  max_new_tokens: {sorted(d.max_new_tokens.dropna().unique().tolist()) if 'max_new_tokens' in d else 'unknown (1024)'}")
    print(f"  degenerate repetition (a sentence repeated >=5 times): {fmt_prop(int(d.degenerate.sum()), len(d))}")
    if d.truncated.notna().any():
        print(f"  truncated (token budget ran out before end of turn): {fmt_prop(int(d.truncated.sum()), len(d))}; by arm: "
              f"{ {f'{g_}/{h}': int(g.truncated.sum()) for (g_, h), g in d.groupby(['geom', 'hist'])} }")
        print(f"  truncated reports among those parsed as 'no lesion': {int((d.truncated & ~d.lesion_reported).sum())}/"
              f"{int((~d.lesion_reported).sum())}")
    for (g_, h), g in d.groupby(["geom", "hist"]):
        print(f"  {g_:4s} {h:8s}: any ovarian/adnexal lesion reported {fmt_prop(int(g.lesion_reported.sum()), len(g))}")
    neg0 = d[(d.label != "endometrioma") & (d.geom == "orig")]
    w0 = neg0.pivot_table(index="subject", columns="hist", values="lesion_reported", aggfunc="first")
    b = int(((w0["none"] == False) & (w0.known_em == True)).sum())
    c = int(((w0["none"] == True) & (w0.known_em == False)).sum())
    print(f"  em- (orig): lesion reported none {fmt_prop(int(w0['none'].sum()), len(w0))} -> known "
          f"{fmt_prop(int(w0.known_em.sum()), len(w0))}; McNemar p={mcnemar_exact(b, c):.2g}")
    if d.truncated.notna().any():
        t0 = neg0.pivot_table(index="subject", columns="hist", values="truncated", aggfunc="first").astype(bool)
        ok = ~(t0["none"] | t0.known_em)
        w1 = w0[ok]
        b1 = int(((w1["none"] == False) & (w1.known_em == True)).sum())
        c1 = int(((w1["none"] == True) & (w1.known_em == False)).sum())
        print(f"  em- (orig), pairs with neither report truncated: lesion reported none {fmt_prop(int(w1['none'].sum()), len(w1))} "
              f"-> known {fmt_prop(int(w1.known_em.sum()), len(w1))}; McNemar p={mcnemar_exact(b1, c1):.2g}")
    lat_all = d[d.lesion_sides.isin(["left", "right"])]
    print(f"  all lateral lesion mentions: {lat_all.lesion_sides.value_counts().to_dict()} (by geom: "
          f"{lat_all.groupby('geom').lesion_sides.value_counts().to_dict()})")
    for (g_, h), g in d.groupby(["geom", "hist"]):
        print(f"  {g_:4s} {h:8s}: endometrioma reported {fmt_prop(int(g.em_pos.sum()), len(g))}; "
              f"lesion sides {g.lesion_sides.value_counts().to_dict()}")
    neg = d[(d.label != "endometrioma") & (d.geom == "orig")]
    w = neg.pivot_table(index="subject", columns="hist", values="em_pos", aggfunc="first")
    b = int(((w["none"] == False) & (w.known_em == True)).sum())
    c = int(((w["none"] == True) & (w.known_em == False)).sum())
    print(f"  em- (orig): endometrioma reported none {fmt_prop(int(w['none'].sum()), len(w))} -> known "
          f"{fmt_prop(int(w.known_em.sum()), len(w))}; McNemar p={mcnemar_exact(b, c):.2g}")
    o = d[d["hist"] == "none"].pivot_table(index="subject", columns="geom", values="lesion_sides", aggfunc="first")
    lat = o["orig"].isin(["left", "right"])
    eq = int((o.lr[lat] == o["orig"][lat].map({"left": "right", "right": "left"})).sum())
    same = int((o.lr[lat] == o["orig"][lat]).sum())
    print(f"  mirror (no history): reports with a single lateral side {int(lat.sum())}/{len(o)}; side swapped "
          f"{fmt_prop(eq, int(lat.sum()))}; side unchanged {fmt_prop(same, int(lat.sum()))}")
    k = d[d["hist"] == "known_em"].pivot_table(index="subject", columns="geom", values="lesion_sides", aggfunc="first")
    both_lat = k["orig"].isin(["left", "right"]) & k.lr.isin(["left", "right"])
    print(f"  mirror (known history): side unchanged under L<->R mirroring {int((k.lr[both_lat] == k['orig'][both_lat]).sum())}/{int(both_lat.sum())} "
          f"(volumes with a single lateral side in both reports)")
    for g_, gg in [("all", lat_all), ("orig", lat_all[lat_all.geom == "orig"]), ("lr", lat_all[lat_all.geom == "lr"])]:
        print(f"  lateral lesion mentions = 'left' ({g_}): {int((gg.lesion_sides == 'left').sum())}/{len(gg)}")
    gtl = d[(d["hist"] == "none") & (d.geom == "orig") & d.gt_side.isin(["left", "right"])]
    print(f"  real lateral lesions (orig, no history) NOT reported at all: {int((gtl.lesion_sides == 'none').sum())}/{len(gtl)}"
          + (f" (of which truncated: {int((gtl.truncated & (gtl.lesion_sides == 'none')).sum())})" if gtl.truncated.notna().any() else ""))
    print(f"  real lateral lesions (orig, no history): report side matches mask "
          f"{fmt_prop(int((gtl.lesion_sides == gtl.gt_side).sum()), len(gtl))}; sides {gtl.lesion_sides.value_counts().to_dict()}")


def expB4():
    d = load("expB4_history_controls", "medgemma_*.csv")
    if d.empty:
        return print("expB4: no data")
    print("\n=== expB4 failure-2 controls (generated answers) ===")
    neg = d[d.label != "endometrioma"]
    irr = neg[neg.part.isin(["irrelevant", "known"])].pivot_table(index="subject", columns="hist", values="answer", aggfunc="first")
    b2 = load("expB2_history_dose", "medgemma_T2FS_*.csv")
    if len(b2):
        kn = b2[(b2.abl == "real") & (b2["hist"] == "known_em") & b2.endo.notna()].set_index("subject").endo
        irr["known_em"] = kn.reindex(irr.index)
    # without expB2 (e.g. the 2026-10-26 overnight run), expB4's own 'known' part supplies known_em
    # (same history text as expB2)
    if "known_em" not in irr:
        raise SystemExit("expB4: no known_em answers (neither expB2 nor expB4 'irrelevant' part)")
    base_no = irr[irr["none"] == "no"]
    base_yes = irr[irr["none"] == "yes"]
    for h in ["knee", "pain", "known_em"]:
        b, c = int((base_no[h] == "yes").sum()), int((base_yes[h] == "no").sum())
        print(f"  em- read 'no' without history -> 'yes' with {h:8s}: {fmt_prop(b, len(base_no))}; "
              f"read 'yes' -> 'no': {fmt_prop(c, len(base_yes))}; McNemar vs no history p={mcnemar_exact(b, c):.2g}")
    for h in ["knee", "pain"]:
        b = int(((base_no[h] == "no") & (base_no.known_em == "yes")).sum())
        c = int(((base_no[h] == "yes") & (base_no.known_em == "no")).sum())
        print(f"    known_em vs {h} among the same baseline-no volumes: discordant {b} vs {c}, McNemar p={mcnemar_exact(b, c):.2g}")
    doses = neg[neg.part == "doses"]
    print("  generated 'yes' by dose (em-):", {h: fmt_prop(int((g.answer == "yes").sum()), len(g)) for h, g in doses.groupby("hist")})
    bl = d[d.part == "blanks"]
    for (img, h), g in bl.groupby(["image", "hist"]):
        u = g.drop_duplicates("input_sha")  # identical inputs give identical answers: count once
        print(f"  blank {img:15s} {h:8s}: generated 'yes' {fmt_prop(int((g.answer == 'yes').sum()), len(g))} rows; "
              f"distinct inputs {len(u)}, 'yes' {fmt_prop(int((u.answer == 'yes').sum()), len(u))}; "
              f"direct 'yes' {fmt_prop(int((g.logit_yes_no > 0).sum()), len(g))}")
    ind = d[d.part == "indeterminate"]
    if len(ind):
        print(f"  'cannot assess' chosen: {int((ind.answer == 'cannot assess').sum())}/{len(ind)}")
        z = ind[(ind.image == "zeros_matched") & (ind["hist"] == "known_em")]
        print(f"  blank + known history, with 'cannot assess' offered: 'yes' {int((z.answer == 'yes').sum())}/{len(z)}")
    for (img, h, q), g in ind.groupby(["image", "hist", "q"]):
        print(f"  indeterminate {img:13s} {h:8s} {q}: {g.answer.value_counts().to_dict()} (n={len(g)})")


def expC4():
    d = load("expC4_psychometric", "medgemma_*.csv")
    if d.empty:
        return print("expC4: no data")
    print("\n=== expC4 psychometric sweep (open format; correct-direction shift of the side log-odds) ===")
    print(d.pivot_table(index=["contrast", "radius_mm"], columns="dist_mm", values="correct_dir", aggfunc="mean").round(2))
    print(d.pivot_table(index=["contrast", "radius_mm"], columns="side", values="shift", aggfunc="mean").round(2))
    if "achieved" in d:
        print("  contrast reached inside the sphere (fraction of target; P08):")
        print(d.pivot_table(index="contrast", columns="radius_mm", values="achieved", aggfunc="mean").round(2))
    if "mass_lr" in d:
        print(f"  P(left)+P(right) after 'Final Answer:' (case-summed): median {d.mass_lr.median():.3f}; "
              f"lowercase-only correct direction (pre-patch scoring) "
              f"{((d.shift_lower > 0) == (d.side == 'right')).mean():.2f} vs case-summed {d.correct_dir.mean():.2f}")
        print("  correct direction by contrast x radius, all distances pooled (case-summed):")
        print(d.pivot_table(index="contrast", columns="radius_mm", values="correct_dir", aggfunc="mean").round(2))
    g = d[d.answer.notna()] if "answer" in d else d.iloc[0:0]
    for r, gg in g.groupby("radius_mm"):
        print(f"  generated open answer, bright 45 mm, radius {r:2d} mm: correct {fmt_prop(int((gg.answer == gg.side).sum()), len(gg))}; "
              f"{gg.answer.value_counts().to_dict()}")


def proxy():
    """Is the direct-answer logit a valid proxy for the generated answer? (rigor review item 3)"""
    from scipy.stats import spearmanr
    b2 = load("expB2_history_dose", "medgemma_T2FS_*.csv")
    b4 = load("expB4_history_controls", "medgemma_*.csv")
    if b2.empty or b4.empty:
        return print("proxy: no data")
    print("\n=== proxy: direct-answer logit vs generated answer (em- volumes, real images) ===")
    a = b2[(b2.abl == "real") & (b2.label != "endometrioma") & b2.endo.notna()][["subject", "hist", "logit_yes_no", "endo"]]
    b = b4[(b4.part.isin(["irrelevant", "doses"])) & (b4.image == "real")][["subject", "hist", "logit_yes_no", "answer"]]
    b = b.rename(columns={"answer": "endo"})
    d = pd.concat([a, b]).drop_duplicates(["subject", "hist"])
    agg = d.groupby("hist").agg(mean_logit=("logit_yes_no", "mean"), gen_yes=("endo", lambda x: (x == "yes").mean()),
                                n=("endo", "size")).sort_values("mean_logit")
    print(agg.round(2).to_string())
    rho = spearmanr(agg.mean_logit, agg.gen_yes)
    # no p-value: the conditions share the same volumes and are not independent observations (P13)
    print(f"  across {len(agg)} history conditions: Spearman rho(mean logit, generated yes-rate) = {rho.statistic:.2f} "
          f"(descriptive; conditions share the same {d.subject.nunique()} volumes)")
    w = d.pivot_table(index="subject", columns="hist", values="logit_yes_no")
    g = d.pivot_table(index="subject", columns="hist", values="endo", aggfunc="first")
    flips = (g["none"] == "no") & (g.known_em == "yes")
    up = (w.known_em > w["none"])
    print(f"  per volume: logit rises none->known in {fmt_prop(int(up.sum()), len(up))}; among the {int(flips.sum())} generated "
          f"no->yes flips, logit rose in {fmt_prop(int((up & flips).sum()), int(flips.sum()))}")
    same = ((d.logit_yes_no > 0) == (d.endo == "yes"))
    print(f"  per read: direct-answer sign == generated answer {fmt_prop(int(same.sum()), len(same))} "
          f"(the direct answer is biased toward 'no'; generation adds a reasoning step first)")


def confirm_D1():
    """Pre-registered confirmatory test of failure 2 on D1 (see NOTEBOOK.md, PRE-REGISTRATION)."""
    from scipy.stats import binomtest
    d = load("expB2_history_dose", "medgemma_D1_*.csv").drop_duplicates(["subject", "abl", "hist"])
    if d.empty:
        return print("confirm_D1: no data")
    print("\n=== CONFIRMATORY (pre-registered): failure 2 on D1 T2 ===")
    r = d[d.abl == "real"].pivot_table(index="subject", columns="hist", values="logit_yes_no")
    up = int((r.known_em > r["none"]).sum())
    print(f"  H1: logit(known) > logit(none) in {up}/{len(r)} volumes; one-sided sign test "
          f"p={binomtest(up, len(r), 0.5, alternative='greater').pvalue:.2g}")
    g = d[(d.abl == "real") & d.endo.notna()].pivot_table(index="subject", columns="hist", values="endo", aggfunc="first")
    lab = d.drop_duplicates("subject").set_index("subject").label
    for grp, idx in [("all", list(g.index)), ("em- (no em label file): FALSE-POSITIVE test", [x for x in g.index if lab[x] == "negative"]),
                     ("em+ ('yes' is correct)", [x for x in g.index if lab[x] == "endometrioma"])]:
        gg = g.loc[idx]
        base_no = gg[gg["none"] == "no"]
        b = int((base_no.known_em == "yes").sum())
        c = int(((gg["none"] == "yes") & (gg.known_em == "no")).sum())
        print(f"  H2 {grp}: baseline-'no' flipped to 'yes' {fmt_prop(b, len(base_no))}; reverse {c}; McNemar p={mcnemar_exact(b, c):.2g}")


def _headline_one(model: str) -> dict:
    """Headline metrics for one model, computed identically for every model (rigor: same code)."""
    from scipy.stats import fisher_exact

    def ld(exp, pattern):
        global MODEL
        saved, MODEL = MODEL, model
        try:  # same loader (exclusions, de-duplication) as every other section
            return load(exp, pattern.replace("MODEL", "medgemma"))
        finally:
            MODEL = saved

    out = {"model": model}
    flip = {"left": "right", "right": "left"}
    # F1a: mirror equivariance (reference option prompt, no convention text), one row per patient
    c = ld("expC_laterality", "MODEL_T2FS_*.csv")
    if not c.empty and "side" in c:
        c = c[c.conv == "none"]
        w = c.pivot_table(index="subject", columns="geom", values="side", aggfunc="first")
        lat = w["orig"].isin(["left", "right"])
        out["F1 mirror-equivariant (patients)"] = f"{int((w.lr[lat] == w['orig'][lat].map(flip)).sum())}/{int(lat.sum())}"
        neg = w[w.index.isin(c[c.gt_side == "none"].subject)]
        nl = neg["orig"].isin(["left", "right"])
        out["F1 mirror-'equivariant' among lesion-FREE (chance control)"] = f"{int((neg.lr[nl] == neg['orig'][nl].map(flip)).sum())}/{int(nl.sum())}"
        o = c[c.geom == "orig"]
        out["F1 ref answers 'left'"] = f"{int((o.side == 'left').sum())}/{len(o)}"
        lt = c[c.geom.isin(["orig", "lr"]) & c.gt_side.isin(["left", "right"])].copy()
        lt["truth"] = lt.gt_side.where(lt.geom == "orig", lt.gt_side.map(flip))
        t = [[int(((lt.truth == "right") & (lt.side == "right")).sum()), int(((lt.truth == "right") & (lt.side != "right")).sum())],
             [int(((lt.truth == "left") & (lt.side == "right")).sum()), int(((lt.truth == "left") & (lt.side != "right")).sum())]]
        out["F1 ref: answer tracks true side (orig+mirror reads)"] = (f"right-truth->'right' {t[0][0]}/{sum(t[0])} vs left-truth->'right' "
                                                                     f"{t[1][0]}/{sum(t[1])} (Fisher p={fisher_exact(t)[1]:.2g})")
    # F1b: open-format side, real lesions
    c3 = ld("expC3_side_prompt", "MODEL_T2FS_*.csv")
    if not c3.empty:
        g = c3[(c3.variant == "open") & (c3.source != "synthetic")]
        t = [[int(((g.gt_side == "right") & (g.answer == "right")).sum()), int(((g.gt_side == "right") & (g.answer != "right")).sum())],
             [int(((g.gt_side == "left") & (g.answer == "right")).sum()), int(((g.gt_side == "left") & (g.answer != "right")).sum())]]
        out["F1 open: real lesions correct"] = f"{int((g.answer == g.gt_side).sum())}/{len(g)}"
        out["F1 open: right-sided called right"] = f"{t[0][0]}/{sum(t[0])} (Fisher p={fisher_exact(t)[1]:.2g})"
        r = c3[(c3.variant == "order_rl")]
        if len(r):
            out["F1 'right' listed first -> 'right'"] = f"{int((r.answer == 'right').sum())}/{len(r)}"
    # F1c + F2c: free-text reports
    rr = ld("expR_freetext_report", "MODEL_D2_*.csv")
    if not rr.empty:
        lat = rr[rr.lesion_sides.isin(["left", "right"])]
        out["F1 free text: lateral mentions 'left'"] = f"{int((lat.lesion_sides == 'left').sum())}/{len(lat)}"
        k = rr.pivot_table(index=["subject", "hist"], columns="geom", values="lesion_sides", aggfunc="first")
        both = k["orig"].isin(["left", "right"]) & k.lr.isin(["left", "right"])
        out["F1 free text: side unchanged by mirror (patient x history pairs)"] = f"{int((k.lr[both] == k['orig'][both]).sum())}/{int(both.sum())}"
        tl = rr[rr.gt_side.isin(["left", "right"]) & rr.lesion_sides.isin(["left", "right"])].copy()
        if len(tl):
            tl["truth"] = tl.gt_side.where(tl.geom == "orig", tl.gt_side.map(flip))
            t = [[int(((tl.truth == "right") & (tl.lesion_sides == "right")).sum()), int(((tl.truth == "right") & (tl.lesion_sides != "right")).sum())],
                 [int(((tl.truth == "left") & (tl.lesion_sides == "right")).sum()), int(((tl.truth == "left") & (tl.lesion_sides != "right")).sum())]]
            out["F1 free text: side tracks the truth (real lesions, reads)"] = (
                f"true-right->'right' {t[0][0]}/{sum(t[0])} vs true-left->'right' {t[1][0]}/{sum(t[1])} "
                f"(Fisher p={fisher_exact(t)[1]:.2g}; reads cluster in patients: descriptive)")
        neg = rr[(rr.label != "endometrioma") & (rr.geom == "orig")].assign(rep=lambda x: x.lesion_sides != "none")
        w = neg.pivot_table(index="subject", columns="hist", values="rep", aggfunc="first")
        b = int(((w["none"] == False) & (w.known_em == True)).sum()); cc = int(((w["none"] == True) & (w.known_em == False)).sum())
        out["F2 free text: lesion reported none->known (em-)"] = (f"{int(w['none'].sum())}/{len(w)} -> {int(w.known_em.sum())}/{len(w)} "
                                                                 f"(McNemar p={mcnemar_exact(b, cc):.2g})")
        if "truncated" in neg and neg.truncated.notna().all():
            t = neg.pivot_table(index="subject", columns="hist", values="truncated", aggfunc="first").astype(bool)
            w1 = w[~(t["none"] | t.known_em)]
            b = int(((w1["none"] == False) & (w1.known_em == True)).sum()); cc = int(((w1["none"] == True) & (w1.known_em == False)).sum())
            out["F2 free text: same, pairs with no truncated report"] = (f"{int(w1['none'].sum())}/{len(w1)} -> {int(w1.known_em.sum())}/{len(w1)} "
                                                                        f"(McNemar p={mcnemar_exact(b, cc):.2g})")
            out["F2 free text: truncated reports"] = f"{int(rr.truncated.astype(bool).sum())}/{len(rr)}"
    # F2a: history arms, generated, em-negative real volumes
    b4 = ld("expB4_history_controls", "MODEL_T2FS_*.csv")
    b2 = ld("expB2_history_dose", "MODEL_T2FS_*.csv")
    arms = pd.DataFrame()
    if not b4.empty:
        arms = b4[(b4.image == "real") & b4.part.isin(["irrelevant", "known"])][["subject", "label", "hist", "answer"]]
    if not b2.empty and "endo" in b2 and not (len(arms) and (arms["hist"] == "known_em").any()):
        k2 = b2[(b2.abl == "real") & (b2["hist"] == "known_em") & b2.endo.notna()][["subject", "label", "hist", "endo"]]
        arms = pd.concat([arms, k2.rename(columns={"endo": "answer"})])
    if len(arms):
        arms = arms[arms.label != "endometrioma"].drop_duplicates(["subject", "hist"])
        w = arms.pivot_table(index="subject", columns="hist", values="answer", aggfunc="first")
        for h in ["none", "knee", "pain", "known_em"]:
            if h in w:
                out[f"F2 'yes' with {h} (em-)"] = f"{int((w[h] == 'yes').sum())}/{int(w[h].notna().sum())}"
        if {"none", "known_em"} <= set(w.columns):
            base = w[w["none"] == "no"]
            base_y = w[w["none"] == "yes"]
            for h in ["knee", "pain", "known_em"]:
                if h in w:
                    out[f"F2 baseline-no flipped by {h}"] = f"{int((base[h] == 'yes').sum())}/{len(base)}"
                    out[f"F2 baseline-yes flipped to no by {h}"] = f"{int((base_y[h] == 'no').sum())}/{len(base_y)}"
            if "knee" in w:
                b = int(((base.knee == "no") & (base.known_em == "yes")).sum()); cc = int(((base.knee == "yes") & (base.known_em == "no")).sum())
                out["F2 known vs knee (McNemar p)"] = f"{mcnemar_exact(b, cc):.2g}"
    # F2 direct-answer logit (needed where generated answers sit at a ceiling, e.g. Gemma 3), em- real
    if not b4.empty and "logit_yes_no" in b4:
        r = b4[(b4.image == "real") & b4.part.isin(["irrelevant", "known"]) & (b4.label != "endometrioma")]
        r = r.drop_duplicates(["subject", "hist"])
        lw = r.pivot_table(index="subject", columns="hist", values="logit_yes_no")
        if "none" in lw:
            for h in ["knee", "pain", "known_em"]:
                if h in lw:
                    x = lw[[h, "none"]].dropna()
                    out[f"F2 logit: {h} > none (em-), mean shift"] = (f"{int((x[h] > x['none']).sum())}/{len(x)}, "
                                                                     f"{(x[h] - x['none']).mean():+.2f}")
            if {"known_em", "knee", "pain"} <= set(lw.columns):
                x = lw[["known_em", "knee", "pain"]].dropna()
                out["F2 logit: known > both irrelevant (em-)"] = f"{int((x.known_em > x[['knee', 'pain']].max(axis=1)).sum())}/{len(x)}"
        gen = r[r.answer.isin(["yes", "no"])]
        for h, g in gen.groupby("hist"):
            out[f"F2 readouts agree, sign(logit) == generated ({h})"] = f"{int(((g.logit_yes_no > 0) == (g.answer == 'yes')).sum())}/{len(g)}"
        for img_name in ["zeros_matched", "pelvis_blanked"]:
            g = b4[(b4.part == "blanks") & (b4.image == img_name)]
            if len(g):
                u = g.drop_duplicates(["input_sha", "hist"])
                m0, m1 = u[u["hist"] == "none"].logit_yes_no.mean(), u[u["hist"] == "known_em"].logit_yes_no.mean()
                out[f"F2 logit {img_name}: mean none vs known (distinct inputs)"] = (
                    f"{m0:+.2f} vs {m1:+.2f} (n={u[u['hist'] == 'none'].input_sha.nunique()})")
    # F2b: blank + history
    if not b4.empty:
        for img_name in ["zeros_matched", "pelvis_blanked"]:
            g = b4[(b4.part == "blanks") & (b4.image == img_name)]
            if len(g):
                n0, n1 = g[g["hist"] == "none"], g[g["hist"] == "known_em"]
                u0, u1 = n0.drop_duplicates("input_sha"), n1.drop_duplicates("input_sha")
                out[f"F2 {img_name}: 'yes' none vs known (distinct inputs)"] = (
                    f"{int((u0.answer == 'yes').sum())}/{len(u0)} vs {int((u1.answer == 'yes').sum())}/{len(u1)}")
    return out


def headline():
    print("\n=== HEADLINE comparison across models (same prompts, same pipeline, same code) ===")
    tab = pd.DataFrame([_headline_one(mo) for mo in ["medgemma", "gemma3", "medgemma27", "qwen"]]).set_index("model").T
    with pd.option_context("display.max_colwidth", 60, "display.width", 250):
        print(tab.fillna("-").to_string())


ALL = {"inventory": inventory, "patientlevel": patientlevel, "parser": parser, "exp00": exp00, "expA": expA, "expA2": expA2, "expA3": expA3, "expB": expB, "expB2": expB2, "expB2_D1": lambda: expB2("D1"), "expB3": expB3,
       "expC": expC, "expC2": expC2, "expC3": expC3, "expD": expD, "expD2": expD2, "expE": expE, "expF": expF, "expN": expN, "expP": expP, "expX": expX, "expR": expR, "expB4": expB4, "expC4": expC4, "proxy": proxy, "confirm_D1": confirm_D1, "headline": headline}
if __name__ == "__main__":
    args = sys.argv[1:]
    if "--model" in args:
        i = args.index("--model")
        MODEL = args[i + 1]
        del args[i:i + 2]
    print(f"# model: {MODEL}")
    for e in (args or ALL):
        ALL[e]()

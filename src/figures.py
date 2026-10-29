"""Write-up figures (static PNG, figures/). Every number shown is read from results/ CSVs.

Palette: validated reference categorical slots 1-3 (blue, orange, aqua) + neutral grey,
thin marks, direct labels, recessive grid (dataviz method).

Run: python src/figures.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from io_orient import axial_slice_radiological, world_centroid
from perturb import insert_sphere, lr_flip, pelvis_blanked
from pipeline import load_ras, normalize
from qc_overlay import mask_on, union_mask
from stats import cluster_bootstrap_prop, wilson
import analyze

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#ebeae6", "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
})


def load(exp: str, pattern: str = "medgemma_*.csv") -> pd.DataFrame:
    """Same loader as analyze.py: exclusions and keyed de-duplication (P11)."""
    return analyze.load(exp, pattern)


def cluster_ci(success, cluster):
    """95% CI for a proportion of reads clustered within patients (patient-cluster bootstrap);
    Wilson over patients when every read agrees (bootstrap is degenerate there)."""
    p, lo, hi, n, k = cluster_bootstrap_prop(success, cluster)
    if p in (0.0, 1.0):
        per = pd.Series(np.asarray(success, bool)).groupby(np.asarray(cluster)).all()
        return wilson(int(per.sum()), len(per))
    return lo, hi


def show(ax, sl, title, answer, answer_color):
    ax.imshow(sl, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=8.5, color=INK)
    ax.text(0.5, -0.07, answer, transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            color=INK, weight="bold", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=answer_color, lw=1.5))
    ax.text(0.03, 0.5, "R", color="yellow", transform=ax.transAxes, fontsize=10, weight="bold", va="center")
    ax.text(0.92, 0.5, "L", color="yellow", transform=ax.transAxes, fontsize=10, weight="bold", va="center")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def outline(ax, mask2d, color="#ff3b3b"):
    """Draw a lesion outline (2D boolean array already in display orientation)."""
    if mask2d.any():
        ax.contour(mask2d.astype(float), [0.5], colors=color, linewidths=1.2)


def prop_bar(ax, labels, ks, ns, colors, xlabel, cis=None):
    """Horizontal bars of proportions with 95% CI whiskers (Wilson unless `cis` gives one per bar)
    and direct labels."""
    y = np.arange(len(labels))[::-1]
    cis = cis or [None] * len(labels)
    for yi, k, n, c, ci in zip(y, ks, ns, colors, cis):
        p = k / n if n else np.nan
        lo, hi = ci if ci is not None else wilson(k, n)
        ax.barh(yi, p, height=0.55, color=c, edgecolor="white", linewidth=2)
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1)
        ax.text(min(hi + 0.02, 0.98) if p < 0.85 else p - 0.02, yi, f"{k}/{n}", va="center",
                ha="left" if p < 0.85 else "right", fontsize=8.5, color=INK if p < 0.85 else "white")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------- Failure 1: laterality
def fig_laterality():
    import io as _io
    import zipfile
    from io_orient import load_canonical
    c = load("expC_laterality")
    p_ = load("expP_positive_controls")
    x_ = load("expX_cxr_laterality", "medgemma.csv")
    c3 = load("expC3_side_prompt")
    c4 = load("expC4_psychometric")
    rr = load("expR_freetext_report", "medgemma_D2_*.csv")

    def said(df, col, **where):
        m = np.ones(len(df), bool)
        for k_, v_ in where.items():
            m &= (df[k_] == v_).values
        return f'MedGemma: "{df[m][col].iloc[0]}"' if m.any() else "?"

    # pelvic right-sided endometrioma, original and mirrored (reference prompt)
    sid = "D2-054"
    img, vol = load_ras(sid, "T2FS")
    emask = mask_on(img, union_mask(sid, "em"))
    k = int(np.argmax(emask.sum(axis=(0, 1))))
    # brain: first right-sided tumour answered correctly in open format (illustrative; rates below)
    brain = pd.read_csv(ROOT / "derived" / "brain_inventory.csv").set_index("subject")
    p2 = p_[(p_.part == "P2") & (p_.cond == "open_orig")]
    bsid = next(s_ for s_ in p2.subject if brain.loc[s_, "side"] == "right" and (p2[p2.subject == s_].answer == "right").all())
    bvol = np.asarray(load_canonical(brain.loc[bsid, "path"]).dataobj, dtype=np.float32)
    bseg = np.asarray(load_canonical(brain.loc[bsid, "path"].replace("-t2w", "-seg")).dataobj) > 0
    bk = int(brain.loc[bsid, "max_slice"])
    # CXR: first right-sided real finding answered correctly in open format
    xr = x_[(x_.source == "real") & (x_.cond == "orig") & (x_.fmt == "open") & (x_.truth == "right") & (x_.answer == "right")]
    xname = xr.image.iloc[0]
    z = zipfile.ZipFile(ROOT / "derived" / "cxr_nih" / "raw" / "data" / "images" / "images_001.zip")
    xpath = next(n for n in z.namelist() if n.endswith(xname))
    cxr = np.asarray(Image.open(_io.BytesIO(z.read(xpath))).convert("L").resize((512, 512)))
    bb = pd.read_csv(ROOT / "derived" / "cxr_nih" / "raw" / "data" / "BBox_List_2017.csv")
    bb = bb[bb["Image Index"] == xname].iloc[0]

    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.6, wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    show(ax, axial_slice_radiological(normalize(vol), k),
         f"PELVIS {sid}\nendometrioma, patient RIGHT", said(c, "side", subject=sid, geom="orig", conv="none"), ORANGE)
    outline(ax, axial_slice_radiological(emask, k))
    ax = fig.add_subplot(gs[0, 1])
    show(ax, axial_slice_radiological(normalize(lr_flip(vol)), k),
         "same pelvis, mirrored L↔R\n(lesion now patient LEFT)", said(c, "side", subject=sid, geom="lr", conv="none"), ORANGE)
    outline(ax, axial_slice_radiological(lr_flip(emask), k))
    ax = fig.add_subplot(gs[0, 2])
    show(ax, axial_slice_radiological(normalize(bvol), bk),
         f"BRAIN MRI {bsid[-9:]}\ntumour, patient RIGHT", said(p2, "answer", subject=bsid), BLUE)
    outline(ax, axial_slice_radiological(bseg, bk))
    ax = fig.add_subplot(gs[0, 3])
    show(ax, cxr, f"CHEST X-RAY (PA)\n{bb['Finding Label'].lower()}, patient RIGHT", said(xr, "answer", image=xname), BLUE)
    ax.add_patch(matplotlib.patches.Rectangle((bb["Bbox [x"] / 2, bb["y"] / 2), bb["w"] / 2, bb["h]"] / 2,
                                              fill=False, ec="#ff3b3b", lw=1.2))

    # (bottom left) side read from the image? open format / free text, by domain
    ax = fig.add_subplot(gs[1, :2])
    p2o = p_[(p_.part == "P2") & (p_.cond == "open_orig")]          # one read per patient
    xo = x_[(x_.source == "real") & (x_.fmt == "open") & (x_.cond == "orig")]
    c3o = c3[(c3.variant == "open")]
    real_o, syn_o = c3o[c3o.source != "synthetic"], c3o[c3o.source == "synthetic"]
    rows = [(f"brain MRI, gliomas ({len(p2o)} patients)", int((p2o.answer == p2o.truth).sum()), len(p2o), BLUE, None),
            (f"chest X-ray, real findings ({len(xo)} findings on {xo.image.nunique()} films)",
             int((xo.answer == xo.truth).sum()), len(xo), BLUE, None),
            (f"pelvic MRI, SYNTHETIC 36 mm cyst ({len(syn_o)} reads: L+R in {syn_o.subject.nunique()} pelvises)",
             int((syn_o.answer == syn_o.gt_side).sum()), len(syn_o), AQUA,
             cluster_ci(syn_o.answer == syn_o.gt_side, syn_o.subject)),
            (f"pelvic MRI, real lesions ({real_o.subject.nunique()} patients)", int((real_o.answer == real_o.gt_side).sum()), len(real_o), ORANGE, None)]
    prop_bar(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
             "side named correctly, open question (no option list), 95% CI\n(synthetic: patient-cluster bootstrap)",
             cis=[r[4] for r in rows])
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_title("(a) open question: brain MRI and CXR yes, pelvic lesions no",
                 fontsize=9, loc="left", color=INK)

    # (bottom right) psychometric: share of volumes whose side log-odds move the right way
    ax = fig.add_subplot(gs[1, 2:])
    for cname, col in [("bright", BLUE), ("moderate", AQUA), ("dark", GREY)]:
        sub = c4[(c4.contrast == cname) & (c4.dist_mm == 45)]
        g = sub.groupby("radius_mm").correct_dir.agg(["sum", "count"])
        y = g["sum"] / g["count"]
        # 2 reads (left, right) per pelvis at each point: patient-cluster bootstrap CI (P15)
        ci = np.array([cluster_ci(gg.correct_dir.astype(bool), gg.subject) for _, gg in sub.groupby("radius_mm")])
        ax.plot(g.index, y, "-o", color=col, lw=2, ms=5)
        ax.fill_between(g.index, ci[:, 0], ci[:, 1], color=col, alpha=0.12, lw=0)
        ax.text(g.index[-1] + 0.6, y.iloc[-1] + {"bright": 0.05, "moderate": -0.0, "dark": -0.05}[cname], cname,
                color=col, va="center", fontsize=8)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("SYNTHETIC sphere radius (mm), 45 mm from midline")
    ax.set_ylabel("side log-odds moved\nthe correct way")
    ax.text(0.98, 0.04, f"n = {c4.subject.nunique()} pelvises x 2 sides per point;\n95% CI: patient-cluster bootstrap",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(3, 35)
    ax.set_title(f"(b) SYNTHETIC sharp-edged spheres in {c4.subject.nunique()} pelvises (direct-answer log-odds)",
                 fontsize=9, loc="left", color=INK)
    fig.suptitle("Failure 1: MedGemma extracts no laterality from pelvic MRI; the reported side is a default",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.savefig(FIG / "fig1_laterality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Failure 2: history anchoring
def fig_history():
    b2 = load("expB2_history_dose", "medgemma_T2FS_*.csv")
    b4 = load("expB4_history_controls")
    blanks = b4[b4.part == "blanks"]
    # illustrative case: the first volume (in subject order) showing the modal pattern, i.e. "no" without
    # history, "yes" with it, and "yes" with the pelvis erased; frequencies are in the bottom panels
    pbk = blanks[(blanks.image == "pelvis_blanked") & (blanks["hist"] == "known_em")].set_index("subject").answer
    real_ = b2[(b2.abl == "real") & (b2.label == "negative") & b2.endo.notna()].pivot_table(
        index="subject", columns="hist", values="endo", aggfunc="first")
    sid = next(s for s in sorted(pbk.index) if pbk[s] == "yes" and s in real_.index
               and real_.loc[s, "none"] == "no" and real_.loc[s, "known_em"] == "yes")
    img, vol = load_ras(sid, "T2FS")
    k = vol.shape[2] // 2
    r = b2[(b2.subject == sid) & (b2.abl == "real")].set_index("hist")
    pb = blanks[(blanks.subject == sid) & (blanks.image == "pelvis_blanked")].set_index("hist")

    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.5, wspace=0.75)
    show(fig.add_subplot(gs[0, 0]), axial_slice_radiological(normalize(vol), k),
         f"{sid} (no lesion), no history", f'MedGemma: "{r.endo.get("none", "?")}"', BLUE)
    show(fig.add_subplot(gs[0, 1]), axial_slice_radiological(normalize(vol), k),
         '+ "known ovarian endometrioma,\nfollow-up study"', f'MedGemma: "{r.endo.get("known_em", "?")}"', ORANGE)
    show(fig.add_subplot(gs[0, 2]), axial_slice_radiological(normalize(pelvis_blanked(vol)), k),
         "same volume, pelvis contents\nERASED (SYNTHETIC) + same history", f'MedGemma: "{pb.answer.get("known_em", "?")}"', ORANGE)

    # (bottom left) generated flips among the SAME volumes read "no" without history
    ax = fig.add_subplot(gs[1, :2])
    g = b2[(b2.abl == "real") & (b2.label != "endometrioma") & b2.endo.notna()]
    w = g.pivot_table(index="subject", columns="hist", values="endo", aggfunc="first")
    irr = b4[b4.part == "irrelevant"].pivot_table(index="subject", columns="hist", values="answer", aggfunc="first")
    w = w.join(irr[["knee", "pain"]], how="left")
    base_no, base_yes = w[w["none"] == "no"], w[w["none"] == "yes"]
    hs = [("knee", '"knee pain" (irrelevant)', GREY), ("pain", '"chronic pelvic pain" (neutral)', GREY),
          ("suspected", '"suspected endometriosis"', ORANGE), ("known_em", '"known ovarian endometrioma, follow-up"', ORANGE)]
    # both directions (P16): baseline 'no' -> 'yes', and baseline 'yes' -> 'no'
    labels = [f'{lab}: "no" -> "yes"' for _, lab, _ in hs] + [f'{lab}: "yes" -> "no"' for _, lab, _ in hs]
    prop_bar(ax, labels, [int((base_no[h] == "yes").sum()) for h, _, _ in hs] + [int((base_yes[h] == "no").sum()) for h, _, _ in hs],
             [len(base_no)] * len(hs) + [len(base_yes)] * len(hs), [c for _, _, c in hs] * 2,
             f'em-negative volumes whose answer changes with the history\n'
             f'(baseline "no" n={len(base_no)}, baseline "yes" n={len(base_yes)})')

    # (bottom right) generated 'yes' with vs without history, by image content
    ax = fig.add_subplot(gs[1, 2])
    conds = [("real", "real\nanatomy"), ("pelvis_blanked", "pelvis\nerased"), ("zeros_matched", "all-zero\nvolume")]
    counts = {}
    for x, (img_name, lab) in enumerate(conds):
        for h, col, dx in [("none", BLUE, -0.18), ("known_em", ORANGE, 0.18)]:
            if img_name == "real":
                gg = g[g["hist"] == h]
                k_, n_ = int((gg.endo == "yes").sum()), len(gg)
            else:
                # identical inputs (the all-zero volumes repeat) give identical answers: count each once (P17)
                gg = blanks[(blanks.image == img_name) & (blanks["hist"] == h)].drop_duplicates("input_sha")
                k_, n_ = int((gg.answer == "yes").sum()), len(gg)
            lo, hi = wilson(k_, n_)
            ax.bar(x + dx, k_ / n_, width=0.32, color=col, edgecolor="white", linewidth=2)
            ax.plot([x + dx] * 2, [lo, hi], color=INK, lw=1)
            counts.setdefault(x, []).append(f"{k_}/{n_}")
    ax.set_xticks(range(len(conds)), [f"{c[1]}\n" + "\n".join(counts[i]) for i, c in enumerate(conds)], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel('generated "yes"\n(counts: distinct inputs)')
    ax.text(0.5, 1.13, "known endometrioma", color=ORANGE, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.text(0.5, 1.05, "no history", color=BLUE, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Failure 2: a one-line clinical history creates the finding, even when the pelvis is erased",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.savefig(FIG / "fig2_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- S1: option-position mechanism
def fig_option_position():
    c3, a3 = load("expC3_side_prompt"), load("expA3_type_order")
    if c3.empty or a3.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1.3]})
    # (a) share answering "right", by option order and true side
    rows = []
    for v, vlab in [("ref", "'left' listed first"), ("order_rl", "'right' listed first"), ("open", "no option list")]:
        for side in ["left", "right"]:
            for src, slab in [(False, "real"), (True, "SYNTHETIC")]:  # not pooled (P18)
                g = c3[(c3.variant == v) & (c3.gt_side == side) & ((c3.source == "synthetic") == src)]
                rows.append((f"{vlab}, {slab} lesion {side}", int((g.answer == "right").sum()), len(g),
                             ORANGE if v == "order_rl" else BLUE if v == "ref" else GREY))
    prop_bar(ax1, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
             'answers "right" (95% CI)')
    ax1.set_title("(a) side follows the option list", fontsize=9, loc="left", color=INK)
    # (b) lesion-type answer vs first-listed option (counts)
    opts = ["endometrioma", "simple cyst", "dermoid", "hemorrhagic cyst", "none"]
    t = pd.crosstab(a3.first_option, a3.answer).reindex(index=opts, columns=opts[:-1], fill_value=0)
    ax2.imshow(t.values, cmap="Blues", vmin=0, vmax=t.values.max())
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax2.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                     color="white" if v > t.values.max() / 2 else INK)
    ax2.set_xticks(range(t.shape[1]), t.columns, rotation=20, ha="right")
    ax2.set_yticks(range(t.shape[0]), [f"'{o}' first" for o in t.index])
    ax2.set_xlabel("answer")
    ax2.grid(False)
    ax2.set_title(f"(b) lesion type = first listed option ({len(a3)} reads, endometriomas and cysts)",
                  fontsize=9, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "figS1_option_position.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_laterality()
    fig_history()
    fig_option_position()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))

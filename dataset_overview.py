"""Overview of the D2 cohort of UT-EndoMRI: what files exist, and simple descriptive plots.

Outputs (in figures/dataset_overview/):
  table_D2_files.csv / .png   rows = mask type, columns = patients with that mask, and how many of
                              those also have each image series (T1, T1FS, T2, T2FS). First row =
                              all patients.
  slices_per_series_D2.png    histogram of the number of slices per MRI series, one panel per series
  slice_thickness_D2.png      slice thickness (mm) per series
  labels_D2.png               ground-truth composition (endometrioma / cyst only / neither) and lesion
                              sides (side of the largest lesion, from derived/inventory.csv)
  lesion_volume_D2.png        lesion volume (mL) for endometriomas and cysts (from derived/inventory.csv)

Counts are PATIENTS (subject folders). A mask "exists" when its label file exists; no D2 mask file
is empty. The three "<sid>_pat.nii.gz" files are images of unknown sequence, not masks, and are
ignored (listed in the printout). D2 has one rater, so one file per mask type. Excluded subjects (src/io_orient.py
EXCLUDED_SUBJECTS: D2-007, D2-009) are left out unless --include-excluded is given.

Reads NIfTI headers only (no voxel data). Run from this folder:
    .venv/bin/python dataset_overview.py [--include-excluded]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from io_orient import COHORT_DIRS, EXCLUDED_SUBJECTS  # noqa: E402

OUT = HERE / "figures" / "dataset_overview"
SEQS = ["T1", "T1FS", "T2", "T2FS"]
MASKS = {"ut": "uterus", "ov": "ovary", "cy": "cyst", "em": "endometrioma"}
# same file-name patterns as src/inventory.py. "<sid>_pat.nii.gz" (3 D2 subjects) is NOT a mask: it
# holds an MRI image of unknown sequence, so it is ignored here as in src/inventory.py.
PAT_RE = re.compile(r"^(D[12]-\d+)_pat\.nii\.gz$")
SEQ_RE = re.compile(r"^(D[12]-\d+)_(T1|T1FS|T2|T2FS)\.nii\.gz$")
LABEL_RE = re.compile(r"^(D[12]-\d+)_([a-z]+)(?:_r(?:e)?(\d))?\.nii\.gz$")
SEQ_COLORS = {"T1": "#2a78d6", "T1FS": "#1baf7a", "T2": "#eb6834", "T2FS": "#8a5cd6"}


def scan_d2(include_excluded: bool) -> pd.DataFrame:
    """One row per D2 subject: which series and masks exist, plus slice count / thickness per series."""
    rows = []
    for sdir in sorted(p for p in COHORT_DIRS["D2"].iterdir() if p.is_dir()):
        if sdir.name in EXCLUDED_SUBJECTS and not include_excluded:
            continue
        row = {"subject": sdir.name}
        for f in sorted(sdir.iterdir()):
            if m := SEQ_RE.match(f.name):
                img = nib.load(str(f))                       # header only
                zooms = np.array(img.header.get_zooms()[:3])
                ax = int(np.argmax(zooms))                   # slice axis = thickest voxels (all D2 series are axial)
                row[m.group(2)] = True
                row[f"{m.group(2)}_slices"] = int(img.shape[ax])
                row[f"{m.group(2)}_thickness_mm"] = float(zooms[ax])
            elif PAT_RE.match(f.name):
                row["pat_file"] = True   # ignored image file, reported separately
            elif m := LABEL_RE.match(f.name):
                row[m.group(2)] = True
        rows.append(row)
    d = pd.DataFrame(rows)
    for c in SEQS + list(MASKS) + ["pat_file"]:
        d[c] = d.get(c, pd.Series(False, index=d.index)).fillna(False).astype(bool)
    return d


def file_table(d: pd.DataFrame) -> pd.DataFrame:
    """Rows: all patients, then each mask type. Columns: patients, and of those, how many have each series."""
    rows = [("all patients", d)] + [(f"{k} ({v})", d[d[k]]) for k, v in MASKS.items()]
    t = pd.DataFrame([{"": name, "patients": len(g), **{s: int(g[s].sum()) for s in SEQS}} for name, g in rows])
    return t.set_index("")


def save_table_png(t: pd.DataFrame, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * (len(t) + 2)))
    ax.axis("off")
    tb = ax.table(cellText=t.values, rowLabels=t.index, colLabels=t.columns, loc="center", cellLoc="center")
    tb.scale(1, 1.4)
    for (r, c), cell in tb.get_celld().items():
        if r == 0 or c == -1:
            cell.set_text_props(weight="bold")
        if r == 1:
            cell.set_facecolor("#eef2f8")   # 'all patients' row
    ax.set_title(title, fontsize=10, loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slices(d: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, s in zip(axes, SEQS):
        v = d[f"{s}_slices"].dropna() if f"{s}_slices" in d else pd.Series(dtype=float)
        if len(v):
            ax.hist(v, bins=np.arange(v.min() - 0.5, v.max() + 1.5, max(1, int((v.max() - v.min()) // 20) or 1)),
                    color=SEQ_COLORS[s], edgecolor="white")
            ax.axvline(85, color="k", ls=":", lw=1)  # the pipeline shows at most 85 slices
            ax.set_title(f"{s} (n={len(v)}): median {int(v.median())}, range {int(v.min())}-{int(v.max())}", fontsize=9)
        ax.set_xlabel("slices per series")
    axes[0].set_ylabel("series (patients)")
    fig.suptitle("D2: number of axial slices per MRI series (dotted line: 85 = most slices shown to the model)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_thickness(d: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    data = [d[f"{s}_thickness_mm"].dropna() for s in SEQS]
    for i, (s, v) in enumerate(zip(SEQS, data)):
        x = np.random.default_rng(0).normal(i, 0.06, len(v))
        ax.scatter(x, v, s=12, color=SEQ_COLORS[s], alpha=0.7)
    ax.set_xticks(range(4), [f"{s}\n(n={len(v)})" for s, v in zip(SEQS, data)])
    ax.set_ylabel("slice thickness (mm)")
    ax.set_title("D2: slice thickness per series", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_labels(d: pd.DataFrame, inv: pd.DataFrame, path: Path):
    """Left: ground-truth composition (each patient counted once). Right: lesion side of the largest lesion."""
    d = d.merge(inv[["subject", "em_side", "cy_side"]], on="subject", how="left")
    comp = {"endometrioma": int(d.em.sum()),
            "cyst only": int((d.cy & ~d.em).sum()),
            "neither": int((~d.cy & ~d.em).sum())}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1.4]})
    a1.bar(comp.keys(), comp.values(), color=["#eb6834", "#2a78d6", "#8a8984"])
    for i, v in enumerate(comp.values()):
        a1.text(i, v, str(v), ha="center", va="bottom")
    a1.set_ylabel("patients")
    a1.set_title(f"D2 ground truth ({len(d)} patients;\n"
                 f"{int((d.em & d.cy).sum())} with both counted as endometrioma)", fontsize=9, loc="left")
    sides = ["left", "right", "midline", "bilateral"]
    em = d[d.em].em_side.value_counts().reindex(sides, fill_value=0)
    cy = d[d.cy].cy_side.value_counts().reindex(sides, fill_value=0)
    x = np.arange(len(sides))
    a2.bar(x - 0.2, em.values, 0.4, label=f"endometrioma (n={int(d.em.sum())})", color="#eb6834")
    a2.bar(x + 0.2, cy.values, 0.4, label=f"cyst (n={int(d.cy.sum())})", color="#2a78d6")
    a2.set_xticks(x, sides)
    a2.set_ylabel("patients")
    a2.legend(fontsize=8, frameon=False)
    a2.set_title("side of the largest lesion (world mm, vs uterus centroid;\n"
                 "vs scan centre when there is no uterus mask)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_volumes(d: pd.DataFrame, inv: pd.DataFrame, path: Path):
    v = inv[inv.subject.isin(d.subject)]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, (k, lab, col) in enumerate([("em", "endometrioma", "#eb6834"), ("cy", "cyst", "#2a78d6")]):
        x = v[f"{k}_volume_ml"].dropna()
        ax.scatter(np.random.default_rng(0).normal(i, 0.06, len(x)), x, color=col, s=16, alpha=0.8)
        ax.text(i + 0.12, x.median(), f"median {x.median():.1f} mL", fontsize=8, va="center")
    ax.set_yscale("log")
    ax.set_xticks([0, 1], [f"endometrioma (n={v.em_volume_ml.notna().sum()})", f"cyst (n={v.cy_volume_ml.notna().sum()})"])
    ax.set_ylabel("lesion volume (mL, log scale)")
    ax.set_title("D2: lesion volume (all components of the mask)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-excluded", action="store_true", help="also count D2-007 and D2-009")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d = scan_d2(args.include_excluded)
    excl = "including" if args.include_excluded else "excluding"
    t = file_table(d)
    t.to_csv(OUT / "table_D2_files.csv")
    print(f"D2: {len(d)} patients ({excl} {', '.join(sorted(EXCLUDED_SUBJECTS))})\n")
    print(t.to_string())
    pat = d[d.pat_file]
    print(f"\nignored '_pat' image files (unknown sequence, not a mask): {', '.join(pat.subject)}; "
          f"of these, without any named series (T1/T1FS/T2/T2FS): "
          f"{', '.join(pat.subject[~pat[SEQS].any(axis=1)]) or 'none'}\n")
    save_table_png(t, f"D2 patients by mask type and image series ({excl} D2-007/009)", OUT / "table_D2_files.png")

    plot_slices(d, OUT / "slices_per_series_D2.png")
    plot_thickness(d, OUT / "slice_thickness_D2.png")
    for s in SEQS:
        v = d[f"{s}_slices"].dropna()
        print(f"{s:5s}: {len(v):2d} series, slices median {int(v.median())} (range {int(v.min())}-{int(v.max())}); "
              f"thickness {', '.join(f'{x:g}' for x in sorted(d[f'{s}_thickness_mm'].dropna().round(2).unique()))} mm")

    inv_path = HERE / "derived" / "inventory.csv"
    if inv_path.exists():
        inv = pd.read_csv(inv_path)
        plot_labels(d, inv, OUT / "labels_D2.png")
        plot_volumes(d, inv, OUT / "lesion_volume_D2.png")
    else:
        print(f"\n{inv_path} not found: skipped the label and volume plots")
    print(f"\nwrote {sorted(p.name for p in OUT.iterdir())} to {OUT}")


if __name__ == "__main__":
    main()

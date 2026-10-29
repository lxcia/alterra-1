"""Orientation / label-alignment QC.

For every subject with an endometrioma or cyst label, resample the lesion and uterus
masks into each available sequence's world space (nearest neighbour), canonicalise
to RAS+, and save a radiological-convention axial slice through the lesion with
L/R annotations and the mask-derived side in the title.

Outputs: derived/qc/<subject>.png and derived/qc/qc_alignment.csv (fraction of the
lesion that lands inside the image field of view, mean image intensity inside vs
around the lesion on each sequence).

Besides the QC script, this module provides two helpers that the experiments use to put masks on
an image grid: `union_mask` (merge raters) and `mask_on` (resample a mask onto an image). Masks and
images can have different grids (a mask may be drawn on T2 while the model is shown T2FS, or on a
sagittal grid), so masks are always moved through world coordinates, never by array index.

Run:  .venv/bin/python src/qc_overlay.py [subject ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy import ndimage

from io_orient import SEQUENCES, axial_slice_radiological, load_canonical, subject_dir

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "derived" / "qc"


def union_mask(sid: str, struct: str) -> nib.Nifti1Image | None:
    """Union of all raters' masks for a structure, in the first mask's own space.

    Finds every <sid>_<struct>*.nii.gz file (one per rater), resamples raters 2..n onto rater 1's
    grid (nearest neighbour, via world coordinates) and ORs them. Returns a RAS+ uint8 image, or
    None if the subject has no file for that structure. Same rule as inventory.union_world."""
    # Finds every file for this subject and structure
    files = sorted(subject_dir(sid).glob(f"{sid}_{struct}*.nii.gz"))
    if not files:
        return None
    # This loads rater 1's file in RAS+ orientation. Its grid becomes the reference that every other rater gets moved onto.
    ref = load_canonical(files[0])
    # This turns rater 1's file into a True/False array ("acc" for accumulator). Any nonzero value counts as inside. It starts as rater 1's mask.
    acc = np.asarray(ref.dataobj) > 0
    # Go through remaining raters 
    for f in files[1:]:
        # Redraw this rater's mask on rater 1's grid via world mm; nearest neighbour keeps it 0/1
        m = resample_from_to(load_canonical(f), ref, order=0)
        # Union so the voxel either stays or becomes true if this rater marked it too (can only grow)
        acc |= np.asarray(m.dataobj) > 0
    #  Combined mask as a 0/1 NIfTI with rater 1's affine, so its mm positions are known
    return nib.Nifti1Image(acc.astype(np.uint8), ref.affine)


def mask_on(img: nib.Nifti1Image, mask: nib.Nifti1Image) -> np.ndarray:
    """Boolean array of `mask` resampled onto `img`'s voxel grid (nearest neighbour, world space).

    Used by the experiments to know which voxels of the volume shown to the model are lesion or
    uterus (e.g. to find the uterus centroid before inserting a synthetic sphere)."""
    return np.asarray(resample_from_to(mask, img, order=0).dataobj) > 0


def main(subjects: list[str]):
    inv = pd.read_csv(ROOT / "derived" / "inventory.csv").set_index("subject")
    # default: every subject with an endometrioma or cyst label (QC images are for labelled lesions)
    if not subjects:
        subjects = list(inv.index[(inv.em_present == True) | (inv.get("cy_present") == True)])
    QC.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in subjects:
        seqs = [s for s in SEQUENCES if (subject_dir(sid) / f"{sid}_{s}.nii.gz").exists()]
        lesions = {k: union_mask(sid, k) for k in ["em", "cy"]}
        lesions = {k: v for k, v in lesions.items() if v is not None and np.asarray(v.dataobj).any()}
        if not lesions or not seqs:
            continue
        ut = union_mask(sid, "ut")
        # one panel per (lesion type, sequence): rows = em / cy, columns = available sequences
        fig, axes = plt.subplots(len(lesions), len(seqs), figsize=(4 * len(seqs), 4.3 * len(lesions)),
                                 squeeze=False)
        for li, (kind, les) in enumerate(lesions.items()):
            les_vox = int(np.asarray(les.dataobj).sum())
            for si, seq in enumerate(seqs):
                img = load_canonical(subject_dir(sid) / f"{sid}_{seq}.nii.gz")
                vol = np.asarray(img.dataobj, dtype=np.float32)
                m = mask_on(img, les)                                   # lesion on this sequence's grid
                u = mask_on(img, ut) if ut is not None else None        # uterus on this sequence's grid
                ax = axes[li, si]
                # in-FOV fraction: lesion voxels that landed on this grid, corrected for the
                # different voxel sizes of the mask grid and the image grid. Well below 1 means the
                # mask and image don't overlap in space (misregistration or wrong file).
                vox_ratio = np.prod(les.header.get_zooms()[:3]) / np.prod(img.header.get_zooms()[:3])
                frac_in = float(m.sum() / max(les_vox * vox_ratio, 1))
                row = dict(subject=sid, lesion=kind, seq=seq, frac_in_fov=round(frac_in, 3))
                if m.any():
                    k = int(np.argmax(m.sum(axis=(0, 1))))   # axial slice with the most lesion voxels
                    # contrast check: mean intensity inside the lesion / mean in a 4-voxel ring around it
                    # (endometriomas should be bright on T1FS; cysts dark)
                    ring = ndimage.binary_dilation(m, iterations=4) & ~m
                    row["in_over_ring"] = round(float(vol[m].mean() / (vol[ring].mean() + 1e-6)), 2)
                    sl = axial_slice_radiological(vol, k)
                    lo, hi = np.percentile(sl, [1, 99])
                    ax.imshow(np.clip(sl, lo, hi), cmap="gray")
                    ax.contour(axial_slice_radiological(m.astype(float), k), [0.5], colors="r", linewidths=1)
                    if u is not None and u[..., k].any():
                        ax.contour(axial_slice_radiological(u.astype(float), k), [0.5], colors="c", linewidths=0.8)
                    ax.set_title(f"{seq} z={k}/{vol.shape[2]}  in-FOV={frac_in:.2f}", fontsize=9)
                else:
                    ax.set_title(f"{seq}: lesion outside FOV", fontsize=9)
                # orientation letters: radiological convention puts patient R on the viewer's left
                ax.text(0.02, 0.5, "R", color="yellow", transform=ax.transAxes, fontsize=14, weight="bold")
                ax.text(0.94, 0.5, "L", color="yellow", transform=ax.transAxes, fontsize=14, weight="bold")
                ax.text(0.47, 0.93, "A", color="yellow", transform=ax.transAxes, fontsize=12)
                ax.axis("off")
                rows.append(row)
            # the side written in the title comes from the inventory, so a reader can check by eye
            # that the outlined lesion is on that side of the image
            side = inv.loc[sid, f"{kind}_side"]
            axes[li, 0].text(-0.05, 1.08, f"{sid} {kind} (red) side={side}; uterus=cyan",
                             transform=axes[li, 0].transAxes, fontsize=11, weight="bold")
        fig.suptitle("Radiological convention: patient RIGHT on viewer LEFT", fontsize=10, y=0.995)
        fig.tight_layout()
        fig.savefig(QC / f"{sid}.png", dpi=70)
        plt.close(fig)
        print(sid, [(r["lesion"], r["seq"], r["frac_in_fov"], r.get("in_over_ring")) for r in rows if r["subject"] == sid])
    out = QC / "qc_alignment.csv"
    df = pd.DataFrame(rows)
    # when run for named subjects only, update their rows in the existing CSV and keep the rest
    if out.exists() and len(sys.argv) > 1:
        old = pd.read_csv(out)
        df = pd.concat([old[~old.subject.isin(df.subject)], df])
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])

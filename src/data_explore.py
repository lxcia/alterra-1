"""Describe how the UT-EndoMRI files are stored: orientation codes and acquisition plane.

For every NIfTI file (images and label masks, both cohorts) this reads only the header and
reports:
  * orientation code on disk, e.g. "LPS" = array axis 0 points to the patient's Left, axis 1 to
    Posterior, axis 2 to Superior (nib.aff2axcodes). "RAS" is already canonical; everything else
    is reordered/flipped by io_orient.load_canonical when loaded.
  * acquisition plane: the slice direction is taken to be the array axis with the THICKEST voxels
    (2D MRI is acquired as thin in-plane pixels and thick slices). Its direction in world space
    decides the plane: mostly along z (head-foot) = axial, along y (front-back) = coronal, along x
    (left-right) = sagittal. If all three voxel sizes are within 20% of each other the volume is
    "3D/isotropic" (no preferred slice direction).
  * obliquity: angle between that slice direction and the nearest world axis (0 = straight).

Reads headers only; writes nothing.
Run: .venv/bin/python src/data_explore.py
"""
from __future__ import annotations

import re

import nibabel as nib
import numpy as np
import pandas as pd

from io_orient import COHORT_DIRS, EXCLUDED_SUBJECTS

FILE_RE = re.compile(r"^(D[12]-\d+)_([A-Za-z0-9]+?)(?:_r(?:e)?\d)?\.nii\.gz$")
IMAGE_SEQS = {"T1", "T1FS", "T2", "T2FS"}


def plane_of(img: nib.Nifti1Image) -> tuple[str, float]:
    """(plane, obliquity in degrees) from the affine, as described in the module docstring."""
    zooms = np.array(img.header.get_zooms()[:3], float)
    if zooms.max() / zooms.min() < 1.2:
        return "3D/isotropic", float("nan")
    axis = int(np.argmax(zooms))                    # thickest voxel axis = slice direction
    d = img.affine[:3, axis] / np.linalg.norm(img.affine[:3, axis])  # its unit direction in world
    world = int(np.argmax(np.abs(d)))               # 0 = x (L-R), 1 = y (A-P), 2 = z (S-I)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(d[world])))))
    return {0: "sagittal", 1: "coronal", 2: "axial"}[world], tilt


def main():
    rows = []
    for cohort, cdir in COHORT_DIRS.items():
        for sdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            for f in sorted(sdir.glob("*.nii.gz")):
                m = FILE_RE.match(f.name)
                if not m:
                    continue
                kind = m.group(2)
                img = nib.load(str(f))                  # header only; voxel data not read
                plane, tilt = plane_of(img)
                rows.append(dict(cohort=cohort, subject=sdir.name, file=f.name,
                                 type="image" if kind in IMAGE_SEQS else "mask",
                                 series=kind, axcodes="".join(nib.aff2axcodes(img.affine)),
                                 plane=plane, tilt_deg=round(tilt, 1),
                                 spacing="x".join(f"{z:.2f}" for z in img.header.get_zooms()[:3]),
                                 excluded=sdir.name in EXCLUDED_SUBJECTS))
    d = pd.DataFrame(rows)

    def pct_table(g: pd.DataFrame, col: str) -> str:
        c = g[col].value_counts()
        return ", ".join(f"{k} {v} ({100 * v / len(g):.0f}%)" for k, v in c.items())

    print(f"{len(d)} NIfTI files ({(d.type == 'image').sum()} images, {(d.type == 'mask').sum()} label masks); "
          f"{d.excluded.sum()} belong to the excluded subjects {sorted(EXCLUDED_SUBJECTS)} and are included below.\n")

    print("=== 1. Orientation code on disk (RAS = already canonical) ===")
    for (cohort, typ), g in d.groupby(["cohort", "type"]):
        ras = (g.axcodes == "RAS").mean() * 100
        print(f"  {cohort} {typ:5s} (n={len(g):3d}): RAS {ras:5.1f}% | other {100 - ras:5.1f}%   [{pct_table(g, 'axcodes')}]")
    ras = (d.axcodes == "RAS").mean() * 100
    print(f"  ALL           (n={len(d):3d}): RAS {ras:5.1f}% | other {100 - ras:5.1f}%\n")

    print("=== 2. Acquisition plane (slice direction = thickest voxel axis) ===")
    for (cohort, typ), g in d.groupby(["cohort", "type"]):
        ax = (g.plane == "axial").mean() * 100
        print(f"  {cohort} {typ:5s} (n={len(g):3d}): axial {ax:5.1f}% | other {100 - ax:5.1f}%   [{pct_table(g, 'plane')}]")
    ax = (d.plane == "axial").mean() * 100
    print(f"  ALL           (n={len(d):3d}): axial {ax:5.1f}% | other {100 - ax:5.1f}%\n")

    print("=== 3. Images by series: plane and obliquity ===")
    im = d[d.type == "image"]
    for (cohort, s), g in im.groupby(["cohort", "series"]):
        t = g.tilt_deg.dropna()
        print(f"  {cohort} {s:5s} (n={len(g):3d}): {pct_table(g, 'plane')}; "
              f"tilt median {t.median():.1f} deg, max {t.max():.1f} deg; tilted >20 deg: {int((t > 20).sum())}")
    big = im[im.tilt_deg > 20].sort_values("tilt_deg", ascending=False)
    if len(big):
        print("\n  image series tilted more than 20 deg from their nearest plane:")
        for _, r in big.iterrows():
            print(f"    {r.file:24s} {r.plane:8s} tilt {r.tilt_deg:5.1f} deg  axcodes {r.axcodes}  spacing {r.spacing}")
    nonax = d[(d.type == "mask") & (d.plane != "axial")]
    if len(nonax):
        print(f"\n  label masks NOT on an axial grid ({len(nonax)}; side is still computed in world mm):")
        print("   ", ", ".join(nonax.file))


if __name__ == "__main__":
    main()

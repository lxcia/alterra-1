"""Step 0: build derived/inventory.csv, one row per subject.

What it does, for every subject folder of both cohorts (D1, D2):
  1. Sort the folder's files into image series (<sid>_T2FS.nii.gz, ...) and label masks
     (<sid>_<structure>[_r<rater>].nii.gz, structure = ut uterus, ov ovary, em endometrioma,
     cy cyst, pat undocumented). D1 has up to 3 raters per structure; D2 has one.
  2. Record per series: shape, voxel spacing, orientation code, tilt from axial, and an md5 hash
     of the voxel data (used to find duplicated images, see step 6).
  3. Load every mask in RAS+ and record its voxel count and which image grid it was drawn on.
  4. Midline reference = centroid of the uterus mask (union of raters), in world mm.
  5. For endometrioma (em) and cyst (cy): presence (union and majority of raters), and from the
     merged mask: number of separate lesions, side of the largest one, volume (mL) and
     superior-inferior extent (mm).
  6. After all subjects: list every series whose voxel data is byte-identical to another series,
     and print each group. This is how the D2-007 = D2-009 duplicate (and the others listed in
     PATCH.md P04) were found.

The CSV is the single source of cases and ground truth for every experiment
(`experiments/common.py` reads it). Excluded subjects stay in the CSV, with the reason in
`excluded`, and are dropped when cases are built.

Run:  .venv/bin/python src/inventory.py
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to

from io_orient import (COHORT_DIRS, DATA_ROOT, EXCLUDED_SUBJECTS, SEQUENCES, components,
                       fov_center_world, load_canonical, side_of, world_centroid)

OUT = Path(__file__).resolve().parents[1] / "derived" / "inventory.csv"
# Label file names: <subject>_<structure>.nii.gz (one rater) or <subject>_<structure>_r<N>.nii.gz
# (rater N; one file, D1-026_ut_re3, is spelled "_re<N>", which the optional "e" accepts).
#   group(1) = subject id, group(2) = structure, group(3) = rater or None.
# Structures present in the data: ut (uterus), ov (ovary), em (endometrioma), cy (cyst), and pat
# (3 files; meaning not documented, not used by any analysis).
# Example of what would be extraced by LABEL_RE = D1-001, em, 3
LABEL_RE = re.compile(r"^(D[12]-\d+)_([a-z]+)(?:_r(?:e)?(\d))?\.nii\.gz$")
# Image file names: <subject>_<sequence>.nii.gz. Sequence names are taken verbatim from the
# dataset; nothing checks the image content (that is why D2-042's "T2FS", a copy of its T2, was
# only found by the md5 check below).
# Example of what would be extracted by SEQ_RE = D2-054, T2FS
SEQ_RE = re.compile(r"^(D[12]-\d+)_(T1|T1FS|T2|T2FS)\.nii\.gz$")
MIDLINE_MARGIN_MM = 5.0  # lesions within 5 mm of the uterus centroid x are called 'midline'
BILATERAL_FRACTION = 0.5  # 'bilateral' if the largest opposite-side component is >= 50% of the largest


def scanner_table() -> dict[str, tuple[str, str]]:
    """D1 only: subject id -> (site, scanner) from the dataset's SiteScannerInfo.csv.
    Subjects missing from that table are recorded as 'unknown' (17 of the 45 D1 T2 subjects)."""
    df = pd.read_csv(DATA_ROOT / "SiteScannerInfo.csv")
    return {r["ID"]: (str(r["Site"]).strip(), str(r["MR Scanner"]).strip()) for _, r in df.iterrows()}


def _z_extent_mm(img: nib.Nifti1Image, mask: np.ndarray) -> float:
    """Superior-inferior extent of a mask in world mm (works for axial, sagittal, coronal grids).

    Maps every mask voxel to world coordinates, takes the spread of the z (superior) coordinate,
    and adds one voxel's extent along z so a single-slice mask has its slice thickness, not 0.
    
    As of 10/28 this is not used in any experiment just in the inventory CSV.
    """
    # Lists the (i,j,k) index of every voxel inside the mask
    ijk = np.argwhere(mask)
    # Converts every one of those voxels to millimetres and keeps only the z (head–foot) column
    z = nib.affines.apply_affine(img.affine, ijk)[:, 2]
    # Return the slice thickness
    return float(z.max() - z.min() + np.abs(img.affine[2, :3]).max())


def lesion_summary(img: nib.Nifti1Image, mask: np.ndarray, x_mid: float) -> dict:
    """Volume, components, side and superior-inferior extent (mm) of a RAS+ mask.

    `side` is the side of the LARGEST component (the question asked of the model is about the
    largest lesion), or 'bilateral' when the largest opposite-side component is at least
    BILATERAL_FRACTION of it. 

    `img` provides the grid/affine of `mask`; `x_mid` is the midline x coordinate (world mm)."""
    # mL per voxel = product of the three voxel sizes in mm, / 1000 (1 mL = 1000 mm^3)
    vox_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    # split the mask into separate lesions (blobs of >= 10 voxels, see io_orient.components)
    comps = components(mask)
    sides, vols, ext = [], [], []
    for c in comps:
        # side of each lesion from its centroid's world x relative to the midline (±5 mm = midline)
        x = world_centroid(img, c)[0]
        sides.append(side_of(x, x_mid, MIDLINE_MARGIN_MM))
        vols.append(c.sum() * vox_ml)
        ext.append(_z_extent_mm(img, c))
    # old rule (kept as side_anycomp): any lateral blob counts, both sides present -> bilateral
    uniq = sorted(set(s for s in sides if s != "midline"))
    if not uniq:
        anyc = "midline" if sides else "none"
    elif len(uniq) == 2:
        anyc = "bilateral"
    else:
        anyc = uniq[0]
    # current rule: side of the largest blob ...
    dom = sides[int(np.argmax(vols))] if vols else "none"
    side = dom
    # ... unless the biggest blob on the opposite side is at least half its size
    if dom in ("left", "right"):  # comparable lesions on both sides -> bilateral
        opp = [v for s, v in zip(sides, vols) if s == {"left": "right", "right": "left"}[dom]]
        if opp and max(opp) >= BILATERAL_FRACTION * max(vols):
            side = "bilateral"
    # comp_sides / comp_ml list every blob so the side decision can be audited from the CSV
    return dict(n_comp=len(comps), side=side, dominant_side=dom, side_anycomp=anyc,
                comp_sides=";".join(sides), comp_ml=";".join(f"{v:.1f}" for v in vols),
                volume_ml=float(mask.sum() * vox_ml),
                max_axial_extent_mm=round(max(ext), 1) if ext else 0.0,
                total_axial_extent_mm=round(_z_extent_mm(img, mask), 1) if mask.any() else 0.0)


def union_world(items: list) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Union of rater masks in WORLD space, on the first mask's grid (other grids resampled,
    nearest neighbour). Earlier code silently dropped raters drawn on another grid (P03).

    `items` is a list of (image, boolean array) pairs, one per rater. In D1 different raters
    sometimes drew on different series (e.g. one on T2, another on T1FS), so their arrays have
    different shapes or affines; those are resampled onto the first rater's grid using the
    affines, so the union is taken in physical space. Returns (reference image, union array)."""
    ref = items[0][0]
    u = np.zeros(ref.shape[:3], bool)
    for li, m in items:
        if li.shape[:3] == ref.shape[:3] and np.allclose(li.affine, ref.affine, atol=1e-3):
            # u and m are boolean NumPy arrays of the same shape (True = inside the mask). | works element by element: each voxel of u becomes True if it was already True or it's True in m.
            u |= m  # same grid: combine directly
        else:
            # different grid: nearest-neighbour resampling keeps the mask binary
            r = resample_from_to(nib.Nifti1Image(m.astype(np.uint8), li.affine), (ref.shape[:3], ref.affine), order=0)
            u |= np.asarray(r.dataobj) > 0
    return ref, u


def tilt_deg(img: nib.Nifti1Image) -> float:
    """Angle (deg) between the stacking axis of the RAS+ array and world z; 0 = true axial.

    Column 2 of the affine is the world-space direction of the array's third (slice) axis. For a
    true axial series it points straight along z. Six D1 series are tilted 30-45 deg, so the
    "axial" slices the model sees are really oblique."""
    c = img.affine[:3, 2]
    return float(np.degrees(np.arccos(min(1.0, abs(c[2]) / np.linalg.norm(c)))))


def main():
    scanners = scanner_table()
    rows = []
    # md5 of voxel data -> every "subject:sequence" with exactly that data (filled while looping)
    array_hashes = defaultdict(list)
    for cohort, cdir in COHORT_DIRS.items():
        for sdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            sid = sdir.name
            row: dict = {"cohort": cohort, "subject": sid}

            # ---- 1. sort files into image series and label masks
            seq_imgs = {}
            labels = defaultdict(dict)  # structure -> rater -> path
            for f in sorted(sdir.iterdir()):
                if m := SEQ_RE.match(f.name):
                    seq_imgs[m.group(2)] = f
                elif m := LABEL_RE.match(f.name):
                    labels[m.group(2)][m.group(3) or "1"] = f  # no rater suffix -> rater "1"
                else:
                    print("  unrecognised file:", f)
            row["sequences"] = ";".join(s for s in SEQUENCES if s in seq_imgs)
            row["excluded"] = EXCLUDED_SUBJECTS.get(sid, "")

            # ---- 2. per image series: geometry and a fingerprint of the voxel data
            for s in SEQUENCES:
                if s in seq_imgs:
                    # header info from the file as stored (not reoriented)
                    im = nib.load(str(seq_imgs[s]))
                    row[f"{s}_shape"] = "x".join(map(str, im.shape))
                    row[f"{s}_spacing"] = "x".join(f"{z:.2f}" for z in im.header.get_zooms()[:3])
                    row[f"{s}_axcodes"] = "".join(nib.aff2axcodes(im.affine))
                    row[f"{s}_tilt_deg"] = round(tilt_deg(load_canonical(seq_imgs[s])), 1)
                    # identical voxel data under another subject/sequence name (QC, P01/P04)
                    # The md5 is taken over the raw array bytes (header and file name ignored).
                    # Two series with the same hash contain exactly the same image, even if they
                    # are filed under different subjects or sequence names. Each hash is
                    # collected in array_hashes and compared across all subjects at the end.
                    h = hashlib.md5(np.asarray(im.dataobj).tobytes()).hexdigest()
                    array_hashes[h].append(f"{sid}:{s}")
                    row[f"{s}_md5"] = h
            # site / scanner: D1 from the dataset's table; D2 is a single site and scanner
            if cohort == "D1":
                row["site"], row["scanner"] = scanners.get(sid, ("unknown", "unknown"))
            else:
                row["site"], row["scanner"] = "TCPW", "Philips Ingenia 1.5 T"

            # ---- 3. load all label masks (canonical); record which sequence grid each lives on
            masks = {}  # (structure, rater) -> (RAS+ image, boolean array)
            for struct, raters in labels.items():
                for r, f in raters.items():
                    li = load_canonical(f)
                    m = np.asarray(li.dataobj) > 0  # any nonzero label value counts as inside
                    masks[(struct, r)] = (li, m)
                    row[f"{struct}_r{r}_vox"] = int(m.sum())
                    # which image series has the same array shape as this mask (by shape only;
                    # an empty value means the mask was drawn on a grid none of the images use,
                    # which is how D2-009's misplaced em mask shows up)
                    grid = [s for s in SEQUENCES if s in seq_imgs and
                            nib.load(str(seq_imgs[s])).shape[:3] == nib.load(str(f)).shape[:3]]
                    row[f"{struct}_r{r}_grid"] = ";".join(grid)
            # compact summary, e.g. "em:1,2,3;ov:1;ut:1,2,3" = which raters drew which structure
            row["labels"] = ";".join(f"{s}:{','.join(sorted(r))}" for s, r in sorted(labels.items()))

            # ---- 4. Uterus centroid = midline reference (world-space union over raters)
            x_mid, mid_ref = None, None
            ut = [masks[k] for k in masks if k[0] == "ut"]
            if ut:
                li, u = union_world(ut)
                if u.any():
                    x_mid, mid_ref = world_centroid(li, u)[0], "uterus"
            row["uterus_present"] = x_mid is not None
            # every rater who produced any label file for this subject (denominator for majority)
            all_raters = {k[1] for k in masks}
            row["n_raters"] = len(all_raters)

            # ---- 5. endometrioma (em) and cyst (cy): presence, then side / size of the lesions
            for struct in ["em", "cy"]:
                rs = [k for k in masks if k[0] == struct]
                if not rs:
                    # no label file for this structure: counted as absent. For D1 this may mean
                    # "not labelled" rather than "no lesion" (stated as a limitation).
                    row[f"{struct}_present"] = False
                    row[f"{struct}_label_files"] = 0
                    continue
                row[f"{struct}_label_files"] = len(rs)
                pos = [masks[k][1].any() for k in rs]  # did each rater draw a non-empty mask?
                row[f"{struct}_raters_positive"] = int(sum(pos))
                row[f"{struct}_present_union"] = any(pos)
                # majority over ALL raters who labelled this subject, not only those who drew this
                # structure (the earlier denominator made 'majority' identical to 'union', P03)
                row[f"{struct}_present_majority"] = sum(pos) * 2 > len(all_raters)
                row[f"{struct}_present"] = any(pos)  # primary rule: union over raters
                # Consensus mask for side/extent: world-space union of all raters
                li, u = union_world([masks[k] for k in rs])
                # midline = uterus centroid; without a uterus mask, the centre of the field of view
                ref = x_mid if x_mid is not None else fov_center_world(li)[0]
                row[f"{struct}_midline_ref"] = mid_ref or "fov_center"
                if u.any():
                    # adds em_side, em_volume_ml, em_n_comp, ... (cy_* for cysts)
                    for k, v in lesion_summary(li, u, ref).items():
                        row[f"{struct}_{k}"] = v
            rows.append(row)
            print(sid, row["sequences"], row["labels"],
                  "em:", row.get("em_side"), "cy:", row.get("cy_side"))
    df = pd.DataFrame(rows)

    # ---- 6. identical-array QC: list every other subject:sequence that has byte-identical voxel data
    # For each subject and sequence, look up its hash and list every OTHER "subject:sequence" that
    # has the same hash. Examples of what this finds: D2-007:T2 lists "D2-009:T2" (a duplicated
    # patient); D2-042:T2FS lists "D2-042:T2" (a sequence filed under the wrong name). Empty = unique.
    for s in SEQUENCES:
        if f"{s}_md5" in df:
            df[f"{s}_identical_to"] = [
                ";".join(x for x in array_hashes.get(h, []) if x != f"{sid}:{s}") if isinstance(h, str) else ""
                for sid, h in zip(df.subject, df[f"{s}_md5"])]
    # print each group of identical series once (a hash seen under more than one name)
    dups = {h: v for h, v in array_hashes.items() if len(v) > 1}
    for v in dups.values():
        print("  IDENTICAL VOXEL DATA:", " == ".join(v))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, df.shape)


if __name__ == "__main__":
    main()
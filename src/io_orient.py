"""Loading UT-EndoMRI volumes in a canonical orientation, and deriving lesion side.

This is the lowest layer of the data code: every other module that touches a NIfTI file goes
through `load_canonical` (or `pipeline.load_ras`, which calls it), so orientation is handled in one
place.

Conventions used everywhere in this repo
----------------------------------------
* Every volume is reoriented to RAS+ with ``nib.as_closest_canonical``.
  In RAS+ world/voxel space: +x -> patient RIGHT, +y -> ANTERIOR, +z -> SUPERIOR.
  So a lesion on the patient's LEFT has a SMALLER x than the midline.
* Axial slices are displayed in RADIOLOGICAL convention: anterior at the top of the
  image, the patient's RIGHT on the viewer's LEFT (as if looking up from the feet).
  For a RAS+ array ``vol[x, y, z]`` this is ``vol[::-1, ::-1, k].T``.
* Ground-truth side is computed in world (mm) coordinates from the mask centroid,
  relative to the uterus centroid when a uterus mask exists, otherwise relative to the
  centre of the image field of view. Array indices are never used for side.

Why this matters: NIfTI files store voxels in whatever order the scanner wrote them (D2 files are
LPS, D1 files a mix of LAS, RAS and LPS; masks are sometimes on sagittal or coronal grids). The
affine matrix in each file says how voxel indices map to physical millimetres. Working in RAS+ /
world coordinates makes "left" and "right" mean the same thing for every file.
"""
from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------- where the data lives
# UTENDO_DATA_ROOT overrides the default location of the UT-EndoMRI `data/` folder.
DATA_ROOT = Path(os.environ.get("UTENDO_DATA_ROOT", "/lotterlab/users/lucia/EndoSeg/data/UT-EndoMRI/data"))
# The dataset ships two cohorts in two folders. Subject IDs start with the cohort name
# ("D1-005", "D2-054"), which `subject_dir` relies on.
COHORT_DIRS = {"D1": DATA_ROOT / "D1_MHS", "D2": DATA_ROOT / "D2_TCPW"}
# MRI sequences a subject folder can contain (file names <subject>_<seq>.nii.gz). Not every subject
# has all four; `inventory.py` records which ones exist.
SEQUENCES = ["T1", "T1FS", "T2", "T2FS"]

# Subjects excluded from every experiment and analysis (PATCH.md, P01). D2-007 and D2-009 have
# byte-identical T1/T1FS/T2/T2FS files but contradictory labels (D2-007 lesion-free, D2-009 left
# endometrioma whose mask lies on a grid that matches none of its images), so neither ground
# truth can be trusted.
# How the exclusion is applied: `inventory.py` writes the reason into the `excluded` column of
# derived/inventory.csv; `experiments/common.inventory()` drops those rows, so no experiment builds a
# case for them; `analyze.load()` also drops their rows from every result CSV (for CSVs written
# before the exclusion existed). The duplicate itself was found by the md5 check in `inventory.py`.
EXCLUDED_SUBJECTS = {"D2-007": "identical images to D2-009, contradictory labels",
                     "D2-009": "identical images to D2-007; em mask matches no image grid"}


def subject_dir(subject: str) -> Path:
    """Folder of one subject, e.g. "D2-054" -> DATA_ROOT/D2_TCPW/D2-054 (cohort = first 2 chars)."""
    return COHORT_DIRS[subject[:2]] / subject


def load_canonical(path: Path | str) -> nib.Nifti1Image:
    """Load a NIfTI and reorient it to RAS+ (data array is reordered/flipped, affine updated).

    `as_closest_canonical` only permutes and flips array axes (no interpolation), choosing the
    closest RAS+ orientation. The voxel values are unchanged; the affine is updated so that
    voxel -> world mm mapping stays correct. For an oblique acquisition the array axes are only
    approximately aligned with R, A, S (`inventory.tilt_deg` measures how far off they are)."""
    return nib.as_closest_canonical(nib.load(str(path)))


def axial_slice_radiological(vol: np.ndarray, k: int) -> np.ndarray:
    """Return axial slice k of a RAS+ array in radiological display convention.

    vol[:, :, k] is indexed [x, y] = [left->right, posterior->anterior].
    [::-1, ::-1] flips both: x now runs right->left and y anterior->posterior.
    .T makes y the row axis and x the column axis, so in the displayed image:
    row 0 = most anterior (top of image), column 0 = patient RIGHT (viewer's left)."""
    return vol[::-1, ::-1, k].T


def to_radiological_stack(vol: np.ndarray) -> np.ndarray:
    """RAS+ array (x, y, z) -> stack of radiological axial images (z, rows, cols).

    Same flips as `axial_slice_radiological`, applied to every slice at once: the transpose
    (2, 1, 0) moves z to the front, so stack[k] == axial_slice_radiological(vol, k).
    Slice 0 is the most INFERIOR slice (z increasing = superior). `pipeline.prepare_from_array`
    reverses this so the model sees the most superior slice first."""
    return np.ascontiguousarray(np.transpose(vol[::-1, ::-1, :], (2, 1, 0)))


def world_centroid(mask_img: nib.Nifti1Image, mask: np.ndarray | None = None) -> np.ndarray | None:
    """Centroid of a binary mask in world (mm) coordinates, or None if the mask is empty.

    `mask_img` supplies the affine (and the mask itself if `mask` is not given). `mask` must be on
    `mask_img`'s voxel grid. The centroid is computed in voxel indices (center_of_mass) and then
    mapped to millimetres with the affine; element [0] of the result is the RAS+ x coordinate used
    for left/right."""
    m = np.asarray(mask_img.dataobj) > 0 if mask is None else mask
    if not m.any():
        return None
    ijk = np.array(ndimage.center_of_mass(m))
    return nib.affines.apply_affine(mask_img.affine, ijk)


def fov_center_world(img: nib.Nifti1Image) -> np.ndarray:
    """World coordinate of the centre voxel of an image.

    Fallback midline when a subject has no uterus mask (9 subjects): side is then measured
    relative to the middle of the scanned field of view."""
    # Line below computes the voxel index of the exact centre of the array
    # img.shape[:3] --> number of voxels along each axis
    # np.array(img.shape[:3]) - 1 --> index of the LAST voxel on each axis
    # (...) / 2.0 --> midpoint
    ijk = (np.array(img.shape[:3]) - 1) / 2.0
    # Line below turns that index into a physical position 
    return nib.affines.apply_affine(img.affine, ijk)


def side_of(x_lesion: float, x_midline: float, margin_mm: float = 0.0) -> str:
    """'left'/'right' (patient's) from RAS+ x coordinates; 'midline' if within margin.

    Larger x = further to the patient's right, so a lesion right of the midline has d > 0."""
    d = x_lesion - x_midline
    if abs(d) <= margin_mm:
        return "midline"
    return "right" if d > 0 else "left"


def components(mask: np.ndarray, min_voxels: int = 10):
    """Connected components (26-connectivity) (3x3 block of 1s) of a binary mask, dropping tiny specks.

    Returns a list of boolean arrays, one per separate blob (voxels touching by face, edge or
    corner belong to the same blob). Blobs smaller than `min_voxels` are treated as annotation
    noise and dropped. Used by `inventory.lesion_summary` to find the largest lesion and its side."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    out = []
    # Loop pulls out each blob as its own boolean mask and keeps it only if it has at least 10 min_voxels
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() >= min_voxels:
            out.append(c)
    return out

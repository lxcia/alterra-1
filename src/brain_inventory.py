"""BraTS 2024 GLI (training subset in derived/brain_mri_train/): tumour side from the segmentation.

Side = whole-tumour centroid x vs brain centroid x in world RAS+ mm (+x = patient right); BraTS
volumes are skull-stripped, so nonzero voxels = brain. |dx| <= 10 mm -> 'midline' (excluded from
laterality analyses). Also saves a radiological-convention QC image of the first 4 cases.

This is the ground truth for positive control P2 (does the model read tumour side from brain MRI?).
It uses the same io_orient functions (RAS+ loading, world centroids, side_of) as the pelvic
inventory, so the left/right convention is identical. Each case is a pair of files:
<case>-t2w.nii.gz (the image) and <case>-seg.nii.gz (tumour labels; any nonzero label = tumour).

Run: python src/brain_inventory.py -> derived/brain_inventory.csv, derived/qc/brain_side_qc.png
"""
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from io_orient import axial_slice_radiological, load_canonical, side_of, world_centroid

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted(glob.glob(str(ROOT / "derived" / "brain_mri_train" / "*-t2w.nii.gz"))):
    img = load_canonical(f)
    seg = load_canonical(f.replace("-t2w", "-seg"))
    vol, s = np.asarray(img.dataobj), np.asarray(seg.dataobj) > 0
    tx = world_centroid(seg, s)[0]       # tumour centroid, world x (mm)
    bx = world_centroid(img, vol > 0)[0]  # brain centroid (all nonzero voxels), world x = midline
    k = int(np.argmax(s.sum(axis=(0, 1))))  # axial slice with the most tumour (for the QC image)
    rows.append(dict(subject=Path(f).name.replace("-t2w.nii.gz", ""), path=f, dx_mm=round(tx - bx, 1),
                     side=side_of(tx, bx, 10.0), tumour_ml=round(s.sum() * np.prod(seg.header.get_zooms()[:3]) / 1000, 1),
                     max_slice=k, n_slices=vol.shape[2]))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "derived" / "brain_inventory.csv", index=False)
print(df.side.value_counts().to_dict(), "median tumour mL", df.tumour_ml.median())
# QC: first 4 cases, tumour outlined in red, R/L marked, side and offset in the title
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (_, r) in zip(axes, df.head(4).iterrows()):
    img, seg = load_canonical(r.path), load_canonical(r.path.replace("-t2w", "-seg"))
    ax.imshow(axial_slice_radiological(np.asarray(img.dataobj), r.max_slice), cmap="gray")
    ax.contour(axial_slice_radiological((np.asarray(seg.dataobj) > 0).astype(float), r.max_slice), [0.5], colors="r")
    ax.set_title(f"{r.subject}\nside={r.side} (dx={r.dx_mm} mm)", fontsize=8)
    ax.text(0.02, 0.5, "R", color="yellow", transform=ax.transAxes, weight="bold")
    ax.text(0.93, 0.5, "L", color="yellow", transform=ax.transAxes, weight="bold")
    ax.axis("off")
fig.suptitle("BraTS: radiological convention (patient right on viewer left)", fontsize=9)
fig.savefig(ROOT / "derived" / "qc" / "brain_side_qc.png", dpi=80, bbox_inches="tight")

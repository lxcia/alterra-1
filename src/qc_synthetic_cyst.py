"""QC figure for expC2's synthetic cyst insertion: none / left / right / bilateral at the
uterus-centroid slice, radiological convention. -> derived/qc/synthetic_cyst_<subject>.png

Run: python src/qc_synthetic_cyst.py D2-001"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from io_orient import axial_slice_radiological, world_centroid
from perturb import insert_sphere
from pipeline import load_ras, normalize
from qc_overlay import mask_on, union_mask

sid = sys.argv[1] if len(sys.argv) > 1 else "D2-001"
img, vol = load_ras(sid, "T2FS")
ut = mask_on(img, union_mask(sid, "ut"))
cen = world_centroid(img, ut)
bright = float(np.percentile(vol, 99.5))
L, R = cen - [45.0, 0, 0], cen + [45.0, 0, 0]
vols = {"none": vol, "left": insert_sphere(img, vol, L, 18.0, bright), "right": insert_sphere(img, vol, R, 18.0, bright)}
vols["bilateral"] = insert_sphere(img, vols["left"], R, 18.0, bright)
k = int(round(nib.affines.apply_affine(np.linalg.inv(img.affine), cen)[2]))
fig, ax = plt.subplots(1, 4, figsize=(14, 4))
for a, (name, v) in zip(ax, vols.items()):
    a.imshow(axial_slice_radiological(normalize(v), k), cmap="gray")
    a.set_title(f"{sid} synthetic cyst: patient {name}")
    a.text(0.02, 0.5, "R", color="yellow", transform=a.transAxes, fontsize=14, weight="bold")
    a.text(0.94, 0.5, "L", color="yellow", transform=a.transAxes, fontsize=14, weight="bold")
    a.axis("off")
fig.suptitle("SYNTHETIC: radiological convention (patient right on viewer left)")
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "derived" / "qc" / f"synthetic_cyst_{sid}.png"
fig.savefig(out, dpi=70)
print("wrote", out)

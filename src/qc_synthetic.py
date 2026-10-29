"""QC figure for the contrast counterfactuals of expA2 (one row per case, max-lesion slice).

Run: python src/qc_synthetic.py D2-077 D2-002 ...  -> derived/qc/synthetic_contrast.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "experiments"))
from io_orient import axial_slice_radiological
from perturb import set_lesion
from pipeline import load_ras, normalize
from qc_overlay import mask_on, union_mask
from inventory import OUT as INV
import pandas as pd

sys.argv += [] if len(sys.argv) > 1 else ["D2-077", "D2-002"]
seq = "T1FS"
inv = pd.read_csv(INV).set_index("subject")
subs = sys.argv[1:]
fig, axes = plt.subplots(len(subs), 4, figsize=(13, 3.4 * len(subs)), squeeze=False)
for r, sid in enumerate(subs):
    img, vol = load_ras(sid, seq)
    kind = "em" if inv.loc[sid, "em_present"] == True else "cy"
    mask = mask_on(img, union_mask(sid, kind))
    k = int(np.argmax(mask.sum(axis=(0, 1))))
    for c, (name, ratio) in enumerate([("ref", "ref"), ("lesion_bright", 2.0), ("lesion_dark", 0.5), ("lesion_filled", None)]):
        v = vol if ratio == "ref" else set_lesion(vol, mask, ratio)[0]
        u8 = normalize(v)
        axes[r, c].imshow(axial_slice_radiological(u8, k), cmap="gray", vmin=0, vmax=255)
        axes[r, c].set_title(f"{sid} ({kind}) {name}", fontsize=9)
        axes[r, c].axis("off")
fig.suptitle("SYNTHETIC contrast counterfactuals (T1FS), radiological convention", fontsize=10)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "derived" / "qc" / "synthetic_contrast.png"
fig.savefig(out, dpi=70)
print("wrote", out)

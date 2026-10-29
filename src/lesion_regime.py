"""Place the real D2 lesions on the psychometric sweep's axes (expC4): mean T2FS lesion intensity
as a percentile of the patient's own volume (the sweep's contrast axis: bright=99.5, moderate=90,
dark=1) and the equivalent-sphere radius in mm. Writes results/lesion_regime.csv.

Run: python src/lesion_regime.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from io_orient import components
from pipeline import load_ras
from qc_overlay import mask_on, union_mask

ROOT = Path(__file__).resolve().parents[1]
inv = pd.read_csv(ROOT / "derived" / "inventory.csv")
d2 = inv[(inv.cohort == "D2") & inv.sequences.fillna("").str.contains("T2FS")]
rows = []
for _, c in d2.iterrows():
    for kind, side_col in [("em", "em_side"), ("cy", "cy_side")]:
        if c.get(f"{kind}_present") != True or c[side_col] not in ("left", "right"):
            continue
        img, vol = load_ras(c.subject, "T2FS")
        m = mask_on(img, union_mask(c.subject, kind))
        comp = max(components(m), key=lambda x: x.sum()) if m.any() else m
        vox_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000
        ml = comp.sum() * vox_ml
        rows.append(dict(subject=c.subject, lesion=kind, side=c[side_col], volume_ml=round(ml, 1),
                         equiv_radius_mm=round((3 * ml * 1000 / (4 * np.pi)) ** (1 / 3), 1),
                         intensity_pct=round(percentileofscore(vol.ravel()[::7], float(vol[comp].mean())), 1)))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "lesion_regime.csv", index=False)
print(df.to_string())
print(df.groupby("lesion")[["equiv_radius_mm", "intensity_pct"]].median())

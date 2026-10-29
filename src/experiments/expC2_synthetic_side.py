"""Lead C mechanism: SYNTHETIC cyst insertion with known side (counterfactual ground truth).

On lesion-free D2 volumes that have a uterus mask, paste one large, obviously bright,
spherical "cyst" (radius 18 mm; intensity = 99.5th percentile of the volume, blended with a
soft mask) 45 mm to the patient's LEFT or RIGHT of the uterus centroid (world mm), at the
uterus centroid's height and AP position. Conditions: none | left | right | bilateral.
Questions:
  Q_SIDE  (score + optional generate): which side is the largest ovarian cystic lesion?
  Q_CYST  (score): is there an adnexal cyst?
If side answers do not follow the inserted cyst, laterality is not read from the image even
when the lesion is large, unique, and unambiguous.

Run: python src/experiments/expC2_synthetic_side.py [--model medgemma] [--seq T2FS] [--generate]
"""
import argparse

import numpy as np
import pandas as pd

from common import FINAL_YN, Q_SIDE, d2_cases, out_dir, save
from io_orient import world_centroid
from models import VLM
from parse import parse_side
from perturb import insert_sphere
from pipeline import build_messages, load_ras, prepare_from_array
from qc_overlay import mask_on, union_mask

Q_CYST = ("\n\nBased on the visual evidence in the slices provided above, is there an ovarian or adnexal "
          "cystic lesion? " + FINAL_YN)
OFFSET_MM, RADIUS_MM = 45.0, 18.0

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--generate", action="store_true")
args = ap.parse_args()


cases = d2_cases(require=(args.seq,))
cases = cases[(cases.label == "negative") & (cases.uterus_present == True)].reset_index(drop=True)
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    img, vol = load_ras(c.subject, args.seq)
    ut = mask_on(img, union_mask(c.subject, "ut"))
    if not ut.any():
        continue
    cen = world_centroid(img, ut)
    bright = float(np.percentile(vol, 99.5))
    # RAS+: +x is the patient's RIGHT
    left, right = cen - [OFFSET_MM, 0, 0], cen + [OFFSET_MM, 0, 0]
    vols = {"none": vol, "left": insert_sphere(img, vol, left, RADIUS_MM, bright), "right": insert_sphere(img, vol, right, RADIUS_MM, bright)}
    vols["bilateral"] = insert_sphere(img, vols["left"], right, RADIUS_MM, bright)
    for cond, v in vols.items():
        vin = prepare_from_array(v, c.subject, args.seq, tags={"synthetic_cyst": cond})
        meta = {"exp": "expC2", "subject": c.subject, "seq": args.seq, "cond": cond}
        msgs, _ = build_messages(vin, Q_SIDE)
        ss = m.score(msgs, [" left", " right", " bilateral", " none"], assistant_prefix="Final Answer:", meta=meta)
        msgs_c, _ = build_messages(vin, Q_CYST)
        sc = m.score(msgs_c, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
        row = dict(subject=c.subject, cond=cond, side_argmax=max(ss, key=ss.get).strip(),
                   **{f"lp_{k.strip()}": val for k, val in ss.items()}, cyst_logit=sc[" yes"] - sc[" no"])
        if args.generate:
            out = m.generate(msgs, meta=meta)
            row.update(side=parse_side(out)[0], side_raw=out, truncated=m.truncated())
        rows.append(row)
        print(c.subject, cond, row["side_argmax"], row.get("side"), round(row["lp_right"] - row["lp_left"], 2),
              round(row["cyst_logit"], 2), flush=True)
save(pd.DataFrame(rows), out_dir("expC2_synthetic_side") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}.csv")

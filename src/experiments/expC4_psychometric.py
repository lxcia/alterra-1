"""Failure 1 mechanism: psychometric sweep. When does side information reach the answer?

On lesion-free D2 T2FS volumes with a uterus mask, paste ONE SYNTHETIC sphere and vary:
  radius   : 5, 10, 15, 20, 30 mm
  contrast : bright (99.5th pct of the volume), moderate (90th pct), dark (1st pct of BODY voxels,
             i.e. voxels above the volume median; the 1st pct of the whole volume is air = 0)
  distance : 30, 45, 60 mm lateral to the uterus centroid
  side     : patient left / right
The sphere is blended through a mask blurred with sigma 1 mm (not 1 voxel; P08), and the contrast
actually reached is recorded (`achieved` = fraction of the way from the original to the target
intensity, averaged over the sphere).
Question: OPEN format (no option list; expC3.Q_OPEN). Outcomes:
  * gap = log P(right) - log P(left) after "Final Answer:" (direct answer), each summed over the
    surface forms " right"/" Right" and " left"/" Left" (the model writes "Left"/"Right" here; the
    lowercase-only gap is kept as gap_lower, and `mass_lr` = total probability of the four forms),
    compared with the same volume without a sphere: shift = gap(cond) - gap(none). Correct
    direction = shift > 0 for a right sphere, < 0 for a left one. (Grid, cheap.)
  * generated open answer for the bright, 45 mm row (all radii, both sides).

Run: python src/experiments/expC4_psychometric.py [--model medgemma] [--n 20] [--shard i --nshards n]
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from common import d2_cases, out_dir, save
from io_orient import world_centroid
from models import VLM, logsumexp_forms
from parse import parse_side
from perturb import insert_sphere, sphere_mask
from pipeline import build_messages, load_ras, prepare_from_array
from qc_overlay import mask_on, union_mask

Q_OPEN = ("\n\nBased on the visual evidence in the slices provided above, on which side of the patient is "
          "the largest ovarian cystic lesion? Please provide your reasoning and conclude with "
          "'Final Answer: ' followed by the side.")
RADII = [5, 10, 15, 20, 30]
CONTRASTS = {"bright": 99.5, "moderate": 90, "dark": 1}  # dark: percentile of body voxels
FORMS = {"left": [" left", " Left"], "right": [" right", " Right"]}
SIGMA_MM = 1.0
DISTS = [30, 45, 60]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--reverse", action="store_true", help="traverse cases in reverse (extra parallel worker)")
args = ap.parse_args()

cases = d2_cases(require=("T2FS",))
cases = cases[(cases.label == "negative") & (cases.uterus_present == True)].head(args.n).reset_index(drop=True)
m = VLM(args.model)
rows = []


def gap_of(vin, meta):
    msgs, _ = build_messages(vin, Q_OPEN)
    sc = m.score(msgs, FORMS["left"] + FORMS["right"], assistant_prefix="Final Answer:", meta=meta)
    lp_l, lp_r = logsumexp_forms(sc, FORMS["left"]), logsumexp_forms(sc, FORMS["right"])
    return dict(gap=lp_r - lp_l, gap_lower=sc[" right"] - sc[" left"],
                mass_lr=float(np.exp(lp_l) + np.exp(lp_r))), msgs


def target_value(vol, cname):
    if cname == "dark":
        return float(np.percentile(vol[vol > np.percentile(vol, 50)], CONTRASTS[cname]))
    return float(np.percentile(vol, CONTRASTS[cname]))


order = list(cases.iterrows())[::-1] if args.reverse else list(cases.iterrows())
for i, c in order:
    if i % args.nshards != args.shard:
        continue
    m.refresh_cache()
    img, vol = load_ras(c.subject, "T2FS")
    cen = world_centroid(img, mask_on(img, union_mask(c.subject, "ut")))
    g0, _ = gap_of(prepare_from_array(vol, c.subject, "T2FS"), {"exp": "expC4", "subject": c.subject, "cond": "none"})
    for r, cname, d, side in itertools.product(RADII, CONTRASTS, DISTS, ["left", "right"]):
        centre = cen + [d if side == "right" else -d, 0, 0]  # RAS+: +x = patient right
        val = target_value(vol, cname)
        v = insert_sphere(img, vol, centre, float(r), val, sigma_mm=SIGMA_MM)
        sm = sphere_mask(img, vol.shape, centre, float(r))
        achieved = float(np.mean((v[sm] - vol[sm]) / np.where(np.abs(val - vol[sm]) > 1e-6, val - vol[sm], np.nan))) \
            if sm.any() else float("nan")
        vin = prepare_from_array(v, c.subject, "T2FS", tags={"sphere": (r, cname, d, side)})
        meta = {"exp": "expC4", "subject": c.subject, "r": r, "contrast": cname, "d": d, "side": side}
        g, msgs = gap_of(vin, meta)
        shift = g["gap"] - g0["gap"]
        row = dict(subject=c.subject, radius_mm=r, contrast=cname, dist_mm=d, side=side, target=val,
                   achieved=achieved, n_vox=int(sm.sum()), gap=g["gap"], gap_none=g0["gap"], shift=shift,
                   correct_dir=(shift > 0) if side == "right" else (shift < 0),
                   gap_lower=g["gap_lower"], shift_lower=g["gap_lower"] - g0["gap_lower"],
                   mass_lr=g["mass_lr"], mass_lr_none=g0["mass_lr"])
        if cname == "bright" and d == 45:
            out = m.generate(msgs, meta=meta)
            row.update(answer=parse_side(out)[0], raw=out, truncated=m.truncated())
        rows.append(row)
    print(c.subject, "done", flush=True)
save(pd.DataFrame(rows), out_dir("expC4_psychometric") /
                          f"{args.model}_shard{args.shard}of{args.nshards}{'_rev' if args.reverse else ''}.csv")

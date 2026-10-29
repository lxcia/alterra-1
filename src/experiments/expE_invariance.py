"""Extras: perturbations that should NOT change a radiologist's answer (except inversion,
which destroys the contrast physics and SHOULD change it), and prompt synonyms.

Image perturbations (on the reference T2FS / T1FS stack):
  win_1_99, win_0_100 (percentile window), gamma_0.6, gamma_1.6, invert, crop_70 (central 70%
  FOV), rot_10 (10 deg in-plane rotation)
Prompt perturbations (reference image):
  syn_chocolate ('chocolate cyst'), syn_endometriotic ('ovarian endometriotic cyst'),
  order_noyes ('Final Answer: no' or 'Final Answer: yes' option order swapped)
Outcome: logit yes/no for the endometrioma question.

Run: python src/experiments/expE_invariance.py [--model medgemma] [--seq T2FS] [--shard i --nshards n]
"""
import argparse

import pandas as pd

from common import Q_ENDO, d2_cases, out_dir, save
from models import VLM
from perturb import crop_fov, gamma, invert, rotate_inplane
from pipeline import build_messages, load_ras, prepare_from_array

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--neg", type=int, default=15)
args = ap.parse_args()

PROMPTS = {
    "ref": Q_ENDO,
    "syn_chocolate": Q_ENDO.replace("an ovarian endometrioma", "an ovarian chocolate cyst"),
    "syn_endometriotic": Q_ENDO.replace("an ovarian endometrioma", "an ovarian endometriotic cyst"),
    "order_noyes": Q_ENDO.replace("'Final Answer: yes' or 'Final Answer: no'", "'Final Answer: no' or 'Final Answer: yes'"),
}
assert all(PROMPTS[k] != Q_ENDO for k in PROMPTS if k != "ref")

cases = d2_cases(require=(args.seq,))
cases = pd.concat([cases[cases.label != "negative"], cases[cases.label == "negative"].head(args.neg)]).reset_index(drop=True)
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    _, vol = load_ras(c.subject, args.seq)
    P = lambda v, **kw: prepare_from_array(v, c.subject, args.seq, **kw)
    imgs = {"ref": P(vol), "win_1_99": P(vol, p_lo=1, p_hi=99), "win_0_100": P(vol, p_lo=0, p_hi=100),
            "gamma_0.6": P(gamma(vol, 0.6)), "gamma_1.6": P(gamma(vol, 1.6)), "invert": P(invert(vol)),
            "crop_70": P(crop_fov(vol, 0.7)), "rot_10": P(rotate_inplane(vol, 10))}
    conds = [(k, v, Q_ENDO) for k, v in imgs.items()] + [(k, imgs["ref"], q) for k, q in PROMPTS.items() if k != "ref"]
    for name, vin, q in conds:
        msgs, _ = build_messages(vin, q)
        sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:",
                     meta={"exp": "expE", "subject": c.subject, "seq": args.seq, "cond": name})
        rows.append(dict(subject=c.subject, label=c.label, seq=args.seq, cond=name, logit_yes_no=sc[" yes"] - sc[" no"]))
    print(c.subject, c.label, {r["cond"]: round(r["logit_yes_no"], 1) for r in rows[-len(conds):]}, flush=True)
save(pd.DataFrame(rows), out_dir("expE_invariance") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}.csv")

"""Lead D: how does the call depend on WHICH slices are shown, and in what order?

For every D2 lesion case (endometrioma or cyst) and a set of negatives:
  ref          : reference pipeline (all slices, superior first)
  no_lesion    : every slice that contains lesion mask removed (the lesion is invisible)
  lesion_only  : only slices containing the lesion (+-1 slice)
  reversed     : slice order reversed (inferior first)
  shuffled     : random slice order (seed 0)
  stride2_o0/1 : every 2nd slice, grid offset 0 or 1
  single_max   : only the single slice with the largest lesion cross-section
Outcomes: logit yes/no for Q_ENDO and side log-probs for Q_SIDE (scores only; fast).
If 'no_lesion' does NOT lower P(yes) for true endometriomas, the call is not driven by the lesion.

Run: python src/experiments/expD_slices.py [--model medgemma] [--seq T2FS] [--shard i --nshards n]
"""
import argparse

import numpy as np
import pandas as pd

from common import Q_ENDO, Q_SIDE, d2_cases, out_dir, save
from models import VLM
from perturb import keep_z, reverse_order, shuffle_order, single_slice, stride
from pipeline import build_messages, load_ras, prepare_from_array
from qc_overlay import mask_on, union_mask

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--neg", type=int, default=15)
args = ap.parse_args()

cases = d2_cases(require=(args.seq,))
cases = pd.concat([cases[cases.label != "negative"], cases[cases.label == "negative"].head(args.neg)]).reset_index(drop=True)
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    img, vol = load_ras(c.subject, args.seq)
    ref = prepare_from_array(vol, c.subject, args.seq)
    conds = {"ref": ref, "reversed": reverse_order(ref), "shuffled": shuffle_order(ref, 0),
             "stride2_o0": stride(ref, 2, 0), "stride2_o1": stride(ref, 2, 1)}
    lesion_z = []
    if c.label != "negative":
        mask = mask_on(img, union_mask(c.subject, "em" if c.label == "endometrioma" else "cy"))
        area = mask.sum(axis=(0, 1))
        lesion_z = [int(z) for z in np.nonzero(area)[0]]
        near = {z + d for z in lesion_z for d in (-1, 0, 1)}
        conds["no_lesion"] = keep_z(ref, set(ref.z_indices) - set(lesion_z))
        conds["lesion_only"] = keep_z(ref, near)
        conds["single_max"] = single_slice(ref, int(np.argmax(area)))
    for name, vin in conds.items():
        meta = {"exp": "expD", "subject": c.subject, "seq": args.seq, "cond": name}
        msgs, _ = build_messages(vin, Q_ENDO)
        sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
        msgs_s, _ = build_messages(vin, Q_SIDE)
        ss = m.score(msgs_s, [" left", " right", " bilateral", " none"], assistant_prefix="Final Answer:", meta=meta)
        rows.append(dict(subject=c.subject, label=c.label, cond=name, n_shown=len(vin.slices),
                         n_lesion_slices=len(lesion_z), logit_yes_no=sc[" yes"] - sc[" no"],
                         **{f"lp_{k.strip()}": v for k, v in ss.items()}))
        print(c.subject, c.label, name, len(vin.slices), round(rows[-1]["logit_yes_no"], 2), flush=True)
save(pd.DataFrame(rows), out_dir("expD_slices") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}.csv")

"""Lead A mechanism: mask-guided contrast counterfactuals (SYNTHETIC images, labelled as such).

Hold everything fixed except the signal intensity INSIDE the labelled lesion, then ask
whether the endometrioma call follows the physics:
  T1FS:  lesion_bright = lesion scaled to 2.0 x its surrounding-tissue ring mean
                         (endometrioma-like T1 hyperintensity)
         lesion_dark   = lesion scaled to 0.5 x ring mean (simple-cyst-like T1 hypointensity)
         lesion_filled = lesion replaced by the ring mean (lesion erased; crude inpainting)
A model that reads T1 contrast should give P(yes | bright) > P(yes | dark) for the SAME lesion.

The scaling is applied with a softly blurred mask (sigma 1 voxel) to avoid hard edges.
Cases: every D2 endometrioma or cyst with the chosen sequence.

Run: python src/experiments/expA2_contrast.py [--model medgemma] [--seq T1FS] [--generate]
"""
import argparse

import pandas as pd

from common import Q_ENDO, Q_TYPE, d2_cases, out_dir, save
from models import VLM
from parse import parse_lesion_type, parse_yesno
from perturb import set_lesion
from pipeline import build_messages, load_ras, prepare_from_array
from qc_overlay import mask_on, union_mask

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T1FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--generate", action="store_true")
args = ap.parse_args()


cases = d2_cases(require=(args.seq,))
cases = cases[cases.label != "negative"].reset_index(drop=True)
m = VLM(args.model)
rows = []
for i, c in cases.iterrows():
    if i % args.nshards != args.shard:
        continue
    img, vol = load_ras(c.subject, args.seq)
    mask = mask_on(img, union_mask(c.subject, "em" if c.label == "endometrioma" else "cy"))
    if not mask.any():
        continue
    vols = {"ref": (vol, None)}
    for name, r in [("lesion_bright", 2.0), ("lesion_dark", 0.5), ("lesion_filled", None)]:
        vols[name] = set_lesion(vol, mask, r)
    orig_ratio = set_lesion(vol, mask, 1.0)[1]
    for name, (v, _) in vols.items():
        vin = prepare_from_array(v, c.subject, args.seq, tags={"synthetic": name})
        meta = {"exp": "expA2", "subject": c.subject, "seq": args.seq, "cond": name}
        msgs, _ = build_messages(vin, Q_ENDO)
        sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
        row = dict(subject=c.subject, label=c.label, seq=args.seq, cond=name, orig_ratio=orig_ratio,
                   logit_yes_no=sc[" yes"] - sc[" no"])
        if args.generate:
            out = m.generate(msgs, meta={**meta, "q": "endo"})
            msgs_t, _ = build_messages(vin, Q_TYPE)
            out_t = m.generate(msgs_t, meta={**meta, "q": "type"})
            row.update(endo=parse_yesno(out)[0], lesion_type=parse_lesion_type(out_t)[0], endo_raw=out, type_raw=out_t)
        rows.append(row)
        print(c.subject, c.label, name, round(orig_ratio, 2), round(row["logit_yes_no"], 2), row.get("endo"),
              row.get("lesion_type"), flush=True)
save(pd.DataFrame(rows), out_dir("expA2_contrast") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}.csv")

"""Lead C mechanism: where does the constant 'left' come from?

Cases with KNOWN side:
  * real D2 lesions (endometrioma or cyst) lateralised left/right by mask (world coordinates);
  * SYNTHETIC: lesion-free D2 volumes with an 18 mm bright cyst inserted 45 mm left or right
    of the uterus centroid (as in expC2).
Prompt variants (all with the reference image stack unless stated):
  ref          : Q_SIDE (options listed left, right, bilateral, none)
  order_rl     : options listed right, left, bilateral, none   (position-bias test)
  open         : no option list; 'conclude with Final Answer: <side>'
  yn_right     : "Is the largest ovarian cystic lesion on the patient's RIGHT side?" yes/no
  yn_left      : same for LEFT
  single_slice : ref prompt, but only the one slice through the lesion centre (2D)
Outcome: generated answer (parsed), plus next-token log-probs.

Run: python src/experiments/expC3_side_prompt.py [--model medgemma] [--seq T2FS] [--shard i --nshards n]
"""
import argparse

import nibabel as nib
import numpy as np
import pandas as pd

from common import FINAL_YN, Q_SIDE, d2_cases, out_dir, save
from io_orient import world_centroid
from models import VLM, logsumexp_forms
from parse import parse_side, parse_yesno
from perturb import insert_sphere, single_slice
from pipeline import build_messages, load_ras, prepare_from_array
from qc_overlay import mask_on, union_mask

Q_ORDER_RL = Q_SIDE.replace("'Final Answer: left', 'Final Answer: right'", "'Final Answer: right', 'Final Answer: left'")
Q_OPEN = ("\n\nBased on the visual evidence in the slices provided above, on which side of the patient is "
          "the largest ovarian cystic lesion? Please provide your reasoning and conclude with "
          "'Final Answer: ' followed by the side.")
Q_YN = ("\n\nBased on the visual evidence in the slices provided above, is the largest ovarian cystic lesion "
        "on the patient's {side} side? " + FINAL_YN)
assert Q_ORDER_RL != Q_SIDE
SIDE_CANDS = [" left", " right", " bilateral", " none"]
# The open variant has no option list and the model answers "Left"/"Right": score both cases and
# sum them (P09); lp_* are then log-probabilities summed over the two surface forms.
# " Bilateral" is two tokens for both MedGemma and Qwen, so only its lowercase form is scored.
OPEN_FORMS = {"left": [" left", " Left"], "right": [" right", " Right"], "bilateral": [" bilateral"],
              "none": [" none", " None"]}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--seq", default="T2FS")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--n-synth", type=int, default=20, help="lesion-free cases used for synthetic insertion")
ap.add_argument("--variants", default="ref,order_rl,open,yn_right,yn_left,single_slice",
                help="comma-separated subset of prompt variants to run")
args = ap.parse_args()

cases = d2_cases(require=(args.seq,))
items = []  # (subject, source, gt_side)
for _, c in cases.iterrows():
    gt = c.em_side if c.label == "endometrioma" else c.cy_side if c.label == "cyst" else None
    if gt in ("left", "right"):
        items.append((c.subject, "real_" + c.label, gt))
neg = cases[(cases.label == "negative") & (cases.uterus_present == True)].head(args.n_synth)
for s in neg.subject:
    items += [(s, "synthetic", "left"), (s, "synthetic", "right")]

m = VLM(args.model)
rows = []
for i, (sid, source, gt) in enumerate(items):
    if i % args.nshards != args.shard:
        continue
    img, vol = load_ras(sid, args.seq)
    if source == "synthetic":
        ut = mask_on(img, union_mask(sid, "ut"))
        cen = world_centroid(img, ut)
        centre = cen + [45.0 if gt == "right" else -45.0, 0, 0]  # RAS+: +x = patient right
        vol = insert_sphere(img, vol, centre, 18.0, float(np.percentile(vol, 99.5)))
        z_centre = int(round(nib.affines.apply_affine(np.linalg.inv(img.affine), centre)[2]))
    else:
        les = mask_on(img, union_mask(sid, "em" if "endometrioma" in source else "cy"))
        z_centre = int(np.argmax(les.sum(axis=(0, 1))))
    ref = prepare_from_array(vol, sid, args.seq, tags={"source": source, "gt": gt})
    variants = {"ref": (ref, Q_SIDE), "order_rl": (ref, Q_ORDER_RL), "open": (ref, Q_OPEN),
                "yn_right": (ref, Q_YN.format(side="right")), "yn_left": (ref, Q_YN.format(side="left"))}
    if z_centre in ref.z_indices:
        variants["single_slice"] = (single_slice(ref, z_centre), Q_SIDE)
    for name, (vin, q) in variants.items():
        if name not in args.variants.split(","):
            continue
        msgs, _ = build_messages(vin, q)
        meta = {"exp": "expC3", "subject": sid, "source": source, "gt": gt, "variant": name}
        out = m.generate(msgs, meta=meta)
        row = dict(subject=sid, source=source, gt_side=gt, variant=name, n_slices=len(vin.slices), raw=out,
                   truncated=m.truncated())
        if name.startswith("yn_"):
            row["answer"] = parse_yesno(out)[0]
            sc = m.score(msgs, [" yes", " no"], assistant_prefix="Final Answer:", meta=meta)
            row["logit_yes_no"] = sc[" yes"] - sc[" no"]
        elif name == "open":
            row["answer"] = parse_side(out)[0]
            sc = m.score(msgs, sum(OPEN_FORMS.values(), []), assistant_prefix="Final Answer:", meta=meta)
            row.update({f"lp_{k}": logsumexp_forms(sc, f) for k, f in OPEN_FORMS.items()})
        else:
            row["answer"] = parse_side(out)[0]
            sc = m.score(msgs, SIDE_CANDS, assistant_prefix="Final Answer:", meta=meta)
            row.update({f"lp_{k.strip()}": v for k, v in sc.items()})
        rows.append(row)
        print(sid, source, gt, name, row["answer"], flush=True)
save(pd.DataFrame(rows), out_dir("expC3_side_prompt") / f"{args.model}_{args.seq}_shard{args.shard}of{args.nshards}.csv")

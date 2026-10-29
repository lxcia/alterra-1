"""Positive controls: image-dependent questions MedGemma should get right through the same pipeline.

All questions are OPEN format (no option list, so no option-position bias) unless stated, and the
instruction never names the answer. Generated answers, greedy.
  P1 body region : 20 D2 pelvis T2FS vs 20 BraTS brain T2 volumes. "Which body region…?"
  P2 brain side  : 39 lateralised BraTS gliomas (side from segmentation), "which side is the
                   tumour?" in the reference option format AND open format, original and L<->R
                   mirrored. Tests whether MedGemma can report side from ANY MRI.
  P3 big object  : 20 lesion-free D2 pelvis volumes with / without a SYNTHETIC 80 mm bright sphere
                   at the uterus centroid. Asked in both polarities ("Is there…?" and "Are these
                   images free of…?") so a yes-bias cannot pass the test.
  P4 plane       : the same 20 pelvis volumes resliced to axial / coronal / sagittal stacks
                   (aspect-corrected, radiological convention). "In which plane…?"

Run: python src/experiments/expP_positive_controls.py [--model medgemma] [--parts P1,P2,P3,P4]
"""
import argparse
import re

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

from common import FINAL_YN, Q_SIDE, ROOT, d2_cases, out_dir, save
from io_orient import load_canonical, world_centroid
from models import VLM
from parse import parse_side, parse_yesno
from perturb import insert_sphere, lr_flip
from pipeline import VolumeInput, build_messages, load_ras, normalize, prepare_from_array, sample_indices
from qc_overlay import mask_on, union_mask

OPEN_TAIL = "Please provide your reasoning and conclude with 'Final Answer: ' followed by your answer."
Q_REGION = "\n\nBased on the visual evidence in the slices provided above, which body region is shown? " + OPEN_TAIL
Q_PLANE = ("\n\nBased on the visual evidence in the slices provided above, in which anatomical plane "
           "(orientation) are these slices? " + OPEN_TAIL)
Q_BRAIN_SIDE_REF = Q_SIDE.replace("largest ovarian cystic lesion", "brain tumour")
Q_BRAIN_SIDE_OPEN = ("\n\nBased on the visual evidence in the slices provided above, on which side of the patient "
                     "is the brain tumour? " + OPEN_TAIL)
Q_OBJ_YES = ("\n\nBased on the visual evidence in the slices provided above, is there a large, very bright, "
             "round object in these images? " + FINAL_YN)
Q_OBJ_NO = ("\n\nBased on the visual evidence in the slices provided above, are these images free of any "
            "large, very bright, round object? " + FINAL_YN)
assert Q_BRAIN_SIDE_REF != Q_SIDE

INSTR_NEUTRAL = ("You are an instructor teaching medical students. You are analyzing a contiguous block of "
                 "axial T2-weighted MRI slices. Please review the slices provided below carefully.")
INSTR_BRAIN = ("You are an instructor teaching medical students. You are analyzing a contiguous block of "
               "axial T2-weighted MRI slices of the brain. Please review the slices provided below carefully.")
INSTR_PELVIS_NOPLANE = ("You are an instructor teaching medical students. You are analyzing a contiguous block of "
                        "T2-weighted fat-suppressed MRI slices of the female pelvis. Please review the slices "
                        "provided below carefully.")


def final_text(out: str) -> str:
    hits = re.findall(r"final\s*answer\s*[:\-]?\s*\**\s*([^\n]*)", out or "", re.I)
    return hits[-1].strip().lower() if hits else ""


def parse_region(out):
    t = final_text(out)
    if re.search(r"pelvi|uter|ovar|abdom", t):
        return "pelvis"
    if re.search(r"brain|head|cran|cereb", t):
        return "brain"
    return "other" if t else "unparsed"


def parse_plane(out):
    t = final_text(out)
    for p, pat in [("axial", r"axial|transverse"), ("coronal", r"coronal"), ("sagittal", r"sagittal")]:
        if re.search(pat, t):
            return p
    return "other" if t else "unparsed"


def resliced_stack(img: nib.Nifti1Image, vol: np.ndarray, plane: str, subject: str) -> VolumeInput:
    """RAS+ volume -> slices in `plane`, physical aspect ratio, radiological convention, <=85 slices.
    axial: anterior up, patient right on viewer left; coronal: superior up, right on left;
    sagittal: superior up, anterior on viewer left."""
    u8 = normalize(vol)
    zx, zy, zz = img.header.get_zooms()[:3]
    if plane == "axial":
        sl, sp = [u8[::-1, ::-1, k].T for k in range(u8.shape[2])][::-1], (zy, zx)
    elif plane == "coronal":
        sl, sp = [u8[::-1, k, ::-1].T for k in range(u8.shape[1])][::-1], (zz, zx)
    else:
        sl, sp = [u8[k, ::-1, ::-1].T for k in range(u8.shape[0])], (zz, zy)
    keep = sample_indices(len(sl))
    out = []
    for i in keep:
        a = sl[i]
        h, w = a.shape
        scale = 512 / max(h * sp[0], w * sp[1])
        out.append(np.asarray(Image.fromarray(a).resize((max(1, round(w * sp[1] * scale)), max(1, round(h * sp[0] * scale))))))
    return VolumeInput(out, keep, "T2FS", subject, {"plane": plane})


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--parts", default="P1,P2,P3,P4")
ap.add_argument("--n", type=int, default=20)
args = ap.parse_args()
parts = args.parts.split(",")

m = VLM(args.model)
rows = []
pelvis = d2_cases(require=("T2FS",))
pelvis = pelvis[pelvis.uterus_present == True].head(args.n)
brain = pd.read_csv(ROOT / "derived" / "brain_inventory.csv")


def run(part, subject, cond, vin, q, instr, parser, truth):
    msgs, _ = build_messages(vin, q, instruction=instr)
    out = m.generate(msgs, meta={"exp": "expP", "part": part, "subject": subject, "cond": cond})
    rows.append(dict(part=part, subject=subject, cond=cond, truth=truth, answer=parser(out), raw=out))
    print(part, subject, cond, truth, "->", rows[-1]["answer"], flush=True)


if "P1" in parts:
    for _, c in pelvis.iterrows():
        _, vol = load_ras(c.subject, "T2FS")
        run("P1", c.subject, "pelvis", prepare_from_array(vol, c.subject, "T2FS"), Q_REGION, INSTR_NEUTRAL, parse_region, "pelvis")
    for _, b in brain.head(args.n).iterrows():
        vol = np.asarray(load_canonical(b.path).dataobj, dtype=np.float32)
        run("P1", b.subject, "brain", prepare_from_array(vol, b.subject, "T2"), Q_REGION, INSTR_NEUTRAL, parse_region, "brain")

if "P2" in parts:
    flip = {"left": "right", "right": "left"}
    for _, b in brain[brain.side.isin(["left", "right"])].iterrows():
        vol = np.asarray(load_canonical(b.path).dataobj, dtype=np.float32)
        for geom, v, truth in [("orig", vol, b.side), ("lr", lr_flip(vol), flip[b.side])]:
            vin = prepare_from_array(v, b.subject, "T2", tags={"geom": geom})
            run("P2", b.subject, f"ref_{geom}", vin, Q_BRAIN_SIDE_REF, INSTR_BRAIN, lambda o: parse_side(o)[0], truth)
            run("P2", b.subject, f"open_{geom}", vin, Q_BRAIN_SIDE_OPEN, INSTR_BRAIN, lambda o: parse_side(o)[0], truth)

if "P3" in parts:
    for _, c in pelvis[pelvis.label == "negative"].iterrows():
        img, vol = load_ras(c.subject, "T2FS")
        cen = world_centroid(img, mask_on(img, union_mask(c.subject, "ut")))
        with_obj = insert_sphere(img, vol, cen, 40.0, float(np.percentile(vol, 99.5)))
        for cond, v in [("absent", vol), ("present", with_obj)]:
            vin = prepare_from_array(v, c.subject, "T2FS", tags={"synthetic_obj": cond})
            run("P3", c.subject, f"{cond}_askpresent", vin, Q_OBJ_YES, None, lambda o: parse_yesno(o)[0],
                "yes" if cond == "present" else "no")
            run("P3", c.subject, f"{cond}_askabsent", vin, Q_OBJ_NO, None, lambda o: parse_yesno(o)[0],
                "no" if cond == "present" else "yes")

if "P4" in parts:
    for _, c in pelvis.iterrows():
        img, vol = load_ras(c.subject, "T2FS")
        for plane in ["axial", "coronal", "sagittal"]:
            run("P4", c.subject, plane, resliced_stack(img, vol, plane, c.subject), Q_PLANE, INSTR_PELVIS_NOPLANE,
                parse_plane, plane)

save(pd.DataFrame(rows), out_dir("expP_positive_controls") / f"{args.model}_{'_'.join(parts)}.csv")

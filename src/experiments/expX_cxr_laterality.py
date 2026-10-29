"""Item 1c: laterality on chest X-ray, MedGemma's trained domain, through the same code path.

Data: NIH ChestX-ray14 (public; images_001.zip + BBox_List_2017.csv from the HF mirror
alkzar90/NIH-Chest-X-ray-dataset), in derived/cxr_nih/raw/. PA views only.
  synthetic : 40 'No Finding' PA films (distinct patients) with a SYNTHETIC 30 mm-ish bright
              nodule (radius 60 px of 1024) pasted in the patient's RIGHT or LEFT mid-lung
              (image-left = patient-right on a PA film). Ground truth by construction.
  real      : every lateralised bounding-box finding in images_001.zip (cardiomegaly excluded;
              side from the box centre, |offset from image centre| > 60 px), original and L<->R
              mirrored (mirrored truth = opposite side; burned-in R/L markers mirror too).
Single image per prompt (MedGemma's CXR format). Questions: the reference option format and an
open format. Generated answers, greedy.

Run: python src/experiments/expX_cxr_laterality.py [--model medgemma]
"""
import argparse
import io
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from common import ROOT, out_dir, save
from models import VLM
from parse import parse_side
from pipeline import VolumeInput, build_messages

RAW = ROOT / "derived" / "cxr_nih" / "raw" / "data"
INSTR = ("You are an instructor teaching medical students. You are analyzing a frontal (PA) chest X-ray. "
         "Please review the image provided below carefully.")
Q_REF = ("\n\nBased on the visual evidence in the image provided above, on which side of the patient is the "
         "{thing}? Please provide your reasoning and conclude with 'Final Answer: left', 'Final Answer: right', "
         "'Final Answer: bilateral' or 'Final Answer: none'.")
Q_OPEN = ("\n\nBased on the visual evidence in the image provided above, on which side of the patient is the "
          "{thing}? Please provide your reasoning and conclude with 'Final Answer: ' followed by the side.")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="medgemma")
ap.add_argument("--n-synth", type=int, default=40)
args = ap.parse_args()

z = zipfile.ZipFile(RAW / "images" / "images_001.zip")
paths = {n.split("/")[-1]: n for n in z.namelist() if n.endswith(".png")}
entry = pd.read_csv(RAW / "Data_Entry_2017_v2020.csv")
entry = entry[entry["Image Index"].isin(paths) & (entry["View Position"] == "PA")]


def load(name: str) -> np.ndarray:
    im = Image.open(io.BytesIO(z.read(paths[name]))).convert("L").resize((1024, 1024))
    return np.asarray(im, dtype=np.float32)


def vin_of(arr: np.ndarray, name: str, tag: str) -> VolumeInput:
    return VolumeInput([np.clip(arr, 0, 255).astype(np.uint8)], [0], "CXR", name, {"cond": tag})


def add_nodule(arr: np.ndarray, patient_side: str) -> np.ndarray:
    """Paste a soft-edged bright disc in the mid-lung zone of the given patient side."""
    cx = 0.30 * 1024 if patient_side == "right" else 0.70 * 1024  # PA: image-left = patient-right
    cy = 0.45 * 1024
    yy, xx = np.mgrid[:1024, :1024]
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2 <= 60 ** 2).astype(np.float32)
    soft = np.clip(ndimage.gaussian_filter(disc, 4), 0, 1)
    return arr * (1 - soft) + float(np.percentile(arr, 97)) * soft


m = VLM(args.model)
rows = []


def ask(name, source, cond, arr, truth, thing):
    for fmt, q in [("ref", Q_REF), ("open", Q_OPEN)]:
        msgs, _ = build_messages(vin_of(arr, name, cond), q.format(thing=thing), instruction=INSTR, slice_label="")
        out = m.generate(msgs, meta={"exp": "expX", "image": name, "cond": cond, "fmt": fmt})
        rows.append(dict(image=name, source=source, cond=cond, fmt=fmt, truth=truth, answer=parse_side(out)[0], raw=out))
        print(name, source, cond, fmt, truth, "->", rows[-1]["answer"], flush=True)


normals = entry[entry["Finding Labels"] == "No Finding"].drop_duplicates("Patient ID").head(args.n_synth)
for name in normals["Image Index"]:
    base = load(name)
    for side in ["right", "left"]:
        ask(name, "synthetic", f"nodule_{side}", add_nodule(base, side), side, "nodule")

bb = pd.read_csv(RAW / "BBox_List_2017.csv")
bb = bb[bb["Image Index"].isin(entry["Image Index"]) & (bb["Finding Label"] != "Cardiomegaly")]
for _, r in bb.iterrows():
    x0, w = r["Bbox [x"], r["w"]  # NIH boxes are in 1024x1024 coordinates
    cx = x0 + w / 2
    if abs(cx - 512) <= 60:
        continue
    side = "right" if cx < 512 else "left"  # image-left = patient-right on PA
    arr = load(r["Image Index"])
    thing = r["Finding Label"].lower()
    ask(r["Image Index"], "real", "orig", arr, side, thing)
    ask(r["Image Index"], "real", "lr", arr[:, ::-1].copy(), "left" if side == "right" else "right", thing)

save(pd.DataFrame(rows), out_dir("expX_cxr_laterality") / f"{args.model}.csv")

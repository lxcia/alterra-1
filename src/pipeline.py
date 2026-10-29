"""Reference MedGemma 1.5 3D input pipeline for pelvic MRI.

Two official sources (there is no MRI notebook; the MRI rules are in the technical report):
  [N] google-health/medgemma, notebooks/high_dimensional_ct_hugging_face.ipynb (CT only)
      https://github.com/google-health/medgemma/blob/main/notebooks/high_dimensional_ct_hugging_face.ipynb
      (commit ea92d0f17e7ed56c16773d6a23d0a27d7e90b274, 2026-01-15)
  [R] MedGemma 1.5 Technical Report, arXiv:2604.05081 (v2), section 2.3.1 "Preprocessing of
      Volumetric Images": the preprocessing used for MRI in training and evaluation.

From the notebook [N]:
  * at most MAX_SLICE = 85 axial slices, sampled with the notebook's formula
    idx = round(i / 85 * (n - 1)) for i = 1..85 (only when n > 85; otherwise all slices).
    [R] also caps at 85 and samples "equidistantly across the z-axis";
  * each slice -> uint8 RGB -> JPEG-encoded and decoded (the notebook sends base64 JPEG);
  * message = [instruction text] + for each slice [image, "SLICE {n}"] + [query text];
  * greedy decoding (do_sample=False), max_new_tokens=2000 in the notebook.

From the report [R], for MRI (replaces the notebook's CT-only intensity step):
  * "since voxel values of MR images are all relative and no physiological windows exist, no
    windowing was applied ... we normalized them per volume using min-max normalization, and set
    the R, G, and B channels to the same value." -> normalize() maps the volume minimum to 0 and
    the maximum to 255 (P_LO, P_HI = 0, 100 percentiles), grayscale copied into R, G and B.
    (The notebook's three CT windows (-1024,1024), (-135,215), (0,80) in R, G, B are for
    Hounsfield units only.)
    CHANGED 2026-10-26: the original analysis used a 0.5-99.5 percentile window instead
    (P_LO, P_HI = 0.5, 99.5). Results produced before this date used that window; expE's
    "win_0_100" condition (= min-max) changed 2/39 endometrioma answers relative to it.
  * Volume inclusion criteria in [R] (training data): <= 512 x 512 pixels per slice, axial
    orientation, constant slice thickness, at least 5 slices. All UT-EndoMRI series used here are
    axial NIfTI volumes (constant spacing by construction); see DATA_NOTES for tilted D1 series.
  * Each slice is rescaled to 896 x 896 by the model's image processor (not done here).

Our own choices (neither source specifies them):
  * Slice order: most superior slice first (typical axial DICOM InstanceNumber order).
  * In-plane orientation: radiological convention (see io_orient.py).
  * Instruction text: the notebook's, with "CT slices from the center of the abdomen" replaced
    by "axial {sequence} MRI slices of the female pelvis". The report's own MRI evaluation prompt
    ("MRI-US1") is tested separately in experiments/expM_mri_us1.py.

Data flow (from file to model input):
  load_ras(subject, seq)            NIfTI -> RAS+ float array (x, y, z)
  [optional perturbation]           perturb.py works on this RAS+ array (e.g. mirror, insert a sphere)
  prepare_from_array(...)           min-max to uint8 -> radiological axial stack -> superior first
                                    -> sample <=85 slices -> VolumeInput
  build_messages(vin, query)        [instruction, image, "SLICE 1", image, "SLICE 2", ..., query]
                                    (images JPEG round-tripped here, via VolumeInput.images())
  models.VLM.generate / .score      tokenisation and the model call
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from io_orient import load_canonical, subject_dir, to_radiological_stack

MAX_SLICE = 85
P_LO, P_HI = 0.0, 100.0   # min-max, as in the MedGemma 1.5 report (was 0.5, 99.5 before 2026-10-26)

# Instruction text before the slices. "{seq_desc}" is filled with the sequence description below
# (e.g. "T2-weighted fat-suppressed"), so the prompt names the sequence the pixels come from.
INSTRUCTION = ("You are an instructor teaching medical students. You are analyzing a "
               "contiguous block of axial {seq_desc} MRI slices of the female pelvis. "
               "Please review the slices provided below carefully.")

SEQ_DESC = {"T1": "T1-weighted", "T1FS": "T1-weighted fat-suppressed",
            "T2": "T2-weighted", "T2FS": "T2-weighted fat-suppressed"}


def sample_indices(n: int, max_slice: int = MAX_SLICE) -> list[int]:
    """Notebook's uniform sampling formula (note: starts at i=1, as in the notebook).

    Returns positions (0..n-1) of the slices to keep. Volumes with <=85 slices keep all of them:
    every T2 / T2FS series here (30-76 slices). Many T1FS series have more (up to 170) and are
    subsampled. Because i starts at 1, position 0 is skipped when subsampling, exactly as in the
    notebook."""
    if n <= max_slice:
        return list(range(n))
    return [int(round(i / max_slice * (n - 1))) for i in range(1, max_slice + 1)]


def normalize(vol: np.ndarray, p_lo: float = P_LO, p_hi: float = P_HI) -> np.ndarray:
    """Per-volume intensity normalisation to uint8 (the report's MRI rule; see module docstring).

    With the defaults (0th and 100th percentile) this is min-max normalisation: the volume minimum
    becomes 0 and the maximum 255, nothing is clipped. Other p_lo / p_hi give a percentile window
    (values outside are clipped), used only by robustness experiments (e.g. expE). Computed once
    per volume (not per slice), so brightness is comparable across slices.
    NaNs are replaced by the volume minimum first; a constant volume maps to all zeros."""
    vol = np.nan_to_num(vol.astype(np.float32), nan=float(np.nanmin(vol)) if np.isfinite(vol).any() else 0.0)
    lo, hi = np.percentile(vol, [p_lo, p_hi])
    v = np.clip(vol, lo, hi)
    v = (v - lo) / max(hi - lo, 1e-6) * 255.0
    return np.round(v).astype(np.uint8)


def jpeg_roundtrip(gray: np.ndarray) -> Image.Image:
    """uint8 2D -> RGB -> JPEG bytes -> PIL image (the notebook sends JPEG data URIs).

    The grayscale slice is copied into all 3 channels, encoded as JPEG at PIL's default quality
    and decoded again, so the model sees the same compression artefacts as in the notebook."""
    rgb = np.stack([gray] * 3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="jpeg")
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    return img


@dataclass
class VolumeInput:
    """A prepared stack: radiological uint8 slices, superior-first, plus provenance."""
    slices: list[np.ndarray]           # uint8 2D arrays actually shown
    z_indices: list[int]               # RAS+ z index of each shown slice (for lesion bookkeeping)
    seq: str
    subject: str
    tags: dict = field(default_factory=dict)  # free-form notes on the condition (e.g. geometry)

    def images(self) -> list[Image.Image]:
        """The slices as the PIL images the model receives (after the JPEG round-trip)."""
        return [jpeg_roundtrip(s) for s in self.slices]

    def digest(self) -> str:
        """Short hash of the shown slices (to check whether two conditions produce the same input)."""
        h = hashlib.sha1()
        for s in self.slices:
            h.update(np.ascontiguousarray(s).tobytes())
        return h.hexdigest()[:16]


def load_ras(subject: str, seq: str):
    """Load one series of one subject as (RAS+ image, float32 array (x, y, z)).

    The file name is <subject>_<seq>.nii.gz; the sequence name is trusted as given (see
    inventory.py for the identical-file check that caught a mislabelled series)."""
    img = load_canonical(subject_dir(subject) / f"{subject}_{seq}.nii.gz")
    return img, np.asarray(img.dataobj, dtype=np.float32)


def prepare_from_array(vol_ras: np.ndarray, subject: str, seq: str, max_slice: int = MAX_SLICE,
                       p_lo: float = P_LO, p_hi: float = P_HI, tags: dict | None = None) -> VolumeInput:
    """Reference preprocessing of a RAS+ float volume (x, y, z).

    Taking an array (not a file) lets experiments perturb the volume first and then send it through
    exactly the same preprocessing as the original."""
    u8 = normalize(vol_ras, p_lo, p_hi)
    stack = to_radiological_stack(u8)           # (z, rows, cols), z=0 inferior
    order = list(range(stack.shape[0]))[::-1]   # superior first
    # pick <=85 slices along the superior-first order; keep their RAS+ z indices for bookkeeping
    keep = [order[i] for i in sample_indices(len(order), max_slice)]
    return VolumeInput([stack[z] for z in keep], keep, seq, subject, dict(tags or {}))


def prepare(subject: str, seq: str, **kw) -> VolumeInput:
    """load_ras + prepare_from_array for the unperturbed volume."""
    _, vol = load_ras(subject, seq)
    return prepare_from_array(vol, subject, seq, **kw)


def build_messages(vin: VolumeInput | None, query: str, instruction: str | None = None,
                   seq_desc: str | None = None, slice_label: str = "SLICE {n}") -> tuple[list, list]:
    """Notebook message layout. Returns (messages, images); images are referenced in order.

    vin=None gives a text-only prompt (no image tokens) for text-only ablations.
    `instruction` replaces the default instruction entirely (used to add a clinical history);
    `seq_desc` only changes the sequence name in it (used to test the effect of the name);
    `slice_label=""` drops the "SLICE n" text items (used for single 2D images, e.g. chest X-rays)."""
    if instruction is None:
        desc = seq_desc if seq_desc is not None else SEQ_DESC.get(vin.seq, vin.seq) if vin else "pelvic"
        # an empty description used to leave a double space ("axial  MRI"); collapse it (P19)
        instruction = INSTRUCTION.format(seq_desc=desc).replace("  ", " ")
    content = [{"type": "text", "text": instruction}]
    images = vin.images() if vin is not None else []
    for n, im in enumerate(images, 1):
        content.append({"type": "image", "image": im})
        if slice_label:
            content.append({"type": "text", "text": slice_label.format(n=n)})
    content.append({"type": "text", "text": query})
    return [{"role": "user", "content": content}], images

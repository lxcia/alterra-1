# MedGemma 1.5 on endometriosis MRI: two failure modes

Google advertises MedGemma 1.5 (`google/medgemma-1.5-4b-it`) as able to interpret 3D CT and MRI
volumes. This project tests it on a clinically routine task: reading axial pelvic MRI (UT-EndoMRI,
D2 cohort, T2-weighted fat-suppressed) for an ovarian endometrioma and its side. The
input pipeline follows Google's own recipe (CT example notebook + the MRI preprocessing in the
MedGemma 1.5 technical report). Two failures:

1. **Failure 1:** When asked which side an ovarian lesion is on, MedGemma 1.5 4B picks whichever side is listed first in the answer options, and the image has no effect on the answer.
2. **Failure 2:** If the prompt says the patient has a “known ovarian endometrioma”, MedGemma 1.5 4B reports one on scans that have none, and an irrelevant history does not have this effect.

## Where to look

**Results**
- `results/<experiment>/`: raw per-read CSVs (prompt condition, generated text, parsed answer,
  log-probabilities) with a `.provenance.json` (command, code hash, prompt format, time).

**Failure 1: default side** (`src/experiments/`)
- `expC3_side_prompt.py`: main experiment. Side question with the reference option order, reversed
  order, open question, yes/no per side, single lesion slice; 22 real + 40 synthetic lesions.
- `expC_laterality.py`: the same volume mirrored left–right (does the answer follow the lesion?).
- `expC2_synthetic_side.py`: synthetic bright cyst inserted on the left, right or both sides.
- `expC4_psychometric.py`: synthetic sphere sweep over size, contrast and distance from midline.
- `expA3_type_order.py`: the same option-position rule on the lesion-type question.
- `src/lesion_regime.py`: where the real lesions fall on the sweep's size/brightness axes.

**Failure 2: clinical history** (`src/experiments/`)
- `expB4_history_controls.py`: main experiment. No history vs knee pain vs pelvic pain vs known
  endometrioma, on the 58 endometrioma-negative volumes.
- `expB_history.py`: history × image ablation (real, blank, noise, text only); left/right in the history.
- `expB2_history_dose.py`: strength of the history (possible → known → large bilateral; resected).
- `expB3_history_x_paraphrase.py`: history effect under five question wordings.
- `expR_freetext_report.py` (+ `src/parse_report.py`): free-text findings reports with and without history.
- `expM_mri_us1.py` (+ `src/analyze_expM.py`): Google's own MRI evaluation prompt (MRI-US1).
- `expF_pushback.py`: the same pull in dialogue (user disagrees with the answer).

**Data processing**
- `src/io_orient.py`: loads NIfTI volumes in RAS+ orientation, radiological display, lesion side
  rules, excluded subjects (D2-007, D2-009).
- `src/inventory.py`: builds `derived/inventory.csv` (one row per subject: series, labels, lesion
  presence, side, size, which grid each mask lies on).
- `src/pipeline.py`: the MedGemma input pipeline (min-max normalisation, grayscale RGB,
  superior-first slices, ≤85 slices, JPEG round-trip, `[instruction, image, "SLICE n", …, question]`).
- `src/experiments/common.py`: cohort selection (`d2_cases`), shared question texts, output helpers.
- `src/perturb.py`: mirroring, synthetic spheres, blank/noise volumes, slice-level edits.
- `src/qc_overlay.py`, `src/qc_synthetic.py`, `src/qc_synthetic_cyst.py`: QC images of mask
  alignment and synthetic insertions (`derived/qc/`).
- `dataset_overview.py`, `src/data_explore.py`: cohort description and figures (`figures/dataset_overview/`).

## Setup

```bash
uv venv -p 3.11 .venv
UV_LINK_MODE=symlink uv pip install -p .venv/bin/python -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
```

Models (Hugging Face; the Google models are gated, accept their terms first). Revisions are pinned
in `src/models.py` (`REVISIONS`):

| Key | Model | Role |
|---|---|---|
| `medgemma` | `google/medgemma-1.5-4b-it` | main model |
| `gemma3` | `google/gemma-3-4b-it` | same architecture, no medical tuning |
| `medgemma27` | `google/medgemma-27b-it` | MedGemma 1 (no 3D training), larger; ~55 GB GPU memory |
| `qwen` | `Qwen/Qwen2.5-VL-7B-Instruct` | different model family |

Put a Hugging Face read token in `~/.cache/huggingface/token` (or `HF_TOKEN`).

**Environment variables**
- `VLM_PROMPT_FORMAT`: prompt format, default `v2` (one `<bos>` token, exactly as Google's
  notebook; 2000 new tokens). All results in `results/` were produced with v2. `v1` reproduces an
  earlier run that had a duplicated `<bos>`.
- `UTENDO_DATA_ROOT`: path to the UT-EndoMRI `data/` folder.
- `VLM_HF_CACHE_QWEN`: Hugging Face hub cache for Qwen, if it is stored elsewhere.
- `VLM_CACHE_ONLY=1`: fail on any uncached model call instead of loading a model.

## Data

- **UT-EndoMRI** (Liang et al., *Sci Data* 2025): download from the source in the paper's Data
  Records section, unpack so that `data/D2_TCPW/` and `data/SiteScannerInfo.csv` exist, and set
  `UTENDO_DATA_ROOT`. Raw data is never modified. Two D2 subjects (D2-007, D2-009) are excluded:
  identical images with contradictory labels.
- **Positive-control data** (only for `ct_repro.py`, `expP` P2 and `expX`), under `derived/`:
  - Google's CT example series (via `idc-index`) and notebook in `derived/ct_repro/`;
  - BraTS 2024 GLI training subset (`*-t2w.nii.gz`, `*-seg.nii.gz`) in `derived/brain_mri_train/`;
  - NIH ChestX-ray14 `images_001.zip`, `Data_Entry_2017.csv`, `BBox_List_2017.csv` in `derived/cxr_nih/raw/data/`.

`derived/` and `data/` are not in git.

## Reproduce

Build the inventory first (CPU):

```bash
.venv/bin/python src/inventory.py
.venv/bin/python src/brain_inventory.py      # only for expP P2
```

To reproduce everything in `results/`:

```bash
bash run_all.sh                 # computes any uncached model call on GPU $CUDA_VISIBLE_DEVICES
CACHE_ONLY=1 bash run_all.sh    # tables only, from results/cache/; CPU, fails on any cache miss
```

Each experiment is also a standalone script that writes `results/<experiment>/<model>_*.csv`, e.g.:

```bash
.venv/bin/python src/experiments/expC3_side_prompt.py                                   # Failure 1, MedGemma 4B
.venv/bin/python src/experiments/expC3_side_prompt.py --model gemma3 --variants ref,order_rl,open
.venv/bin/python src/experiments/expB4_history_controls.py --parts irrelevant,known     # Failure 2
```

Long experiments take `--shard i --nshards n` to split across GPUs. `scripts/overnight/` has the
same commands split into per-GPU queue files, and `run_lane.sh <name> <gpu> <queue-file>
[extras-file]` runs one queue in order, retrying a failed job once (logs in `results/logs/`, not in git).

**Runtime.** Prompts are 10k–22k tokens (one image per slice). One generated answer takes ~25–40 s
for the 4B models and ~55 s for the 27B on an H100; the full set of experiments took one night on
four GPUs.

**Cache.** Every model call is stored in `results/cache/*.jsonl`, keyed by model revision, prompt
format, decoding settings, prompt text and image hashes. Re-running a command reuses cached calls,
so an interrupted job resumes where it stopped. The cache is not in git, so `CACHE_ONLY=1` works
only where the cache exists.

**Tables.**

```bash
.venv/bin/python src/analyze.py <experiment>                 # e.g. expC3, expB4; --model gemma3|medgemma27|qwen
.venv/bin/python src/analyze_expM.py
```

`run_all.sh` writes these tables to `results/summary_<model>.txt` and `results/summary_expM.txt`.

## Methods in brief

- **Cohort:** D2 T2FS, 68 patients: 10 endometrioma, 14 simple cyst only, 44 without either.
  Expert masks lie on the T2FS grid.
- **Lesion side:** centroid of each lesion vs the uterus centroid (field-of-view centre if no
  uterus); within 5 mm = midline (excluded from side analyses). 22 real lesions have a clear side.
- **Synthetic lesions:** a bright sphere (18 mm radius, 99.5th-percentile intensity) inserted
  45 mm left or right of the uterus centroid in 20 lesion-free volumes. Procedurally drawn, not
  from a generative model.
- **Readouts:** (1) generated answer, greedy decoding, parsed after the last "Final Answer:"
  (`src/parse.py`); (2) score = log P(" yes") − log P(" no") (or per side) for the next token
  after a pre-filled "Final Answer:" (`VLM.score` in `src/models.py`).

## Other scripts

- `src/ct_repro.py`: reproduces Google's CT notebook output byte for byte (pipeline check).
- `src/experiments/expP_positive_controls.py`, `expX_cxr_laterality.py`: positive controls (body
  region, plane, synthetic object, brain tumour side, chest X-ray side).
- `src/experiments/exp00_baseline.py`: baseline presence and side answers (context).
- `src/experiments/expA_sequence.py`, `expA2_contrast.py`, `expD_slices.py`,
  `expD2_shuffle_confirm.py`, `expE_invariance.py`, `expN_noise.py`: appendix experiments and
  noise floors (sequence name, contrast edits, slice selection/order, meaningless perturbations,
  paraphrase noise).
- `src/parser_audit.py`, `src/dump_prompts.py`, `src/figures.py`: parser hand-check sample, every prompt string, figures.



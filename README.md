# ResWFM: Zone-Aware Residual Wavelet Flow Matching for Contrast-Enhanced Prostate MRI Synthesis

> **Note (anonymous review):** author names and institutional references are
> withheld below pending the outcome of double-blind review. `LICENSE` uses
> a placeholder copyright holder for the same reason. Replace both once the
> manuscript's authorship is no longer under review — see the `TODO` markers
> in `LICENSE`.

Code accompanying a submission to IEEE Transactions on Medical Imaging.
ResWFM synthesizes contrast-enhanced T1 (CE-T1) prostate MRI from
non-contrast sequences, formulated as a **residual** flow-matching problem:

```
CE_T1 = NC_T1 + Δ,     Δ = CE_T1 − NC_T1 learned via wavelet flow matching
```

## Method summary

- **NC-anchored residual target.** Rather than generating CE-T1 directly,
  the model predicts only the enhancement map Δ, with the flow source fixed
  at zero (`guided_diffusion/wfm_prostate_run07.py`). Anatomy comes for free
  from NC_T1; all model capacity goes to the gadolinium-uptake signal.
- **Zone-FiLM conditioning.** Each anatomical zone (background / transition
  zone / peripheral zone) gets its own learned per-channel FiLM
  scale-and-shift, applied at every encoder downsample
  (`guided_diffusion/zone_film.py`), so the network can weight feature
  channels differently inside PZ vs. TZ vs. background — mirroring how a
  radiologist leans on different sequences per zone.
- **Frequency-graded subband loss.** The wavelet-domain loss boosts
  high-frequency detail subbands proportionally to their number of
  high-pass axes inside the prostate
  (`guided_diffusion/wfm_prostate_run10.py`), countering the L2 bias toward
  blurry reconstructions at exactly the bands that carry fine detail.
- **Zone-weighted training objective.** TZ and PZ voxels are up-weighted in
  the loss (PZ highest, since ~70% of clinically significant cancers
  originate there); see `scripts/train_wfm_prostate_mask.py`.

Full mathematical formulation and quantitative results are in the
manuscript; this repo is the reference implementation.

## Model configuration

This repository packages exactly one configuration: **ResWFM (Run10,
zone-FiLM) at 160×160×32**, the 5-level UNet backbone
(`channel_mult=1,2,2,4,4`, ~50.6M parameters) baked into `run_reswfm.sh`'s
`COMMON_ARGS` — the configuration used for the manuscript's reported
results. No other architecture depth, backbone, or component-ablation
variant is included here.

## Relationship to WFM

ResWFM's method-level contributions (NC-residual reframing, zone-FiLM,
frequency-graded subband loss, zone-weighted training/eval) are original to
this work — they do not exist in the public WFM codebase
(https://github.com/yalcintur/WFM), which targets unconditional BraTS
multi-modality synthesis via a mean-of-modalities prior, not a residual
formulation. What *is* shared is generic infrastructure (the 3D UNet
backbone, Haar DWT/IDWT wiring, logging/dist utilities, base training-loop
skeleton) — vendored into `guided_diffusion/`/`DWT_IDWT/` here under their
original MIT license (see `LICENSE`, `THIRD_PARTY_NOTICES.md`). **This repo
has no runtime dependency on the WFM repository** — everything needed to
train, sample, and evaluate ResWFM is self-contained here.

## What's included

- Model/loss code: `guided_diffusion/` (architecture, zone-FiLM, residual
  wavelet flow-matching losses, data loader)
- Training loop: `scripts/train_wfm_prostate_mask.py`,
  `scripts/train_wfm_prostate_run07.py`, `scripts/train_wfm_prostate_run10.py`
- **Inference**: `scripts/sample_wfm_prostate_run10.py` — see
  [`inference/README.md`](inference/README.md) for a worked example
- Evaluation (mask-weighted PSNR/SSIM/MSE/MAE):
  `scripts/evaluate_prostate.py`, `scripts/evaluate_prostate_run10.py`
- A genericized run script, `run_reswfm.sh` — no absolute or
  machine-specific paths; everything is a CLI arg / environment variable
  with a relative default

## What's NOT included (yet)

- **Data.** No imaging data is distributed here. The dataset is not yet
  publicly released.
- **Preprocessing.** The resampling/cropping/normalization pipeline that
  turns raw DICOM/NIfTI into the model's expected input format is not
  included in this pass. `inference/README.md` documents the *expected
  output* format of that pipeline so the model's input contract is still
  fully specified.
- **Split manifests.** Train/val/test subject-ID lists are withheld along
  with the data they index.
- **Trained checkpoints.** See `CHECKPOINTS.md`.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+. GPU strongly recommended for training;
inference (1-2 integration steps) is fast enough to run on CPU for a single
case.

## Usage

```bash
# Set these to your own data (no defaults point anywhere outside this repo)
export RESWFM_DATA_DIR=/path/to/your/dataset
export RESWFM_SPLIT_DIR=/path/to/your/split/manifests

bash run_reswfm.sh sanitycheck   # 10 iterations, smoke test
bash run_reswfm.sh train         # full training run
bash run_reswfm.sh resume        # resume from the latest checkpoint
bash run_reswfm.sh sample_test   # inference on the test manifest
bash run_reswfm.sh eval_test     # PSNR/SSIM/MSE/MAE within the prostate mask
```

### Inference on a single case

See [`inference/README.md`](inference/README.md) — it documents the exact
expected input layout for one subject and the `sample_wfm_prostate_run10.py`
invocation, independent of having the full dataset or `run_reswfm.sh`.

## Data format (expected input contract)

```
<subject_dir>/T2.nii.gz
<subject_dir>/ADC.nii.gz
<subject_dir>/DWI.nii.gz
<subject_dir>/NC_T1.nii.gz
<subject_dir>/CE_T1.nii.gz      # training/eval only
<subject_dir>/mask.nii.gz       # 0=background, 1=TZ, 2=PZ
```

NC_T1 and CE_T1 must be normalized with **NC_T1's own** foreground intensity
window applied to both (never a window derived from CE_T1, which is
unavailable at inference time) — this is required by the residual
formulation, not a preprocessing convenience.

## Citation

```bibtex
@article{anonymous2026reswfm,
    title   = {Zone-Aware Residual Wavelet Flow Matching for Contrast-Enhanced Prostate MRI Synthesis},
    author  = {Anonymous Authors},
    journal = {IEEE Transactions on Medical Imaging},
    year    = {2026},
    note    = {Under review}
}
```

## License

This repository is **dual-licensed**:

- Everything except `DWT_IDWT/` is **MIT** (see `LICENSE`).
- `DWT_IDWT/` (the Haar DWT/IDWT implementation) is **CC BY-NC-SA 4.0**
  (Adobe Inc.), **non-commercial only** — see `DWT_IDWT/LICENSE`.

See `THIRD_PARTY_NOTICES.md` for the full explanation and attribution
chain.

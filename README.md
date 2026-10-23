# ResWFM: Zone-Aware Residual Wavelet Flow Matching for Contrast-Enhanced Prostate MRI Synthesis

Building on Wavelet Flow Matching (WFM), we introduce **ResWFM**, a framework for synthesizing late-phase contrast-enhanced T1-weighted (CE-T1W) prostate MRI from multimodal non-contrast MRI. This task is challenging because enhancement depends on contrast-agent kinetics and tissue-specific vascular properties that cannot be directly observed in non-contrast images. The model must infer these signals from indirect anatomical and functional information, including subtle, spatially localized enhancement patterns. ResWFM addresses this challenge by using non-contrast T1-weighted (NC-T1W) MRI as an anatomical anchor and learning the enhancement residual in a 3D Haar wavelet domain. Zone-aware feature modulation and a frequency-graded, zone-weighted training objective further tailor the framework to prostate anatomy. On the held-out test set, ResWFM achieved a whole-volume PSNR of 21.42 dB and SSIM of 0.789, with a whole-gland SSIM of 0.761 and peripheral-zone SSIM of 0.729. Inference takes 0.12 seconds per subject using a two-step Euler solver.

## Architecture

<p align="center">
  <img src="docs/figures/reswfm_architecture.png" alt="ResWFM architecture showing wavelet-domain conditioning, residual flow prediction, Euler integration, inverse wavelet reconstruction, and addition to the NC-T1W anchor." width="100%">
</p>

ResWFM transforms four conditioning modalities (T2W, NC-T1W, ADC, and DWI) and the whole-gland signed distance function (SDF) into the 3D Haar wavelet domain. These features are concatenated with the flow state x_t to form a 48-channel input to a 3D U-Net velocity network with approximately 81.5 million parameters. The flow starts from a zero source (x_0 = 0), rather than random noise or an averaged image prior as we did in WFM. Two Euler steps integrate the predicted velocity to obtain the wavelet-domain endpoint x̂_1. The inverse DWT reconstructs the predicted enhancement residual Δ̂, which is added to the NC-T1W anchor to produce the synthesized CE-T1W volume. Zone-FiLM conditions encoder features on one-hot background (BG), transition-zone (TZ), and peripheral-zone (PZ) labels at each of the four downsampling stages. This modulation is applied alongside the frequency-graded, zone-weighted training objective.

## Qualitative results

<p align="center">
  <img src="docs/figures/reswfm_visual_results.png" alt="Qualitative comparison across three test cases. Columns left to right: T2W with lesion contour and arrow, NC-T1W, reference CE-T1W, ResWFM, WFM, AAD-DCE, DCE-Diff, Pix2Pix." width="100%">
</p>

Three representative lesion-containing test cases (rows). From left to right: T2W (green lesion contour and arrow), NC-T1W, reference CE-T1W, and synthesized CE-T1W images from ResWFM, WFM, AAD-DCE, DCE-Diff, and Pix2Pix. The panels illustrate gland-wide enhancement patterns and signal appearance near the marked lesions relative to the reference. 

### Inference efficiency

Inference time was measured on an NVIDIA L4 GPU and is reported as the mean per-subject time across the test set.

| Model | Inference configuration | Parameters (M) | Avg. time (s/subject) |
|---|---|---:|---:|
| Pix2Pix | 2D, single-pass | 61.40 | 0.72 |
| AAD-DCE | 2D, single-pass | 11.38 | 0.35 |
| DCE-Diff | 100-step DDPM | 73.93 | 85.2 |
| T1C-RFlow | 50-step RFlow | 44.26† | 1.39 |
| WFM | 2-step Euler | 81.52 | 0.11 |
| **ResWFM (proposed)** | **2-step Euler** | **81.54** | **0.12** |

† T1C-RFlow additionally uses a frozen, pretrained MAISI variational
autoencoder (20.94M parameters) for latent encoding/decoding, not included
in the trainable parameter count shown.

### Reconstruction and enhancement fidelity

Results on the held-out test set (57 subjects; lesion-level analysis in 34 subjects with annotated index lesions):

**Whole-volume fidelity:** WFM achieved the highest whole-volume PSNR and SSIM (22.67 dB and 0.812). ResWFM achieved 21.42 dB and 0.789, respectively.

**Zonal SSIM:** ResWFM achieved the highest SSIM among the evaluated methods in the whole gland (0.761), transition zone (0.751), and peripheral zone (0.729). These results highlight a trade-off between whole-volume and zonal image fidelity rather than uniform superiority of either WFM or ResWFM.

**Whole-gland enhancement MAE:**ResWFM achieved a lower MAE than WFM (0.0895 vs. 0.0985), with statistically significant improvements over the synthesis comparators after Holm correction (adjusted p ≤ 0.001).

## Installation

```bash
git clone <this-repository-url>
cd ResWFM-release
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+. A GPU is strongly recommended for
training; inference (1–2 integration steps) is fast enough to run on CPU for
a single case.

## Inference and Training

ResWFM supports model training and inference on a single subject folder or a directory of subjects using a trained model checkpoint.

## Requirements 

Before running the pipeline, ensure that you have:
- A trained checkpoint for inference, or a dataset prepared for training.
- Input volumes that conform to the data format specification.

Pretrained weights may be requested from the corresponding author for non-commercial research purposes. See Data and Model Access.

Set the paths to your dataset and split manifests:
```
export RESWFM_DATA_DIR=/path/to/your/dataset
export RESWFM_SPLIT_DIR=/path/to/your/split/manifests
```
Run commands from the repository root.
For inference configuration, checkpoint selection, input preparation, and single-subject or batch workflows, see inference/README.md.
Each subject must be provided as a directory containing the following NIfTI volumes:
```
<subject_dir>/
├── T2.nii.gz
├── ADC.nii.gz
├── DWI.nii.gz
├── NC_T1.nii.gz
├── mask.nii.gz
└── CE_T1.nii.gz    # Required for training
```

Input requirements
- T2.nii.gz, ADC.nii.gz, DWI.nii.gz, and NC_T1.nii.gz provide the model's imaging inputs.
- mask.nii.gz contains the prostate zonal labels: 0 = background, 1 = transition zone (TZ), and 2 = peripheral zone (PZ).
- CE_T1.nii.gz provides the target during training. At inference, it is not passed to the model; if supplied, it can be used for reference and comparison.
- Intensity normalization: Normalize NC_T1 and CE_T1 using the same foreground intensity window derived from NC_T1. Do not derive the normalization window from CE_T1, which is unavailable at inference. This consistency is essential to the residual formulation.
Inputs must also satisfy the spatial and dimensional requirements of the inference pipeline. Refer to inference/README.mdfor the complete input contract.

The repository does not include the preprocessing pipeline that converts raw imaging data into this format.
ResWFM was developed using an in-house prostate MRI dataset. Researchers wishing to obtain pretrained model weights for non-commercial research purposes may contact the corresponding author by email.
The repository provides the model implementation and associated workflows. Training on an independent dataset requires appropriately prepared imaging volumes, dataset splits, and the configuration described above.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### 1. Obtain a pretrained checkpoint

Pretrained model weights may be available for non-commercial research purposes upon email request to the corresponding author.

Alternatively, train a model using your own dataset and the training instructions provided in this repository.

### 2. Prepare input volumes

For each subject, provide the following NIfTI volumes:

- `T2.nii.gz`
- `ADC.nii.gz`
- `DWI.nii.gz`
- `NC_T1.nii.gz`
- `mask.nii.gz`

The optional `CE_T1.nii.gz` volume may also be provided for reference and comparison with the generated output. It is not used as a model input.

Input volumes must follow the preprocessing and spatial-format requirements described in [Data Format](#data-format-expected-input-contract). Dataset-specific preprocessing is not provided in this repository.

### 3. Run inference

For detailed instructions, including single-subject and batch inference, see [`inference/README.md`](inference/README.md).

To run inference using the test-manifest workflow:

```bash
bash run_reswfm.sh sample_test
```

Ensure that the required model checkpoint and input volumes are available at the paths specified in the inference configuration before running the pipeline.

For the full training/resume/sample/eval command set:

```bash
export RESWFM_DATA_DIR=/path/to/your/dataset
export RESWFM_SPLIT_DIR=/path/to/your/split/manifests

bash run_reswfm.sh sanitycheck   # 10 iterations, smoke test
bash run_reswfm.sh train         # full training run
bash run_reswfm.sh resume        # resume from the latest checkpoint
bash run_reswfm.sh sample_test   # inference on the test manifest
bash run_reswfm.sh eval_test     # PSNR/SSIM/MSE/MAE within the prostate mask
```

## Data format (expected input contract)

```
<subject_dir>/T2.nii.gz
<subject_dir>/ADC.nii.gz
<subject_dir>/DWI.nii.gz
<subject_dir>/NC_T1.nii.gz
<subject_dir>/CE_T1.nii.gz      # required for training; optional at inference (never fed to the model)
<subject_dir>/mask.nii.gz       # 0=background, 1=TZ, 2=PZ
```

NC_T1 and CE_T1 must be normalized with **NC_T1's own** foreground intensity
window applied to both (never a window derived from CE_T1, which is
unavailable at inference time) — this is required by the residual
formulation, not a preprocessing convenience.

## Data and Pretrained Weights

- **Data:** The model was developed using an in-house prostate MRI dataset. Prepare your own dataset according to the input requirements described in [`inference/README.md`](inference/README.md).
- **Preprocessing:** The preprocessing pipeline for converting raw imaging data into the model's expected input format is not included. Refer to [`inference/README.md`](inference/README.md) for the required input specifications.
- **Pretrained weights:** Pretrained model weights may be requested from the corresponding author for non-commercial research purposes. Please contact the corresponding author by email for further information.
- **Training:** To train the model on your own dataset, follow the training instructions provided in this repository.

The repository provides the implementation and inference workflow. Running inference requires a trained checkpoint and input volumes that conform to the specified data format.

## Repository structure

```
ResWFM-release/
├── docs/figures/              Architecture + qualitative-results figures
├── guided_diffusion/          Model, losses, Zone-FiLM, data loader
├── DWT_IDWT/                  3D Haar DWT/IDWT (CC BY-NC-SA 4.0, see below)
├── scripts/                   Train / sample (inference) / evaluate entry points
├── inference/                 Worked single-case inference example
├── run_reswfm.sh              Genericized train/sample/eval driver script
├── requirements.txt
└── LICENSE, THIRD_PARTY_NOTICES.md, DWT_IDWT/LICENSE
```

## License and third-party notices

This repository is **dual-licensed**:

- Everything except `DWT_IDWT/` is **MIT** (see `LICENSE`).
- `DWT_IDWT/` (the Haar DWT/IDWT implementation) is **CC BY-NC-SA 4.0**
  (Adobe Inc.), **non-commercial only** — see `DWT_IDWT/LICENSE`.

See `THIRD_PARTY_NOTICES.md` for the full explanation and attribution
chain.

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

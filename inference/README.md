# Inference

This folder exists to hold a single example case so you can smoke-test
inference end to end. **It is intentionally empty of data right now** — the
dataset and preprocessing pipeline are not released yet (see the main
README's "What's not included" section).

## Expected layout

Drop one de-identified, approved subject here as:

```
inference/example_case/<SUBJECT_ID>/
    T2.nii.gz
    ADC.nii.gz
    DWI.nii.gz
    NC_T1.nii.gz
    CE_T1.nii.gz       # see note below
    mask.nii.gz         # prostate zone mask: 0=background, 1=TZ, 2=PZ
```

`CE_T1.nii.gz` must be present for `ProstateVolumesRun08` (the data loader) to
pick the subject up at all — it performs a file-completeness check over all
six filenames and silently skips any subject missing one. It is **not**
passed to the model: `wfm_prostate_mask.py`'s conditioning list is
`['t2', 'adc', 'dwi', 'nc_t1']` only, and `CE_T1` is extracted separately as
the returned "target" value, used only to write `*_target_ce_t1.nii.gz` for
comparison. Verified directly in the conditioning-tensor construction code,
not assumed.

All volumes must already be resampled/cropped/normalized exactly as the
model expects (see "Preprocessing" in the main README — not released here,
so for now this means: whatever pipeline you used to produce training data
in this shape). `NC_T1`/`CE_T1` must share the same intensity normalization
window (NC_T1's own foreground window applied to both) — this is required by
the residual formulation (`Δ = CE_T1 − NC_T1`), not optional.

## Running inference on this one case

```bash
echo "<SUBJECT_ID>" > inference/example_case_manifest.txt

python scripts/sample_wfm_prostate_run10.py \
    --data_dir=inference/example_case \
    --manifest_file=inference/example_case_manifest.txt \
    --model_path=<path to a trained checkpoint> \
    --output_dir=inference/example_case_output \
    --sampling_steps=2 \
    --solver=euler \
    --dims=3 --num_channels=64 --num_res_blocks=2 --num_heads=1 \
    --learn_sigma=False --use_scale_shift_norm=False --attention_resolutions= \
    --channel_mult=1,2,2,4,4 --num_groups=32 --in_channels=48 --out_channels=8 \
    --bottleneck_attention=False --resample_2d=False --additive_skips=False \
    --use_freq=False --predict_xstart=True --image_size=160 --class_cond=False \
    --dropout=0.0 --use_checkpoint=False --resblock_updown=True --use_fp16=False \
    --use_new_attention_order=False --num_head_channels=-1 --num_heads_upsample=-1
```

(All the `--channel_mult=1,2,2,4,4 ...` flags must exactly match whatever
architecture the checkpoint was trained with — these are the 160×160×32
ResWFM defaults also baked into `run_reswfm.sh`'s `COMMON_ARGS`. If you train
your own checkpoint with `run_reswfm.sh`, just use
`bash run_reswfm.sh sample_test` instead of calling the script directly, and
point `RESWFM_DATA_DIR`/the test manifest at this folder.)

`--sampling_steps=2 --solver=euler` is the two-step Euler configuration
reported as ResWFM's headline result in the manuscript (21.42 dB PSNR,
0.789 SSIM whole-volume, 0.12 s/subject). `run_reswfm.sh`'s own default is
`--sampling_steps=1 --solver=heun` (a different, also-supported one-step
configuration) — pass the flags explicitly if you want the exact
manuscript-reported setting rather than the script's default.

Output: `inference/example_case_output/<SUBJECT_ID>/<SUBJECT_ID>_predicted_ce_t1.nii.gz`
(plus `_target_ce_t1.nii.gz` if `CE_T1.nii.gz` was present) and a `timing.txt`
with per-subject inference latency.

No trained checkpoint is bundled in this repo yet — see `CHECKPOINTS.md`.

## Verified (2026-10-22)

The command above (with real paths substituted) was run end to end against
a real trained checkpoint (`run10_zonefilm_160x160x32`, step 195000) and one
real test subject (NUP_9), both kept local-only — not committed, not part of
this repository. Checked, not assumed:

- Architecture match: building the model with the flags above and loading
  the checkpoint with `strict=True` succeeded with **zero missing or
  unexpected keys** (402 tensors, 81,543,304 parameters — matching the
  checkpoint exactly).
- `CE_T1.nii.gz`'s role (loader-completeness requirement only, never a model
  input) was confirmed directly in the conditioning-tensor construction code.
- Output: correct 160×160×32 geometry, affine/spacing/orientation matching
  the acquired reference exactly, all values finite in [0, 1], no NaN/Inf.
- Pure model-forward time (the script's own internal timer, excluding
  checkpoint load and data I/O): 3.82 s on CPU for this one subject.
- Single-case sanity check against the acquired CE-T1W (not a benchmark
  result — n=1): PSNR 21.27 dB / SSIM 0.736 (prostate-masked), consistent
  with the manuscript's reported 57-subject whole-volume average (21.42 dB /
  0.789 SSIM). A visual slice comparison showed the predicted volume
  recovering the gland enhancement pattern present in the reference and
  absent from the NC-T1W input, with residual error concentrated at tissue
  boundaries.

This confirms the pipeline is correct for **this one case** — it does not
establish accuracy across the full test cohort, and the checkpoint used
(step 195000) is not the checkpoint behind the manuscript's published Table
I numbers (which used a separately selected best/final checkpoint).

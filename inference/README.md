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
    CE_T1.nii.gz       # ground truth, only needed if you also want to run evaluate_prostate_run10.py
    mask.nii.gz         # prostate zone mask: 0=background, 1=TZ, 2=PZ
```

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
    --sampling_steps=1 \
    --solver=heun \
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

Output: `inference/example_case_output/<SUBJECT_ID>/<SUBJECT_ID>_predicted_ce_t1.nii.gz`
(plus `_target_ce_t1.nii.gz` if `CE_T1.nii.gz` was present) and a `timing.txt`
with per-subject inference latency.

No trained checkpoint is bundled in this repo yet — see `CHECKPOINTS.md`.

# Inference

> The inference script is [`../scripts/sample_wfm_prostate_run10.py`](../scripts/sample_wfm_prostate_run10.py),
> not a file in this folder. This folder documents the inference workflow
> and provides a staging location for example cases — it has no code of
> its own.

This workflow has been exercised end-to-end against a trained checkpoint and
a real test case external to this repository, confirming that the pipeline
produces geometrically and numerically valid output consistent with the
expected reconstruction. It has not been evaluated across a full test cohort
from this repository, and reproducing it requires your own checkpoint and
correctly-formatted input data — neither is distributed here (see the main
[README](../README.md#data-and-pretrained-weights)).

## Expected input layout

Each subject is a folder of NIfTI volumes, named by subject ID:

```
<data_dir>/<SUBJECT_ID>/
    T2.nii.gz
    ADC.nii.gz
    DWI.nii.gz
    NC_T1.nii.gz
    CE_T1.nii.gz          # optional — see note below
    mask.nii.gz           # prostate zone mask: 0=background, 1=TZ, 2=PZ
```

`inference/example_case/` is excluded from version control (`.gitignore`)
so no imaging data is ever committed to this repository; use it as a
convenient local staging location, or point `--data_dir` anywhere else.

`CE_T1.nii.gz` is **optional at inference**. It is never used as a model
input — only `T2`, `ADC`, `DWI`, and `NC_T1` are passed to the network as
conditioning. If present, it is used to write a `*_target_ce_t1.nii.gz`
file alongside each prediction for comparison; if absent, inference still
runs and only the prediction is written.

All volumes must already be resampled, cropped, and normalized exactly as
the model expects (see [Data format](../README.md#data-format-expected-input-contract)
in the main README). `NC_T1` and `CE_T1` must share the same intensity
normalization window — NC_T1's own foreground window applied to both — which
the residual formulation (`Δ = CE_T1 − NC_T1`) requires, not merely a
preprocessing convenience.

## Running inference

The `--channel_mult`, `--image_size`, and related architecture flags in both
examples below must exactly match the configuration the checkpoint was
trained with — these values are the 160×160×32 ResWFM configuration, also
set in `run_reswfm.sh`'s `COMMON_ARGS`. The checkpoint is loaded with strict
parameter matching: if the architecture flags don't match how the checkpoint
was trained, loading fails with a shape-mismatch error rather than silently
producing incorrect output.

`--sampling_steps=2 --solver=euler` is the two-step Euler configuration used
for ResWFM's results in the manuscript. `run_reswfm.sh`'s own default is
`--sampling_steps=1 --solver=heun`, a different, also-supported one-step
configuration — pass the flags explicitly to reproduce the manuscript
setting.

### Single-subject inference

Run on exactly one subject with `--subject_id` — no manifest file needed:

```bash
python scripts/sample_wfm_prostate_run10.py \
    --data_dir=inference/example_case \
    --subject_id=<SUBJECT_ID> \
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

`<data_dir>/<SUBJECT_ID>/` must exist with the required input files. If the
ID isn't found, the script raises a clear error before loading the model.

### Folder-based (batch) inference

Omit `--subject_id` to process every subject folder found under `--data_dir`:

```bash
python scripts/sample_wfm_prostate_run10.py \
    --data_dir=/path/to/a/folder/of/subject/folders \
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

To run on an explicit subset instead of every subject in `--data_dir`, pass
`--manifest_file=<path>` (one subject ID per line) in place of
`--subject_id`. This is what `run_reswfm.sh sample_test` uses internally,
pointed at `RESWFM_DATA_DIR` and the test manifest.

## Output

```
<output_dir>/<SUBJECT_ID>/<SUBJECT_ID>_predicted_ce_t1.nii.gz
<output_dir>/<SUBJECT_ID>/<SUBJECT_ID>_target_ce_t1.nii.gz   # only if CE_T1.nii.gz was provided
<output_dir>/timing.txt
```

The predicted (and, if produced, target) volumes share the input NC_T1
volume's affine, header, and 160×160×32 geometry. `timing.txt` reports
per-subject inference latency.

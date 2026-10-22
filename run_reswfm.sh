#!/bin/bash
# ============================================================
# ResWFM — residual Wavelet Flow Matching + zone-FiLM + frequency-graded
# subband loss, for prostate NC-T1 -> CE-T1 synthesis.
#
# This is a genericized version of the lab's internal run script
# (run_prostate_run10_w_part2_160_160_32.sh): every path below is
# configurable via environment variables with relative defaults, so this
# repo has no machine-specific or user-specific paths baked in. No GCS/VM
# backup step is included here — that was internal infrastructure, not
# part of the method.
#
# Expected data layout (one folder per subject), set via RESWFM_DATA_DIR:
#   <data_dir>/<SUBJECT_ID>/T2.nii.gz
#   <data_dir>/<SUBJECT_ID>/ADC.nii.gz
#   <data_dir>/<SUBJECT_ID>/DWI.nii.gz
#   <data_dir>/<SUBJECT_ID>/NC_T1.nii.gz
#   <data_dir>/<SUBJECT_ID>/CE_T1.nii.gz
#   <data_dir>/<SUBJECT_ID>/mask.nii.gz      (0=background, 1=TZ, 2=PZ)
#
# Manifests (one subject ID per line) via RESWFM_SPLIT_DIR:
#   <split_dir>/train.txt, val.txt, test.txt
#
# Usage:
#   bash run_reswfm.sh sanitycheck
#   bash run_reswfm.sh train
#   bash run_reswfm.sh resume
#   bash run_reswfm.sh sample_test
#   bash run_reswfm.sh eval_test
# ============================================================

set -e

# --- Paths: all overridable via env vars, all relative by default -----
DATA_DIR="${RESWFM_DATA_DIR:-./data/prostate_dataset}"
SPLIT_DIR="${RESWFM_SPLIT_DIR:-./data/splits}"
TRAIN_MANIFEST="${SPLIT_DIR}/train.txt"
VAL_MANIFEST="${SPLIT_DIR}/val.txt"
TEST_MANIFEST="${SPLIT_DIR}/test.txt"

RUN_NAME="${RESWFM_RUN_NAME:-reswfm_run}"
EXPERIMENT_DIR="${RESWFM_EXPERIMENT_DIR:-./experiments/${RUN_NAME}}"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
RESULTS_DIR="${EXPERIMENT_DIR}/results"
TENSORBOARD_DIR="${EXPERIMENT_DIR}/tensorboard_logs"

GPU="${RESWFM_GPU:-0}"
IMAGE_SIZE="${RESWFM_IMAGE_SIZE:-160}"   # in-plane size; z=32 is fixed by the data loader
BATCH_SIZE="${RESWFM_BATCH_SIZE:-4}"
MAX_ITER="${RESWFM_MAX_ITER:-200000}"
SAVE_INTERVAL=5000
LR="${RESWFM_LR:-3e-5}"
LR_MIN="${RESWFM_LR_MIN:-1e-5}"
VAL_INTERVAL=2000
PATIENCE="${RESWFM_PATIENCE:-40}"
MIN_DELTA=0.001

# Zone-FiLM conditioning
FILM_ALPHA=1.0
NUM_ZONES=3           # BG / TZ / PZ

# Frequency-graded subband loss
HF_BOOST=2.0
HF_FREQ_GAMMA=0.5

TZ_WEIGHT=5.0
PZ_WEIGHT=8.0

SAMPLING_STEPS="${RESWFM_SAMPLING_STEPS:-1}"
SOLVER="${RESWFM_SOLVER:-heun}"

MODEL_CKPT="${CHECKPOINT_DIR}/wfm_prostate_best.pt"

MODE=${1:-"train"}

# channel_mult=1,2,2,4,4 (5 levels, ~50.6M params) — sized for 160x160x32
# full-FOV input (post-DWT 80x80x16). See the paper / repo README for the
# architecture rationale.
COMMON_ARGS="
--dims=3
--num_channels=64
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=1,2,2,4,4
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--num_groups=32
--in_channels=48
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--additive_skips=False
--use_freq=False
--predict_xstart=True
--image_size=${IMAGE_SIZE}
--class_cond=False
--dropout=0.0
--use_checkpoint=False
--resblock_updown=True
--use_fp16=False
--use_new_attention_order=False
--num_head_channels=-1
--num_heads_upsample=-1
--sigma_max=0.5
--film_alpha=${FILM_ALPHA}
--num_zones=${NUM_ZONES}
--hf_boost=${HF_BOOST}
--hf_freq_gamma=${HF_FREQ_GAMMA}
"

if [[ "$MODE" == "sanitycheck" ]]; then
    echo "=== ResWFM: sanitycheck (10 iterations) ==="
    echo "Data dir : ${DATA_DIR}"
    echo "Manifest : ${TRAIN_MANIFEST}"
    mkdir -p "${CHECKPOINT_DIR}"
    python scripts/train_wfm_prostate_run10.py \
        --data_dir="${DATA_DIR}" \
        --manifest_file="${TRAIN_MANIFEST}" \
        --devices=${GPU} \
        --batch_size=1 \
        --lr=${LR} \
        --max_iterations=10 \
        --save_interval=10 \
        --log_interval=1 \
        --use_tensorboard=False \
        --tz_weight=${TZ_WEIGHT} \
        --pz_weight=${PZ_WEIGHT} \
        ${COMMON_ARGS}
    echo "=== sanitycheck DONE ==="

elif [[ "$MODE" == "train" ]]; then
    echo "=== ResWFM: TRAINING ==="
    echo "Run       : ${RUN_NAME}"
    echo "Data      : ${DATA_DIR}  (${IMAGE_SIZE}x${IMAGE_SIZE}x32)"
    mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}"
    python scripts/train_wfm_prostate_run10.py \
        --data_dir="${DATA_DIR}" \
        --manifest_file="${TRAIN_MANIFEST}" \
        --val_manifest_file="${VAL_MANIFEST}" \
        --val_interval=${VAL_INTERVAL} \
        --devices=${GPU} \
        --batch_size=${BATCH_SIZE} \
        --lr=${LR} \
        --max_iterations=${MAX_ITER} \
        --save_interval=${SAVE_INTERVAL} \
        --log_interval=100 \
        --use_tensorboard=True \
        --tensorboard_path="${TENSORBOARD_DIR}" \
        --tz_weight=${TZ_WEIGHT} \
        --pz_weight=${PZ_WEIGHT} \
        --patience=${PATIENCE} \
        --min_delta=${MIN_DELTA} \
        --lr_min=${LR_MIN} \
        --roi_normalize_mse_wav=True \
        ${COMMON_ARGS}

elif [[ "$MODE" == "resume" ]]; then
    LATEST=$(ls "${CHECKPOINT_DIR}"/wfm_prostate_mask_*.pt 2>/dev/null | grep -vE '_opt\.pt$' | sort | tail -1)
    if [[ -z "$LATEST" ]]; then echo "No checkpoint to resume from in ${CHECKPOINT_DIR}"; exit 1; fi
    STEP=$(basename "$LATEST" | grep -oE '[0-9]+' | head -1)
    echo "=== ResWFM: RESUME from ${LATEST} (step ${STEP}) -> max ${MAX_ITER} ==="
    mkdir -p "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}"
    python scripts/train_wfm_prostate_run10.py \
        --data_dir="${DATA_DIR}" \
        --manifest_file="${TRAIN_MANIFEST}" \
        --val_manifest_file="${VAL_MANIFEST}" \
        --resume_checkpoint="${LATEST}" \
        --resume_step=${STEP} \
        --val_interval=${VAL_INTERVAL} \
        --devices=${GPU} \
        --batch_size=${BATCH_SIZE} \
        --lr=${LR} \
        --max_iterations=${MAX_ITER} \
        --save_interval=${SAVE_INTERVAL} \
        --log_interval=100 \
        --use_tensorboard=True \
        --tensorboard_path="${TENSORBOARD_DIR}" \
        --tz_weight=${TZ_WEIGHT} \
        --pz_weight=${PZ_WEIGHT} \
        --patience=${PATIENCE} \
        --min_delta=${MIN_DELTA} \
        --lr_min=${LR_MIN} \
        --roi_normalize_mse_wav=True \
        ${COMMON_ARGS}

elif [[ "$MODE" == "sample_test" ]]; then
    echo "=== ResWFM: SAMPLING (test set) ==="
    mkdir -p "${RESULTS_DIR}/test"
    python scripts/sample_wfm_prostate_run10.py \
        --data_dir="${DATA_DIR}" \
        --manifest_file="${TEST_MANIFEST}" \
        --model_path="${MODEL_CKPT}" \
        --output_dir="${RESULTS_DIR}/test" \
        --devices=${GPU} \
        --sampling_steps=${SAMPLING_STEPS} \
        --solver=${SOLVER} \
        ${COMMON_ARGS}

elif [[ "$MODE" == "eval_test" ]]; then
    echo "=== ResWFM: EVALUATION (test set, within prostate mask) ==="
    python scripts/evaluate_prostate_run10.py \
        --pred_dir="${RESULTS_DIR}/test" \
        --output="${EXPERIMENT_DIR}/metrics_ce_t1.txt" \
        --use_mask \
        --verbose

else
    echo "Usage: bash run_reswfm.sh [sanitycheck|train|resume|sample_test|eval_test]"
    exit 1
fi

"""
evaluate_prostate_run10.py  —  Run 10 evaluation (matches run_10 sampler naming)

Identical metrics to evaluate_prostate.py (PSNR/SSIM/MSE/MAE, --use_mask =
non-zero foreground only), but reads the file naming produced by
sample_wfm_prostate_run10.py:

  <pred_dir>/
    NUPT_1/
      NUPT_1_predicted_ce_t1.nii.gz
      NUPT_1_target_ce_t1.nii.gz
    NUPT_2/
      ...

Usage:
  python scripts/evaluate_prostate_run10.py \
      --pred_dir ~/experiments/run_10_roi_film/results/test \
      --output   metrics_ce_t1_run10.txt \
      --use_mask --verbose
"""

import argparse
import os
from glob import glob

import numpy as np
from tqdm import tqdm

# Reuse the exact metric implementation from evaluate_prostate.py (same dir on
# sys.path when run as `python scripts/evaluate_prostate_run10.py`).
from evaluate_prostate import load_nifti, compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate prostate CE_T1 synthesis (run_10 naming)")
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Dir with <subj>/<subj>_predicted_ce_t1.nii.gz + _target_ce_t1.nii.gz')
    parser.add_argument('--output',   type=str, default='metrics_ce_t1_run10.txt')
    parser.add_argument('--use_mask', action='store_true',
                        help='Compute metrics on non-zero region only')
    parser.add_argument('--verbose',  action='store_true',
                        help='Print per-subject metrics')
    args = parser.parse_args()

    pred_files = sorted(
        glob(os.path.join(args.pred_dir, "**/*_predicted_ce_t1.nii.gz"), recursive=True)
    )

    if not pred_files:
        print(f"ERROR: no *_predicted_ce_t1.nii.gz files found in {args.pred_dir}")
        return

    print(f"Found {len(pred_files)} subjects")

    results = {'psnr': [], 'ssim': [], 'mse': [], 'mae': []}
    per_subject = []

    for pred_path in tqdm(pred_files, desc="Evaluating"):
        subj_dir = os.path.dirname(pred_path)
        subj_id  = os.path.basename(subj_dir)
        gt_path  = pred_path.replace("_predicted_ce_t1.nii.gz", "_target_ce_t1.nii.gz")

        if not os.path.exists(gt_path):
            print(f"  WARNING: no target for {subj_id}, skipping")
            continue

        pred = load_nifti(pred_path)
        gt   = load_nifti(gt_path)

        if pred.shape != gt.shape:
            print(f"  WARNING: shape mismatch for {subj_id} "
                  f"pred={pred.shape} gt={gt.shape}, skipping")
            continue

        m = compute_metrics(pred, gt, use_mask=args.use_mask)
        for k, v in m.items():
            results[k].append(v)
        per_subject.append((subj_id, m))

        if args.verbose:
            print(f"  {subj_id}: PSNR={m['psnr']:.2f} dB  "
                  f"SSIM={m['ssim']:.4f}  MSE={m['mse']:.2e}  MAE={m['mae']:.4f}")

    if not results['psnr']:
        print("No valid pairs found.")
        return

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — CE_T1 synthesis (prostate, run_10)")
    print("=" * 60)
    print(f"Subjects evaluated : {len(results['psnr'])}")
    print(f"Mask only          : {args.use_mask}")
    print("-" * 60)
    print(f"PSNR : {np.mean(results['psnr']):7.2f} ± {np.std(results['psnr']):.2f} dB")
    print(f"SSIM : {np.mean(results['ssim']):7.4f} ± {np.std(results['ssim']):.4f}")
    print(f"MSE  : {np.mean(results['mse']):7.2e} ± {np.std(results['mse']):.2e}")
    print(f"MAE  : {np.mean(results['mae']):7.4f} ± {np.std(results['mae']):.4f}")
    print("=" * 60)
    print("\nPercentiles (PSNR / SSIM):")
    for p in [0, 25, 50, 75, 100]:
        print(f"  {p:3d}%  PSNR={np.percentile(results['psnr'], p):.2f}  "
              f"SSIM={np.percentile(results['ssim'], p):.4f}")

    out_path = args.output if os.path.isabs(args.output) \
               else os.path.join(args.pred_dir, args.output)

    with open(out_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION RESULTS — CE_T1 synthesis (prostate, run_10)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Subjects evaluated : {len(results['psnr'])}\n")
        f.write(f"Mask only          : {args.use_mask}\n")
        f.write(f"Prediction dir     : {args.pred_dir}\n")
        f.write("-" * 60 + "\n")
        f.write(f"PSNR : {np.mean(results['psnr']):.2f} ± {np.std(results['psnr']):.2f} dB\n")
        f.write(f"SSIM : {np.mean(results['ssim']):.4f} ± {np.std(results['ssim']):.4f}\n")
        f.write(f"MSE  : {np.mean(results['mse']):.2e} ± {np.std(results['mse']):.2e}\n")
        f.write(f"MAE  : {np.mean(results['mae']):.4f} ± {np.std(results['mae']):.4f}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-subject results:\n")
        f.write("-" * 60 + "\n")
        for sid, m in per_subject:
            f.write(f"{sid}: PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  "
                    f"MSE={m['mse']:.2e}  MAE={m['mae']:.4f}\n")

    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()

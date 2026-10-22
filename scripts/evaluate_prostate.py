"""
Evaluate synthesized CE_T1 images: PSNR, SSIM, MSE, MAE.

Expects output structure from sample_wfm_prostate.py:
  <pred_dir>/
    NUPT_1/
      sample_ce_t1.nii.gz
      target_ce_t1.nii.gz
    NUPT_2/
      ...

Usage:
  python scripts/evaluate_prostate.py \
      --pred_dir /path/to/results_prostate_wfm \
      --output   metrics_ce_t1.txt \
      --use_mask            # compute metrics on non-zero (prostate) region only
      --verbose             # print per-subject results
"""

import argparse
import numpy as np
import os
from glob import glob
from tqdm import tqdm

import nibabel as nib


# ---------------------------------------------------------------------------
# Metric functions (identical to evaluate.py ) 
# ---------------------------------------------------------------------------

def load_nifti(path: str) -> np.ndarray:
    img = nib.load(path).get_fdata()
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0, None)
    if img.max() > 0:
        img = img / img.max()
    return img.astype(np.float32)


def compute_psnr(pred, gt, max_val=1.0):
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim(pred, gt, data_range=1.0):
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(gt, pred, data_range=data_range)
    except ImportError:
        return _ssim_fallback(pred, gt, data_range)


def _ssim_fallback(pred, gt, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_p, mu_g   = np.mean(pred), np.mean(gt)
    s_p,  s_g    = np.std(pred),  np.std(gt)
    s_pg         = np.mean((pred - mu_p) * (gt - mu_g))
    num  = (2 * mu_p * mu_g + C1) * (2 * s_pg + C2)
    den  = (mu_p ** 2 + mu_g ** 2 + C1) * (s_p ** 2 + s_g ** 2 + C2)
    return num / den


def compute_metrics(pred, gt, use_mask=False):
    if use_mask:
        mask = gt > 0
        if mask.sum() == 0:
            return {'psnr': 0.0, 'ssim': 0.0, 'mse': 0.0, 'mae': 0.0}
        p_m, g_m = pred[mask], gt[mask]
        mse      = float(np.mean((p_m - g_m) ** 2))
        mae      = float(np.mean(np.abs(p_m - g_m)))
        max_val  = max(float(g_m.max()), 1.0)
        psnr     = 10 * np.log10(max_val ** 2 / mse) if mse > 0 else float('inf')
        ssim_val = compute_ssim(pred, gt, data_range=max_val)
    else:
        mse      = float(np.mean((pred - gt) ** 2))
        mae      = float(np.mean(np.abs(pred - gt)))
        psnr     = compute_psnr(pred, gt)
        ssim_val = compute_ssim(pred, gt)

    return {'psnr': psnr, 'ssim': ssim_val, 'mse': mse, 'mae': mae}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate prostate CE_T1 synthesis")
    parser.add_argument('--pred_dir',  type=str, required=True,
                        help='Directory with sample_ce_t1.nii.gz / target_ce_t1.nii.gz')
    parser.add_argument('--output',    type=str, default='metrics_ce_t1.txt')
    parser.add_argument('--use_mask',  action='store_true',
                        help='Compute metrics on non-zero region only')
    parser.add_argument('--verbose',   action='store_true',
                        help='Print per-subject metrics')
    args = parser.parse_args()

    # Find all prediction files
    pred_files = sorted(
        glob(os.path.join(args.pred_dir, "**/sample_ce_t1.nii.gz"), recursive=True)
    )

    if not pred_files:
        print(f"ERROR: no sample_ce_t1.nii.gz files found in {args.pred_dir}")
        return

    print(f"Found {len(pred_files)} subjects")

    results = {'psnr': [], 'ssim': [], 'mse': [], 'mae': []}
    per_subject = []

    for pred_path in tqdm(pred_files, desc="Evaluating"):
        subj_dir  = os.path.dirname(pred_path)
        subj_id   = os.path.basename(subj_dir)
        gt_path   = os.path.join(subj_dir, "target_ce_t1.nii.gz")

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

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — CE_T1 synthesis (prostate)")
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

    # Write report
    out_path = args.output if os.path.isabs(args.output) \
               else os.path.join(args.pred_dir, args.output)

    with open(out_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION RESULTS — CE_T1 synthesis (prostate)\n")
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

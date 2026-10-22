"""
sample_wfm_prostate_run10.py  —  Run 10 inference

Residual-WFM sampling (Run 07/09) using the zone-FiLM model.  The model is
built via train_wfm_prostate_run10._build_model so inference uses the exact
same architecture as training, and the mask is threaded through the forward
(handled inside WaveletFlowMatchingProstateRun10.predict_ce_t1).
"""

import argparse
import os
import pathlib
import random
import sys
import time

import nibabel as nib
import numpy as np
import torch as th
from tqdm.auto import tqdm

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser
from guided_diffusion.prostateloader_run08 import ProstateVolumesRun08
from guided_diffusion.wfm_prostate_run10 import WaveletFlowMatchingProstateRun10
# Same closure-based construction (create_model + attached zone-FiLM gates).
from train_wfm_prostate_run10 import _build_model


def main():
    args = create_argparser().parse_args()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.log("=" * 60)
    logger.log("WFM Prostate Sampling — Run 10 (zone-FiLM, ROI)")
    logger.log("=" * 60)
    logger.log(f"Model     : {args.model_path}")
    logger.log(f"Data dir  : {args.data_dir}")
    logger.log(f"Solver    : {args.solver}  steps={args.sampling_steps}")
    logger.log("=" * 60)

    model = _build_model(args)
    logger.log(f"Loading checkpoint from {args.model_path}")
    sd = dist_util.load_state_dict(args.model_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.to(dist_util.dev())
    model.eval()

    wfm = WaveletFlowMatchingProstateRun10(
        num_timesteps=args.diffusion_steps,
        sigma_max=args.sigma_max,
    )

    manifest = args.manifest_file if args.manifest_file else None
    ds = ProstateVolumesRun08(
        args.data_dir,
        mode='eval', manifest_file=manifest,
    )
    logger.log(f"Dataset: {len(ds)} subjects")

    dataloader = th.utils.data.DataLoader(
        ds, batch_size=1, num_workers=4, shuffle=False,
    )

    total_time = 0.0
    all_times  = []                       # pure model-forward (sampler) time per subject
    use_cuda   = th.cuda.is_available()

    for idx, batch in enumerate(tqdm(dataloader, desc="Sampling")):
        if args.num_samples > 0 and idx >= args.num_samples:
            break

        batch = {k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                 for k, v in batch.items()}

        subj = batch['subj'][0] if isinstance(batch['subj'], (list, tuple)) \
               else batch['subj']

        out_subdir = pathlib.Path(args.output_dir) / subj
        out_subdir.mkdir(parents=True, exist_ok=True)

        # CUDA is asynchronous: synchronise before/after so we time the actual
        # kernel execution (the sampler forward pass), not just the launch.
        if use_cuda:
            th.cuda.synchronize()
        t0 = time.time()

        output_img, target = wfm.predict_ce_t1(
            model=model, batch=batch,
            device=dist_util.dev(),
            num_steps=args.sampling_steps,
            solver=args.solver,
            progress=False,
            use_mask=args.use_mask,
        )

        if use_cuda:
            th.cuda.synchronize()
        elapsed = time.time() - t0
        total_time += elapsed
        all_times.append(elapsed)

        output_np = output_img[0, 0].cpu().numpy()
        target_np = target[0, 0].cpu().numpy()

        # Geometry from the actual CE_T1 template -- not a reconstructed
        # affine from a --spacing constant, which had drifted stale (defaulted
        # to the legacy 1.5x1.5x3.0mm ROI-crop spacing while training/data
        # is actually 1.0x1.0x3.0mm; see prostate/data_analysis/README_files/
        # wfm_run10_data_v2_vm_commands.md). Reading the template directly
        # can't go stale again the same way. ProstateVolumesRun08 loads via
        # nibabel.get_fdata() (confirmed in prostateloader_run08.py), so the
        # array axis order here matches the template's affine with no
        # nibabel<->sitk transpose risk.
        tmpl_path = os.path.join(args.data_dir, subj, "CE_T1.nii.gz")
        tmpl_nii = nib.load(tmpl_path)

        nib.save(nib.Nifti1Image(output_np, tmpl_nii.affine, tmpl_nii.header),
                 str(out_subdir / f"{subj}_predicted_ce_t1.nii.gz"))
        nib.save(nib.Nifti1Image(target_np, tmpl_nii.affine, tmpl_nii.header),
                 str(out_subdir / f"{subj}_target_ce_t1.nii.gz"))

        if (idx + 1) % 10 == 0:
            logger.log(f"Processed {idx+1} subjects | "
                       f"avg {total_time/len(all_times):.3f}s/subject")

    # ── Timing statistics (warm-up = first subject, excluded from the average) ──
    times  = np.asarray(all_times, dtype=np.float64)
    n      = int(times.size)
    warm   = float(times[0]) if n else 0.0
    steady = times[1:] if n > 1 else times          # drop CUDA/cuDNN warm-up pass
    avg    = float(steady.mean())   if steady.size else 0.0
    med    = float(np.median(steady)) if steady.size else 0.0
    mn     = float(steady.min())    if steady.size else 0.0
    mx     = float(steady.max())    if steady.size else 0.0
    total  = float(times.sum())
    dev_name = th.cuda.get_device_name(0) if use_cuda else "CPU"

    timing_path = pathlib.Path(args.output_dir) / "timing.txt"
    with open(timing_path, "w") as f:
        f.write(f"Device           : {dev_name}\n")
        f.write(f"Sampling steps   : {args.sampling_steps}\n")
        f.write(f"Solver           : {args.solver}\n")
        f.write(f"Use mask         : {args.use_mask}\n")
        f.write(f"Total subjects   : {n}\n")
        f.write(f"Warmup subj #1   : {warm:.4f} s   (excluded from avg)\n")
        f.write(f"Avg time (s)     : {avg:.4f}      (steady-state, n={steady.size})\n")
        f.write(f"Avg time (ms)    : {avg*1000:.1f}\n")
        f.write(f"Median (s)       : {med:.4f}\n")
        f.write(f"Min / Max (s)    : {mn:.4f} / {mx:.4f}\n")
        f.write(f"Total time (s)   : {total:.2f}\n")

    logger.log("=" * 60)
    logger.log("SAMPLING COMPLETE")
    logger.log(f"Device          : {dev_name}")
    logger.log(f"Total subjects  : {n}")
    logger.log(f"Avg time/subj   : {avg:.4f}s  (warm-up {warm:.3f}s excluded; median {med:.4f}s)")
    logger.log(f"Timing written  : {timing_path}")
    logger.log(f"Results saved to: {args.output_dir}")
    logger.log("=" * 60)


def create_argparser():
    defaults = dict(
        seed=42,
        data_dir="",
        mask_dir="",
        manifest_file="",
        model_path="",
        output_dir="./results_prostate_run10/",
        devices=[0],
        num_samples=-1,
        use_mask=True,            # False -> no-mask inference (null SDF + FiLM identity)
        sampling_steps=1,
        solver='heun',
        sigma_max=0.5,
        film_alpha=1.0,
        num_zones=3,
        hf_boost=2.0,
        hf_freq_gamma=0.5,
        dims=3,
        num_channels=64,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=False,
        use_scale_shift_norm=False,
        attention_resolutions="",
        channel_mult="1,2,2,4",   # must match training (3 downsamples for 96³ ROI)
        diffusion_steps=1000,
        noise_schedule="linear",
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        num_groups=32,
        in_channels=48,
        out_channels=8,
        bottleneck_attention=False,
        resample_2d=False,
        additive_skips=False,
        use_freq=False,
        predict_xstart=True,
        image_size=96,
        class_cond=False,
        dropout=0.0,
        use_checkpoint=False,
        resblock_updown=True,
        use_fp16=False,
        use_new_attention_order=False,
        num_head_channels=-1,
        num_heads_upsample=-1,
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

"""
train_wfm_prostate_run10.py  —  Run 10 training

Residual WFM (Run 07) + zone-conditional FiLM data fusion at each UNet level
boundary + frequency-graded subband-weighted loss, trained on prostate ROI
crops (e.g. 96×96×32 @ 1.5×1.5×3.0 mm) from scratch.

Methodological changes vs Run 09:
  - Fusion:  zone-FiLM (ZoneFiLM) instead of mask-attention gates.  Each zone
             (BG/TZ/PZ) gets its own learned per-channel (gamma, beta) at every
             encoder level — radiologist-style per-zone feature routing.
  - Loss:    frequency-graded subband weighting (WaveletFlowMatchingProstateRun10).
  - Data:    prostate ROI crops (default image_size 96), trained from scratch.

Everything else (residual target, image-space L1/edge/gradient losses,
augmentation, optimiser, training loop) is inherited from Run 07.
"""

import argparse
import random
import sys

import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
sys.path.append("..")

import torch.nn as nn
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser, args_to_dict, create_model,
)
from guided_diffusion.prostateloader_run08 import ProstateVolumesRun08
from guided_diffusion.wfm_prostate_run10 import WaveletFlowMatchingProstateRun10
from guided_diffusion.zone_film          import ZoneFiLM
from guided_diffusion.nn                  import timestep_embedding

# Reuse Run 07 training loop (residual run_step + run_val).
from train_wfm_prostate_run07 import ProstateTrainLoopRun07


def _make_film_forward(model, channel_mult, film_alpha, num_zones):
    """
    Replicate the base UNet forward but apply a zone-FiLM gate after every
    spatial downsample in the encoder.  Downsamples are detected by spatial
    shrink between consecutive input-blocks (no dependency on a specific
    Downsample class), mirroring run_09's mask-attention closure so the
    architecture is identical apart from the gate type.
    """
    import torch as _th

    # Channels seen immediately after each Downsample, BEFORE the next level's
    # ResBlock expands them. For channel_mult=(1,2,2,4,4), model_channels=64:
    #   [64, 128, 128, 256]  (= channel_mult[:-1] * model_channels)
    gate_channels = [model.model_channels * m for m in channel_mult[:-1]]
    model.zone_films = nn.ModuleList([
        ZoneFiLM(feat_channels=c, num_zones=num_zones, alpha=film_alpha)
        for c in gate_channels
    ])

    def forward(x, timesteps, mask=None, **_):
        emb = model.time_embed(
            timestep_embedding(timesteps, model.model_channels)
        )

        hs = []
        h = x.type(model.dtype) if hasattr(model, "dtype") else x
        prev_spatial = h.shape[-3:]
        downsample_count = 0

        for module in model.input_blocks:
            h = module(h, emb)
            new_spatial = h.shape[-3:]

            # If any spatial dim shrank, this block performed a downsample.
            if any(n < p for n, p in zip(new_spatial, prev_spatial)):
                if mask is not None and downsample_count < len(model.zone_films):
                    zone_at_level = F.interpolate(
                        mask.float(), size=new_spatial, mode='nearest',
                    )
                    h = model.zone_films[downsample_count](h, zone_at_level)
                downsample_count += 1
            prev_spatial = new_spatial

            hs.append(h)

        h = model.middle_block(h, emb)

        for module in model.output_blocks:
            h = _th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

        if hasattr(model, "dtype"):
            h = h.type(x.dtype)
        return model.out(h)

    return forward


def _build_model(args):
    """
    Build the base UNet via create_model() (identical config to run_07/09),
    then attach zone-FiLM gates and replace the forward method.
    """
    model_args = args_to_dict(args, [
        'image_size', 'num_channels', 'num_res_blocks', 'channel_mult',
        'learn_sigma', 'class_cond', 'use_checkpoint', 'attention_resolutions',
        'num_heads', 'num_head_channels', 'num_heads_upsample',
        'use_scale_shift_norm', 'dropout', 'resblock_updown', 'use_fp16',
        'use_new_attention_order', 'dims', 'num_groups', 'in_channels',
        'out_channels', 'bottleneck_attention', 'resample_2d', 'additive_skips',
        'use_freq',
    ])
    model = create_model(**model_args)

    channel_mult = tuple(int(c) for c in args.channel_mult.split(","))
    new_forward = _make_film_forward(
        model, channel_mult, args.film_alpha, args.num_zones,
    )
    model.forward = new_forward
    return model


def main():
    args = create_argparser().parse_args()
    if args.lr_min is not None and args.lr_min < 0:
        args.lr_min = None  # sentinel -> flat LR (no decay), see create_argparser()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dist_util.setup_dist(devices=args.devices)

    summary_writer = None
    if args.use_tensorboard:
        logdir = args.tensorboard_path or None
        summary_writer = SummaryWriter(log_dir=logdir)
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    logger.log("=" * 60)
    logger.log("WFM Prostate Run 10 — residual WFM + zone-FiLM + graded subband loss")
    logger.log("=" * 60)
    logger.log(f"Data dir      : {args.data_dir}   (prostate ROI crops)")
    logger.log(f"image_size    : {args.image_size}")
    logger.log(f"batch_size    : {args.batch_size}")
    logger.log(f"film_alpha    : {args.film_alpha}   num_zones: {args.num_zones}")
    logger.log(f"hf_boost      : {args.hf_boost}   hf_freq_gamma: {args.hf_freq_gamma}")
    logger.log(f"LR            : {args.lr}")
    logger.log(f"Max iter      : {args.max_iterations}")
    logger.log(f"Init from     : {args.resume_checkpoint or '(scratch)'}")
    logger.log("=" * 60)

    model = _build_model(args)
    model.to(dist_util.dev())
    total_params = sum(p.numel() for p in model.parameters())
    film_params  = sum(p.numel() for p in model.zone_films.parameters())
    logger.log(f"Model parameters: {total_params:,}")
    logger.log(f"  Zone-FiLM parameters: {film_params:,} "
               f"({film_params / total_params * 100:.3f}%)")

    if args.resume_checkpoint:
        logger.log(f"Loading weights from {args.resume_checkpoint}")
        sd = dist_util.load_state_dict(args.resume_checkpoint,
                                        map_location=dist_util.dev())
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.log(f"  loaded: {len(sd) - len(unexpected)}  "
                   f"missing: {len(missing)}  unexpected: {len(unexpected)}")

    wfm = WaveletFlowMatchingProstateRun10(
        num_timesteps=args.diffusion_steps,
        sigma_max=args.sigma_max,
        tz_weight=args.tz_weight,
        pz_weight=args.pz_weight,
        hf_boost=args.hf_boost,
        hf_freq_gamma=args.hf_freq_gamma,
        roi_normalize_mse_wav=args.roi_normalize_mse_wav,
    )

    train_ds = ProstateVolumesRun08(
        args.data_dir,
        mode='train', manifest_file=args.manifest_file or None,
    )
    logger.log(f"Training set: {len(train_ds)} subjects")

    train_loader = th.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True, drop_last=True,
    )

    val_loader = None
    if args.val_manifest_file:
        val_ds = ProstateVolumesRun08(
            args.data_dir,
            mode='eval', manifest_file=args.val_manifest_file,
        )
        logger.log(f"Validation set: {len(val_ds)} subjects")
        val_loader = th.utils.data.DataLoader(
            val_ds, batch_size=1, num_workers=4, shuffle=False,
        )

    train_loop = ProstateTrainLoopRun07(
        model=model, wfm=wfm,
        data=train_loader, val_loader=val_loader,
        batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval, save_interval=args.save_interval,
        val_interval=args.val_interval, resume_step=args.resume_step,
        max_iterations=args.max_iterations,
        patience=args.patience, min_delta=args.min_delta,
        summary_writer=summary_writer,
        lr_min=args.lr_min,
    )

    train_loop.run_loop()


def create_argparser():
    defaults = dict(
        seed=42,
        data_dir="",
        mask_dir="",
        manifest_file="",
        val_manifest_file="",
        val_interval=2000,
        lr=1e-4,                  # from scratch; residual target is small/smooth
        weight_decay=0.0,
        batch_size=4,             # ROI crops are small → larger batch affordable
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',     # from scratch by default
        resume_step=0,
        max_iterations=40000,
        patience=10,
        min_delta=0.001,
        lr_min=-1.0,               # -1.0 -> None (flat LR, no decay) once parsed below
        use_tensorboard=True,
        tensorboard_path='',
        devices=[0],
        num_workers=8,
        sigma_max=0.5,
        tz_weight=5.0,
        pz_weight=8.0,
        film_alpha=1.0,           # zone-FiLM strength (0 = identity)
        num_zones=3,              # BG / TZ / PZ
        hf_boost=2.0,             # base inside-prostate detail boost
        hf_freq_gamma=0.5,        # extra boost per high-pass axis
        roi_normalize_mse_wav=False,  # opt-in fix for full-FOV mse_wav dilution; see wfm_prostate_run07.py
        dims=3,
        num_channels=64,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=False,
        use_scale_shift_norm=False,
        attention_resolutions="",
        channel_mult="1,2,2,4",   # 3 downsamples — sized for 96³ ROI (bottleneck 6×6×2)
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
        image_size=64,            # prostate ROI crop size
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

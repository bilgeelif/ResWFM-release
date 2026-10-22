"""
train_wfm_prostate_run07.py  —  Run 07: Residual WFM

Training script for the residual reframing of CE_T1 synthesis.
Target is Δ = CE_T1 − NC_T1, source is zero, model learns the enhancement
map directly. See guided_diffusion/wfm_prostate_run07.py for the full
rationale.

Inherits ProstateTrainLoopMask (Run 05) and overrides:
  - run_step   : new loss weights, image-space terms now penalise Δ errors
  - run_val    : uses predict_ce_t1() instead of the parent's hardcoded
                 4-modality-mean source, so the reconstructed CE_T1 is
                 computed via NC_T1 + Δ̂ rather than the wrong starting point.
"""

import argparse
import random
import sys
import time

import numpy as np
import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    create_model, add_dict_to_argparser, args_to_dict,
)
from guided_diffusion.prostateloader_run06 import ProstateVolumesRun06
from guided_diffusion.wfm_prostate_run07 import WaveletFlowMatchingProstateRun07
from guided_diffusion.wfm_prostate_mask import MASK_KEY

from train_wfm_prostate_mask import (
    ProstateTrainLoopMask,
    compute_vol_metrics,
    compute_zone_metrics,
)

from torch.utils.tensorboard import SummaryWriter


class ProstateTrainLoopRun07(ProstateTrainLoopMask):
    """Residual WFM training loop."""

    def run_step(self, batch):
        self.opt.zero_grad()

        t = th.randint(0, self.wfm.num_timesteps, (self.batch_size,),
                       device=dist_util.dev())

        losses, model_output, pred_ce_t1 = self.wfm.training_losses(
            model=self.model, x_start=batch, t=t,
        )

        self._last_channel_losses = losses["mse_wav"]
        self._last_grad_loss      = losses.get("grad_loss",   th.tensor(0.0))
        self._last_img_l1_roi     = losses.get("img_l1_roi",  th.tensor(0.0))
        self._last_img_l1_half    = losses.get("img_l1_half", th.tensor(0.0))
        self._last_edge_l1        = losses.get("edge_l1",     th.tensor(0.0))
        self._last_pz_l1          = losses.get("pz_l1",       th.tensor(0.0))

        # All loss terms operate on Δ. Magnitudes are similar to Run 06's
        # CE_T1-space losses (enhancement inside prostate ≈ 0.1–0.2 voxel
        # intensity).  Same weight schedule.
        loss = (losses["mse_wav"].mean()
                + 0.1 * self._last_grad_loss
                + 2.0 * self._last_img_l1_roi
                + 1.0 * self._last_img_l1_half
                + 1.0 * self._last_edge_l1
                + 0.5 * self._last_pz_l1)

        if self.summary_writer is not None:
            gs = self.step + self.resume_step
            self.summary_writer.add_scalar(
                'train/img_l1_half', self._last_img_l1_half.item(), gs
            )

        if not th.isfinite(loss):
            logger.log(f"Warning: non-finite loss {loss}", level=logger.WARN)

        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.opt.step()

        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("train_loss", loss.item())

        # Return reconstructed CE_T1 for parent's TB image preview
        return loss.detach(), model_output, pred_ce_t1

    @th.no_grad()
    def run_val(self, global_step):
        """
        Residual WFM validation.  Reconstructs CE_T1 = NC_T1 + Δ̂ via the WFM's
        predict_ce_t1 method, so all metrics compare against the GT CE_T1
        consistently with how Run 07 was trained.
        """
        self.model.eval()
        t_val = time.time()

        val_losses, psnrs, ssims, mses, maes = [], [], [], [], []
        wg_grad_scores = []
        zone_metrics = {z: dict(psnr=[], ssim=[], mae=[])
                        for z in ('whole_gland', 'tz', 'pz')}

        for batch in self.val_loader:
            batch = {k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                     for k, v in batch.items()}

            # Val flow-matching loss (operates on Δ)
            t_rand = th.randint(0, self.wfm.num_timesteps,
                                (batch['ce_t1'].shape[0],), device=dist_util.dev())
            losses, _, _ = self.wfm.training_losses(
                model=self.model, x_start=batch, t=t_rand,
            )
            val_losses.append(losses["mse_wav"].mean().item())

            # Reconstruct CE_T1 via Δ̂ + NC_T1 (single-step direct prediction)
            pred_img, target = self.wfm.predict_ce_t1(
                self.model, batch, device=dist_util.dev(), num_steps=1,
            )

            mask = batch[MASK_KEY]

            for b in range(pred_img.shape[0]):
                pred_np = pred_img[b, 0].cpu().numpy()
                gt_np   = target[b, 0].cpu().numpy()
                mask_np = mask[b, 0].cpu().numpy()

                m = compute_vol_metrics(pred_np, gt_np)
                psnrs.append(m['psnr']); ssims.append(m['ssim'])
                mses.append(m['mse']);   maes.append(m['mae'])

                wg_roi = (mask_np > 0).astype(np.float32)
                gx_err = np.abs((pred_np[1:] - pred_np[:-1]) -
                                (gt_np[1:]   - gt_np[:-1]))
                gy_err = np.abs((pred_np[:, 1:] - pred_np[:, :-1]) -
                                (gt_np[:, 1:]   - gt_np[:, :-1]))
                grad_err = (
                    (wg_roi[1:]    * gx_err).sum() / (wg_roi[1:].sum()    + 1e-8) +
                    (wg_roi[:, 1:] * gy_err).sum() / (wg_roi[:, 1:].sum() + 1e-8)
                ) / 2.0
                wg_grad_scores.append(float(1.0 - min(grad_err, 1.0)))

                for zone_name, zone_label in [('whole_gland', None),
                                               ('tz',          1),
                                               ('pz',          2)]:
                    zmask = (mask_np > 0) if zone_label is None \
                            else (mask_np == zone_label)
                    zm = compute_zone_metrics(pred_np, gt_np, zmask.astype(np.float32))
                    zone_metrics[zone_name]['psnr'].append(zm['psnr'])
                    zone_metrics[zone_name]['ssim'].append(zm['ssim'])
                    zone_metrics[zone_name]['mae'].append(zm['mae'])

        self.model.train()

        mean_loss = float(np.mean(val_losses))
        mean_psnr = float(np.mean(psnrs))
        mean_ssim = float(np.mean(ssims))
        mean_mse  = float(np.mean(mses))
        mean_mae  = float(np.mean(maes))

        zm_avg = {z: {k: float(np.mean(v)) for k, v in d.items()}
                  for z, d in zone_metrics.items()}

        elapsed = time.time() - t_val
        logger.log(
            f"[VAL] step {global_step}: loss={mean_loss:.6f}  "
            f"PSNR={mean_psnr:.2f}dB  SSIM={mean_ssim:.4f}  ({elapsed:.1f}s)"
        )
        logger.log(
            f"  [Zones] WG: PSNR={zm_avg['whole_gland']['psnr']:.2f}  "
            f"SSIM={zm_avg['whole_gland']['ssim']:.4f}  "
            f"| TZ: PSNR={zm_avg['tz']['psnr']:.2f}  SSIM={zm_avg['tz']['ssim']:.4f}  "
            f"| PZ: PSNR={zm_avg['pz']['psnr']:.2f}  SSIM={zm_avg['pz']['ssim']:.4f}"
        )

        for k, v in [("val_loss", mean_loss), ("val_PSNR", mean_psnr),
                     ("val_SSIM", mean_ssim), ("val_MSE", mean_mse),
                     ("val_MAE", mean_mae)]:
            logger.logkv(k, v)
        logger.dumpkvs()

        mean_grad_score = float(np.mean(wg_grad_scores))
        wg_psnr  = zm_avg['whole_gland']['psnr']
        wg_ssim  = zm_avg['whole_gland']['ssim']

        psnr_norm       = np.clip((wg_psnr - 15.0) / 13.0, 0.0, 1.0)
        composite_score = 0.40 * psnr_norm + 0.35 * wg_ssim + 0.25 * mean_grad_score
        logger.log(
            f"  [Score] composite={composite_score:.4f}  "
            f"(PSNR_n={psnr_norm:.3f}, SSIM={wg_ssim:.4f}, grad={mean_grad_score:.4f})"
        )
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('val/composite_score', composite_score, global_step)
            self.summary_writer.add_scalar('val/wg_grad_score',   mean_grad_score, global_step)

        if composite_score > self.best_wg_psnr + self.min_delta:
            self.best_wg_psnr            = composite_score
            self._rounds_without_improve = 0
            self._save_best(global_step, wg_psnr)
        else:
            self._rounds_without_improve += 1
            logger.log(
                f"  [EarlyStop] No improve for "
                f"{self._rounds_without_improve}/{self.patience} rounds "
                f"(best={self.best_wg_psnr:.4f}, now={composite_score:.4f})"
            )
            if self._rounds_without_improve >= self.patience:
                self._stop_training = True

        if self.summary_writer is not None:
            for k, v in [('val/loss', mean_loss), ('val/PSNR', mean_psnr),
                         ('val/SSIM', mean_ssim), ('val/MSE', mean_mse),
                         ('val/MAE', mean_mae)]:
                self.summary_writer.add_scalar(k, v, global_step)
            for zone in ('whole_gland', 'tz', 'pz'):
                for metric in ('psnr', 'ssim', 'mae'):
                    self.summary_writer.add_scalar(
                        f'val_zone/{zone}_{metric}',
                        zm_avg[zone][metric], global_step
                    )


def main():
    args = create_argparser().parse_args()

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
    logger.log("WFM Prostate Run 07 — Residual WFM (target = Δ = CE_T1 − NC_T1)")
    logger.log("=" * 60)
    logger.log(f"Data dir    : {args.data_dir}")
    logger.log(f"Mask dir    : {args.mask_dir}")
    logger.log(f"image_size  : {args.image_size}")
    logger.log(f"LR          : {args.lr}")
    logger.log(f"Max iter    : {args.max_iterations}")
    logger.log(f"Init from   : {args.resume_checkpoint or '(scratch)'}")
    logger.log(f"TZ / PZ wt  : {args.tz_weight} / {args.pz_weight}")
    logger.log("=" * 60)

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
    model.to(dist_util.dev())
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.resume_checkpoint:
        logger.log(f"Loading from {args.resume_checkpoint}")
        model.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint,
                                      map_location=dist_util.dev())
        )

    wfm = WaveletFlowMatchingProstateRun07(
        num_timesteps=args.diffusion_steps,
        sigma_max=args.sigma_max,
        tz_weight=args.tz_weight,
        pz_weight=args.pz_weight,
    )

    train_ds = ProstateVolumesRun06(
        args.data_dir, mask_dir=args.mask_dir,
        mode='train', manifest_file=args.manifest_file or None,
    )
    logger.log(f"Training set: {len(train_ds)} subjects")

    train_loader = th.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True, drop_last=True,
    )

    val_loader = None
    if args.val_manifest_file:
        val_ds = ProstateVolumesRun06(
            args.data_dir, mask_dir=args.mask_dir,
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
    )

    train_loop.run_loop()


def create_argparser():
    defaults = dict(
        seed=42,
        data_dir="",
        manifest_file="",
        val_manifest_file="",
        val_interval=2000,
        lr=1e-5,                  # full training rate (from-scratch friendly)
        weight_decay=0.0,
        batch_size=4,
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',     # optional 48-ch warm-start
        resume_step=0,
        max_iterations=60000,     # train budget
        patience=10,
        min_delta=0.001,
        use_tensorboard=True,
        tensorboard_path='',
        devices=[0],
        num_workers=8,
        sigma_max=0.5,
        mask_dir="",
        tz_weight=5.0,
        pz_weight=8.0,
        dims=3,
        num_channels=64,
        num_res_blocks=2,
        num_heads=1,
        learn_sigma=False,
        use_scale_shift_norm=False,
        attention_resolutions="",
        channel_mult="1,2,2,4,4",
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
        image_size=160,
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

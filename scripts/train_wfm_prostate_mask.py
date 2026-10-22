"""
train_wfm_prostate_mask.py  —  Run 03

Training script for WFM with T2 prostate mask conditioning and zone-weighted
loss.  Mirrors train_wfm_prostate.py exactly, with three additions:

  1. Uses ProstateVolumesMask (loads mask.nii.gz alongside sequences)
  2. Uses WaveletFlowMatchingProstateMask (48-channel input, weighted loss)
  3. Adds --tz_weight and --pz_weight arguments
  4. in_channels default changed from 40 → 48

Run with run_prostate_mask.sh — do not call directly.
"""

import argparse
import math
import numpy as np
import os
import random
import sys
import time

import torch as th
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import create_model, add_dict_to_argparser, args_to_dict
from guided_diffusion.prostateloader_mask import ProstateVolumesMask
from guided_diffusion.wfm_prostate_mask import (
    WaveletFlowMatchingProstateMask, apply_dwt, apply_idwt,
    PROSTATE_TARGET, PROSTATE_CONDITIONS, MASK_KEY,
)


# ---------------------------------------------------------------------------
# Metric helpers (identical to train_wfm_prostate.py)
# ---------------------------------------------------------------------------

def _psnr(pred, gt, max_val=1.0):
    mse = np.mean((pred - gt) ** 2)
    return 10 * np.log10(max_val ** 2 / mse) if mse > 1e-10 else 100.0


def _ssim(pred, gt, data_range=1.0):
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(gt, pred, data_range=data_range))
    except ImportError:
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        mu_p, mu_g = pred.mean(), gt.mean()
        sg = np.std(gt); sp = np.std(pred)
        cov = np.mean((pred - mu_p) * (gt - mu_g))
        num = (2 * mu_p * mu_g + C1) * (2 * cov + C2)
        den = (mu_p**2 + mu_g**2 + C1) * (sp**2 + sg**2 + C2)
        return float(num / den)


def compute_vol_metrics(pred_np, gt_np):
    """Full-volume metrics on foreground voxels (gt > 0)."""
    mask = gt_np > 0
    if mask.sum() == 0:
        return dict(psnr=0.0, ssim=0.0, mse=0.0, mae=0.0)
    p, g = pred_np[mask], gt_np[mask]
    mse  = float(np.mean((p - g) ** 2))
    mae  = float(np.mean(np.abs(p - g)))
    psnr = _psnr(p, g, max_val=max(float(g.max()), 1.0))
    ssim = _ssim(pred_np, gt_np)
    return dict(psnr=psnr, ssim=ssim, mse=mse, mae=mae)


def _bbox_crop(arr, mask):
    """Crop arr to the tight 3D bounding box of mask > 0."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return arr
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    return arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def compute_zone_metrics(pred_np, gt_np, zone_mask_np):
    """
    Zone-specific metrics (the key monitoring signal for Run 03).

    PSNR / MAE — on zone voxels only (scattered 1D extraction).
    SSIM       — global formula on zone bbox crop (no sliding window).
                 Matches evaluate_roi_vs_whole_oneDir.py methodology.
    """
    roi = zone_mask_np > 0
    if roi.sum() == 0:
        return dict(psnr=0.0, ssim=0.0, mae=0.0)
    p_roi, g_roi = pred_np[roi], gt_np[roi]
    mse  = float(np.mean((p_roi - g_roi) ** 2))
    mae  = float(np.mean(np.abs(p_roi - g_roi)))
    psnr = _psnr(p_roi, g_roi)

    # Global SSIM on bbox crop — zone-specific, no background dilution
    pred_crop = _bbox_crop(pred_np, roi)
    gt_crop   = _bbox_crop(gt_np,   roi)
    C1, C2    = (0.01) ** 2, (0.03) ** 2
    mu_p, mu_g = pred_crop.mean(), gt_crop.mean()
    s_p,  s_g  = pred_crop.std(),  gt_crop.std()
    cov        = np.mean((pred_crop - mu_p) * (gt_crop - mu_g))
    ssim = float(
        ((2 * mu_p * mu_g + C1) * (2 * cov + C2)) /
        ((mu_p**2 + mu_g**2 + C1) * (s_p**2 + s_g**2 + C2))
    )
    return dict(psnr=psnr, ssim=ssim, mae=mae)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class ProstateTrainLoopMask:
    """Training loop for Run 03 (mask conditioning + zone-weighted loss)."""

    def __init__(self, model, wfm, data, val_loader,
                 batch_size, lr, weight_decay,
                 log_interval, save_interval, val_interval,
                 resume_step, max_iterations,
                 patience=10, min_delta=0.001,
                 summary_writer=None, keep_last_checkpoints=2,
                 lr_min=None):
        self.keep_last_checkpoints = int(keep_last_checkpoints)
        self.model          = model
        self.wfm            = wfm
        self.datal          = data
        self.iterdatal      = iter(data)
        self.val_loader     = val_loader
        self.batch_size     = batch_size
        self.log_interval   = log_interval
        self.save_interval  = save_interval
        self.val_interval   = val_interval
        self.resume_step    = resume_step
        self.max_iterations = max_iterations
        self.summary_writer = summary_writer
        self.step           = 1
        self.best_wg_psnr   = -float('inf')   # tracks best composite structural score

        # Early stopping: count validation rounds without meaningful improvement.
        # patience=10 with val_interval=2000 → stops after 20k steps without gain.
        # min_delta=0.001 filters out noise on 31 validation subjects.
        self.patience               = patience
        self.min_delta              = min_delta
        self._rounds_without_improve = 0
        self._stop_training         = False

        # Optional cosine LR decay, opt-in via lr_min (None = flat LR, the
        # original/default behavior -- other scripts reusing this TrainLoop
        # are unaffected unless they explicitly pass lr_min). Decays over the
        # REMAINING budget from resume_step to max_iterations, not from step 0,
        # so a resume doesn't introduce a discontinuous LR jump: LR stays at
        # `lr` exactly at the resume point and reaches lr_min exactly at
        # max_iterations. If resuming from a checkpoint whose optimizer state
        # carries an old LR, _update_lr() overwrites it on the very first
        # run_loop iteration regardless (called unconditionally every step).
        self.base_lr = lr
        self.lr_min  = lr_min

        self.opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        if resume_step > 0:
            self._load_optimizer_state()

    def _update_lr(self, global_step):
        if self.lr_min is None:
            return self.base_lr
        total    = max(self.max_iterations - self.resume_step, 1)
        progress = min(max(global_step - self.resume_step, 0) / total, 1.0)
        new_lr   = self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (1 + math.cos(math.pi * progress))
        for g in self.opt.param_groups:
            g['lr'] = new_lr
        return new_lr

    def _load_optimizer_state(self):
        opt_checkpoint = os.path.join(
            logger.get_dir(), 'checkpoints',
            f"wfm_prostate_mask_{self.resume_step:06d}_opt.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from: {opt_checkpoint}")
            self.opt.load_state_dict(
                th.load(opt_checkpoint, map_location=dist_util.dev())
            )

    def run_loop(self):
        t_start = time.time()

        while self.step + self.resume_step < self.max_iterations and not self._stop_training:
            t_iter = time.time()

            try:
                batch = next(self.iterdatal)
            except StopIteration:
                self.iterdatal = iter(self.datal)
                batch = next(self.iterdatal)

            batch = {k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                     for k, v in batch.items()}

            # LR updated before run_step so this iteration's optimizer.step()
            # actually uses the freshly computed rate, not last iteration's.
            global_step = self.step + self.resume_step
            current_lr  = self._update_lr(global_step)

            t_load = time.time() - t_iter
            t_fwd  = time.time()
            loss, sample, sample_idwt = self.run_step(batch)
            t_fwd  = time.time() - t_fwd

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('train/lr',        current_lr,  global_step)
                self.summary_writer.add_scalar('train/loss',      loss.item(), global_step)
                self.summary_writer.add_scalar('time/load',       t_load,      global_step)
                self.summary_writer.add_scalar('time/forward',    t_fwd,       global_step)
                if hasattr(self, '_last_grad_loss'):
                    self.summary_writer.add_scalar(
                        'train/grad_loss_roi',
                        self._last_grad_loss.item(), global_step
                    )
                if hasattr(self, '_last_img_l1_roi'):
                    self.summary_writer.add_scalar(
                        'train/img_l1_roi',
                        self._last_img_l1_roi.item(), global_step
                    )
                    self.summary_writer.add_scalar(
                        'train/edge_l1',
                        self._last_edge_l1.item(), global_step
                    )
                    self.summary_writer.add_scalar(
                        'train/pz_l1',
                        self._last_pz_l1.item(), global_step
                    )
                    # mse_wav baseline for at-a-glance balance check vs new terms
                    self.summary_writer.add_scalar(
                        'train/mse_wav_mean',
                        self._last_channel_losses.mean().item(), global_step
                    )

                names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
                if hasattr(self, '_last_channel_losses'):
                    for ch, name in enumerate(names):
                        self.summary_writer.add_scalar(
                            f'train/loss_wav_{name.lower()}',
                            self._last_channel_losses[ch].item(), global_step
                        )

                if self.step % 200 == 0:
                    D = sample_idwt.size(-1)
                    self.summary_writer.add_image(
                        'train/ce_t1_pred',
                        sample_idwt[0, 0, :, :, D // 2].unsqueeze(0), global_step
                    )

            if self.step % self.log_interval == 0:
                elapsed = time.time() - t_start
                logger.log(
                    f"step {global_step}: train_loss={loss.item():.6f}, "
                    f"lr={current_lr:.2e}, "
                    f"elapsed={elapsed:.1f}s, iter/s={self.step/elapsed:.2f}"
                )
                logger.dumpkvs()

            if self.val_loader is not None and self.step % self.val_interval == 0:
                self.run_val(global_step)

            if self.step % self.save_interval == 0:
                self.save()

            self.step += 1

        if self._stop_training:
            logger.log(
                f"[EARLY STOP] No improvement for {self.patience} validation "
                f"rounds (min_delta={self.min_delta}). Stopping training."
            )
        # Always save final state and run a last validation pass
        if (self.step - 1) % self.save_interval != 0:
            self.save()

        if self.val_loader is not None:
            self.run_val(self.step + self.resume_step)

        logger.log("Training complete!")

    def run_step(self, batch):
        self.opt.zero_grad()

        t = th.randint(0, self.wfm.num_timesteps, (self.batch_size,),
                       device=dist_util.dev())

        losses, model_output, model_output_idwt = self.wfm.training_losses(
            model=self.model, x_start=batch, t=t,
        )

        self._last_channel_losses = losses["mse_wav"]
        self._last_grad_loss      = losses.get("grad_loss",  th.tensor(0.0))
        self._last_img_l1_roi     = losses.get("img_l1_roi", th.tensor(0.0))
        self._last_edge_l1        = losses.get("edge_l1",    th.tensor(0.0))
        self._last_pz_l1          = losses.get("pz_l1",      th.tensor(0.0))

        # Dual-domain loss: wavelet MSE (global block-summary) + image-space
        # ROI L1 (full-resolution detail inside the prostate). Watch
        # train/img_l1_roi vs train/mse_wav_mean in TensorBoard and rescale
        # if the ratio drifts outside ~[1/3, 3].
        loss = (losses["mse_wav"].mean()
                + 0.1 * self._last_grad_loss
                + 1.0 * self._last_img_l1_roi
                + 0.5 * self._last_edge_l1
                + 0.5 * self._last_pz_l1)

        if not th.isfinite(loss):
            logger.log(f"Warning: non-finite loss {loss}", level=logger.WARN)

        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.opt.step()

        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("train_loss", loss.item())

        return loss.detach(), model_output, model_output_idwt

    @th.no_grad()
    def run_val(self, global_step):
        self.model.eval()
        t_val = time.time()

        val_losses, psnrs, ssims, mses, maes = [], [], [], [], []
        wg_grad_scores = []   # gradient agreement inside WG (structural sharpness proxy)
        # Zone-specific (the key monitoring metrics for Run 03)
        zone_metrics = {z: dict(psnr=[], ssim=[], mae=[])
                        for z in ('whole_gland', 'tz', 'pz')}

        for batch in self.val_loader:
            batch = {k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
                     for k, v in batch.items()}

            # Val flow-matching loss (includes mask + zone weights)
            t_rand = th.randint(0, self.wfm.num_timesteps,
                                (batch['ce_t1'].shape[0],), device=dist_util.dev())
            losses, _, _ = self.wfm.training_losses(
                model=self.model, x_start=batch, t=t_rand,
            )
            val_losses.append(losses["mse_wav"].mean().item())

            # Direct prediction → image domain
            target, conditions = self.wfm._get_target_and_conditions(batch)
            mask               = batch[MASK_KEY]
            sdf                = batch.get('sdf', None)
            cond_dwt, dwt_list = self.wfm._build_cond_dwt(conditions, mask, sdf=sdf)
            source_dwt         = sum(dwt_list) / len(dwt_list)

            pred_dwt = self.wfm.direct_prediction(
                self.model, cond=cond_dwt, source=source_dwt,
                device=dist_util.dev(),
            )
            pred_img = apply_idwt(pred_dwt).clamp(0., 1.)
            pred_img[batch['nc_t1'] == 0] = 0

            for b in range(pred_img.shape[0]):
                pred_np = pred_img[b, 0].cpu().numpy()
                gt_np   = target[b, 0].cpu().numpy()
                mask_np = mask[b, 0].cpu().numpy()   # values 0/1/2

                # Full-volume metrics
                m = compute_vol_metrics(pred_np, gt_np)
                psnrs.append(m['psnr']); ssims.append(m['ssim'])
                mses.append(m['mse']);   maes.append(m['mae'])

                # Gradient agreement inside WG (boundary sharpness proxy).
                # Score in [0, 1]: higher → sharper, more accurate prostate edges.
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

                # Zone-specific metrics (primary signal for Run 03)
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

        # Zone averages
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

        # Composite structural quality score.
        # PSNR: normalised to roughly [0,1] via a 15–28 dB expected range.
        # SSIM and grad_score are already in [0,1].
        # Weights: 40% PSNR (intensity accuracy) + 35% SSIM (structural) +
        #          25% gradient agreement (boundary sharpness).
        psnr_norm      = np.clip((wg_psnr - 15.0) / 13.0, 0.0, 1.0)
        composite_score = 0.40 * psnr_norm + 0.35 * wg_ssim + 0.25 * mean_grad_score
        logger.log(
            f"  [Score] composite={composite_score:.4f}  "
            f"(PSNR_n={psnr_norm:.3f}, SSIM={wg_ssim:.4f}, grad={mean_grad_score:.4f})"
        )
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('val/composite_score', composite_score, global_step)
            self.summary_writer.add_scalar('val/wg_grad_score',   mean_grad_score, global_step)

        # Best model + early stopping: composite structural quality score.
        # An improvement must exceed min_delta to reset the patience counter,
        # filtering noise from the small 31-subject validation set.
        if composite_score > self.best_wg_psnr + self.min_delta:
            self.best_wg_psnr          = composite_score
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
            # Zone metrics in TensorBoard — the key curves to watch
            for zone in ('whole_gland', 'tz', 'pz'):
                for metric in ('psnr', 'ssim', 'mae'):
                    self.summary_writer.add_scalar(
                        f'val_zone/{zone}_{metric}',
                        zm_avg[zone][metric], global_step
                    )

    def save(self):
        if dist.get_rank() == 0:
            ckpt_dir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            step = self.step + self.resume_step

            model_path = os.path.join(ckpt_dir, f"wfm_prostate_mask_{step:06d}.pt")
            logger.log(f"Saving model to {model_path}")
            th.save(self.model.state_dict(), model_path)

            opt_path = os.path.join(ckpt_dir, f"wfm_prostate_mask_{step:06d}_opt.pt")
            th.save(self.opt.state_dict(), opt_path)

            self._prune_checkpoints(ckpt_dir)

    def _prune_checkpoints(self, ckpt_dir):
        """Disk-safe checkpoint rotation. Keeps:
          - wfm_prostate_best.pt                              (never touched here)
          - the last `keep_last_checkpoints` model .pt        (for resume / safety)
          - the last `keep_last_checkpoints` optimizer states (largest files; a 2nd
            copy guards against a corrupted latest save, e.g. a disk-full crash)
        Everything older is deleted so the disk can't fill up."""
        import glob, re
        keep = max(int(getattr(self, "keep_last_checkpoints", 2)), 1)

        def step_of(p):
            m = re.search(r"wfm_prostate_mask_(\d+)(?:_opt)?\.pt$", os.path.basename(p))
            return int(m.group(1)) if m else -1

        all_ckpts = glob.glob(os.path.join(ckpt_dir, "wfm_prostate_mask_*.pt"))
        models = sorted([p for p in all_ckpts if not p.endswith("_opt.pt")], key=step_of)
        opts   = sorted([p for p in all_ckpts if p.endswith("_opt.pt")],     key=step_of)

        for p in models[:-keep]:          # keep last `keep` model checkpoints
            try: os.remove(p)
            except OSError: pass
        for p in opts[:-keep]:            # keep last `keep` optimizer states
            try: os.remove(p)
            except OSError: pass

    def _save_best(self, global_step, wg_psnr):
        """Save best checkpoint whenever whole-gland PSNR improves."""
        if dist.get_rank() == 0:
            ckpt_dir = os.path.join(logger.get_dir(), 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)

            best_path = os.path.join(ckpt_dir, "wfm_prostate_best.pt")
            th.save(self.model.state_dict(), best_path)

            # Write a small info file so run_prostate_mask.sh sample_test
            # can identify which step the best checkpoint corresponds to
            info_path = os.path.join(ckpt_dir, "best_model_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"step={global_step}\n")
                f.write(f"whole_gland_psnr={wg_psnr:.4f}\n")
            logger.log(
                f"  [BEST] step {global_step}: WG PSNR={wg_psnr:.4f}dB "
                f"→ saved wfm_prostate_best.pt"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = create_argparser().parse_args()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dist_util.setup_dist(devices=args.devices)

    summary_writer = None
    if args.use_tensorboard:
        logdir = args.tensorboard_path if args.tensorboard_path else None
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    logger.log("=" * 60)
    logger.log("WFM Prostate Run 05 — dual-domain loss (wavelet + full-res ROI L1)")
    logger.log("=" * 60)
    logger.log(f"Data dir      : {args.data_dir}")
    logger.log(f"Train manifest: {args.manifest_file}")
    logger.log(f"Val manifest  : {args.val_manifest_file or '(none)'}")
    logger.log(f"TZ weight     : {args.tz_weight}")
    logger.log(f"PZ weight     : {args.pz_weight}")
    logger.log(f"in_channels   : {args.in_channels}  (was 40, now 48 for mask)")
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

    num_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model parameters: {num_params:,}")

    if args.resume_checkpoint:
        logger.log(f"Loading model from {args.resume_checkpoint}")
        model.load_state_dict(
            dist_util.load_state_dict(args.resume_checkpoint,
                                      map_location=dist_util.dev())
        )

    wfm = WaveletFlowMatchingProstateMask(
        num_timesteps=args.diffusion_steps,
        sigma_max=args.sigma_max,
        tz_weight=args.tz_weight,
        pz_weight=args.pz_weight,
    )

    train_manifest = args.manifest_file if args.manifest_file else None
    train_ds = ProstateVolumesMask(args.data_dir, mask_dir=args.mask_dir,
                                   mode='train', manifest_file=train_manifest)
    logger.log(f"Training set: {len(train_ds)} subjects")
    logger.log(f"Mask dir    : {args.mask_dir}")

    train_loader = th.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True, drop_last=True,
    )

    val_loader = None
    if args.val_manifest_file:
        val_ds = ProstateVolumesMask(args.data_dir, mask_dir=args.mask_dir,
                                     mode='eval', manifest_file=args.val_manifest_file)
        logger.log(f"Validation set: {len(val_ds)} subjects")
        val_loader = th.utils.data.DataLoader(
            val_ds, batch_size=1, num_workers=4, shuffle=False,
        )

    train_loop = ProstateTrainLoopMask(
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
        lr=1e-5,
        weight_decay=0.0,
        batch_size=4,
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        max_iterations=200000,
        # Early stopping: patience=10 val rounds × val_interval=2000 = 20k steps
        # without composite score improvement before training halts.
        patience=10,
        min_delta=0.001,
        use_tensorboard=True,
        tensorboard_path='',
        devices=[0],
        num_workers=8,
        sigma_max=0.5,
        # Mask directory (flat dir with NUPT_XXX_mask.nii.gz)
        mask_dir="",
        # Zone weights
        tz_weight=5.0,
        pz_weight=8.0,
        # Model — note in_channels=48 (was 40)
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
        in_channels=48,     # 8 noisy + 40 cond (4 seqs + mask) × 8 wavelet
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

"""
wfm_prostate_run07.py  —  Run 07: Residual Wavelet Flow Matching

Reframes the synthesis task. Instead of generating CE_T1 directly, the model
predicts the enhancement map Δ = CE_T1 − NC_T1.  At inference the predicted Δ
is added back to NC_T1.

    target  = CE_T1 − NC_T1                 (the enhancement map)
    source  = 0                             (trivial flow start)
    velocity = target − source = Δ
    model learns to flow 0 → Δ in wavelet space
    inference: CE_T1 = NC_T1 + IDWT(Δ̂)

Why this reframing matters
--------------------------
1. Δ is structurally sparse — zero outside vascular tissue, concentrated
   inside the gland. The wavelet decomposition of a sparse signal puts
   meaningful information in the detail subbands, which is exactly where
   prior full-image-target WFM runs lost signal to block-summary averaging.

2. The model no longer wastes capacity reproducing the prostate anatomy on
   every forward pass — anatomy comes for free from NC_T1.  All learning
   capacity goes to the actual physical signal (gadolinium uptake patterns).

3. All loss terms (wavelet MSE, image-space L1 at multiple scales, edge L1,
   PZ-specific L1, gradient agreement) operate on Δ rather than CE_T1.  Each
   gradient signal now directly penalises enhancement errors — the
   clinically relevant quantity — rather than mixing structure and contrast.

Architectural changes from Run 06: NONE.  Same 48-channel UNet, same
conditioning tensor.  Only the target and source change.
"""

import torch as th
import torch.nn.functional as F

from .wfm_prostate_mask import (
    WaveletFlowMatchingProstateMask,
    apply_dwt, apply_idwt,
    PROSTATE_TARGET, PROSTATE_CONDITIONS, MASK_KEY, SDF_KEY,
)

NC_T1_KEY = 'nc_t1'


class WaveletFlowMatchingProstateRun07(WaveletFlowMatchingProstateMask):

    def _zero_source(self, delta_dwt):
        """Source for residual WFM is exactly zero (trivial flow start)."""
        return th.zeros_like(delta_dwt)

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}

        target_image, conditions = self._get_target_and_conditions(x_start)   # CE_T1
        nc_t1 = x_start[NC_T1_KEY]
        mask  = x_start[MASK_KEY]
        sdf   = x_start.get(SDF_KEY, None)

        # REFRAMING: target is the enhancement map Δ, not CE_T1
        delta = target_image - nc_t1                # [B, 1, X, Y, Z]

        cond_dwt, dwt_list = self._build_cond_dwt(conditions, mask, sdf=sdf)
        zone_weight_dwt    = self._build_zone_weight_wav(mask)

        delta_dwt  = apply_dwt(delta)
        source_dwt = self._zero_source(delta_dwt)   # zeros — clean start of flow

        t_continuous = t.float() / self.num_timesteps

        if noise is None:
            noise = th.randn_like(delta)
        noise_dwt = apply_dwt(noise)

        x_t      = self.q_sample(source_dwt, delta_dwt, t_continuous, noise=noise_dwt)
        x_t_cond = th.cat([x_t, cond_dwt], dim=1)

        t_scaled     = self._scale_timesteps(t_continuous)
        model_output = model(x_t_cond, t_scaled, **model_kwargs)

        # velocity = Δ_dwt − 0 = Δ_dwt
        velocity        = delta_dwt - source_dwt
        squared_error   = (velocity - model_output) ** 2
        weighted_error  = squared_error * zone_weight_dwt

        if self.roi_normalize_mse_wav:
            # ROI-normalized (sum / weight-mass), not a plain mean over all
            # voxels. A plain mean lets background voxel COUNT dilute the
            # gland's share of this term's gradient at full 160x160x32 FOV --
            # unlike img_l1_roi/edge_l1/pz_l1 below, which were already
            # normalized by their own ROI voxel counts (n_wg/n_b/n_pz) and so
            # were FOV-size-invariant from the start. At the original tight
            # 64x64x32/96x96x32 ROI crop this term's gland-focus came "for
            # free" from geometry (gland was a large share of total voxels);
            # at full-pelvis FOV the gland is a small minority of voxels, so
            # even with the 5x/8x TZ/PZ weight boost, plain-mean gradient mass
            # from gland dropped substantially (est. to roughly a quarter of
            # what it was at the tight crop) -- a plausible driver of run10
            # trailing avgWFM on whole-gland PSNR/SSIM despite competitive
            # whole-volume PSNR. Opt-in (roi_normalize_mse_wav=True), used only
            # by the 160x160x32 lineage -- see class docstring / __init__.
            weight_mass = zone_weight_dwt.sum(dim=[0, 2, 3, 4]) + 1e-8
            mse_per_ch  = weighted_error.sum(dim=[0, 2, 3, 4]) / weight_mass
        else:
            # Original behavior (plain mean over all voxels) -- unchanged,
            # used by the frozen 64x64x32 legacy lineage.
            mse_per_ch  = weighted_error.mean(dim=[0, 2, 3, 4])

        # Predicted Δ in image domain (this is what we add to NC_T1)
        pred_delta_dwt  = source_dwt + model_output                # = model_output
        pred_delta_idwt = apply_idwt(pred_delta_dwt)

        # All image-space losses operate on Δ vs Δ̂ — clinically meaningful
        grad_loss = self._gradient_loss_roi(pred_delta_idwt, delta, mask)

        wg   = (mask > 0).float()
        n_wg = wg.sum() + 1e-8
        img_l1_roi = ((pred_delta_idwt - delta).abs() * wg).sum() / n_wg

        # Multi-scale: half-resolution Δ L1
        pred_delta_half = F.avg_pool3d(pred_delta_idwt, kernel_size=2, stride=2)
        delta_half      = F.avg_pool3d(delta,           kernel_size=2, stride=2)
        wg_half         = F.avg_pool3d(wg,              kernel_size=2, stride=2)
        n_wg_half       = wg_half.sum() + 1e-8
        img_l1_half = ((pred_delta_half - delta_half).abs() * wg_half).sum() / n_wg_half

        # Capsule ring on Δ — enhancement boundary sharpness
        band = F.max_pool3d(wg, kernel_size=3, stride=1, padding=1) - wg
        n_b  = band.sum() + 1e-8
        edge_l1 = ((pred_delta_idwt - delta).abs() * band).sum() / n_b

        # PZ-only Δ L1
        pz   = (mask == 2).float()
        n_pz = pz.sum() + 1e-8
        pz_l1 = ((pred_delta_idwt - delta).abs() * pz).sum() / n_pz

        # Reconstructed CE_T1 (for TensorBoard image previews; not used in loss)
        pred_ce_t1 = (nc_t1 + pred_delta_idwt).clamp(0., 1.)

        return {
            "mse_wav":     mse_per_ch,
            "grad_loss":   grad_loss,
            "img_l1_roi":  img_l1_roi,
            "img_l1_half": img_l1_half,
            "edge_l1":     edge_l1,
            "pz_l1":       pz_l1,
        }, model_output, pred_ce_t1

    def predict_ce_t1(self, model, batch, device=None, num_steps=1,
                      solver='heun', progress=False, clip_intermediate=False):
        """
        Run inference: predict Δ̂, reconstruct CE_T1 = NC_T1 + Δ̂.

        clip_intermediate=False because Δ legitimately takes negative values
        (washout) and values outside [0, 1] — we only clip the final image.
        """
        if device is None:
            device = next(model.parameters()).device

        target_image, conditions = self._get_target_and_conditions(batch)
        nc_t1 = batch[NC_T1_KEY]
        mask  = batch[MASK_KEY]
        sdf   = batch.get(SDF_KEY, None)

        cond_dwt, dwt_list = self._build_cond_dwt(conditions, mask, sdf=sdf)

        # Source = zeros with the same shape apply_dwt produces from NC_T1
        delta_dwt_shape = apply_dwt(nc_t1).shape
        source_dwt = th.zeros(delta_dwt_shape, device=device)

        if num_steps == 1:
            pred_delta_dwt = self.direct_prediction(
                model, cond=cond_dwt, source=source_dwt, device=device,
            )
        else:
            pred_delta_dwt = self.p_sample_loop(
                model=model, shape=source_dwt.shape,
                cond=cond_dwt, source=source_dwt,
                num_steps=num_steps, device=device,
                progress=progress, clip_denoised=clip_intermediate,
                solver=solver,
            )

        pred_delta = apply_idwt(pred_delta_dwt)
        pred_ce_t1 = (nc_t1 + pred_delta).clamp(0., 1.)
        # Preserve zero-padded slices (consistent with sibling WFM variants)
        pred_ce_t1[nc_t1 == 0] = 0

        return pred_ce_t1, target_image

    def sample_for_evaluation(self, model, batch, num_steps=50, device=None,
                              progress=True, clip_denoised=True):
        """Sampling API used by sample_wfm_prostate_run07.py."""
        return self.predict_ce_t1(
            model, batch, device=device,
            num_steps=num_steps, solver='heun',
            progress=progress, clip_intermediate=False,
        )

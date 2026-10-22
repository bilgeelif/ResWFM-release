"""
wfm_prostate_mask.py  —  Run 03

Extends WaveletFlowMatchingProstate with two changes:

1. Mask conditioning
   The T2 prostate mask (labels 0=BG, 1=TZ, 2=PZ) is DWT-ed and appended
   to the condition tensor.  This brings the condition from 32 → 40 channels
   and in_channels from 40 → 48.  The velocity network receives explicit zone
   information at every spatial location.

2. Zone-weighted flow-matching loss
   Prostate voxels receive higher gradient weight:
     background  weight = 1.0  (default)
     TZ          weight = tz_weight (default 5.0)
     PZ          weight = pz_weight (default 8.0, higher: ~70 % of cancers)
   The weight map is DWT-ed so that zone emphasis is applied consistently in
   wavelet space, where the velocity loss is computed.

Everything else (informed prior, ODE solvers, sampling) is identical to
wfm_prostate.py.  in_channels must be set to 48 in the model config.
"""

import torch
import torch as th
import numpy as np

from .nn import mean_flat
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

dwt  = DWT_3D('haar')
idwt = IDWT_3D('haar')

PROSTATE_TARGET     = 'ce_t1'
PROSTATE_CONDITIONS = ['t2', 'adc', 'dwi', 'nc_t1']   # same as before
MASK_KEY            = 'mask'
SDF_KEY             = 'sdf'    # signed distance field: −1 outside, +1 inside WG


def apply_dwt(img):
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(img)
    return th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)


def apply_idwt(x):
    B, _, H, W, D = x.size()
    return idwt(
        x[:, 0].view(B, 1, H, W, D) * 3.,
        x[:, 1].view(B, 1, H, W, D),
        x[:, 2].view(B, 1, H, W, D),
        x[:, 3].view(B, 1, H, W, D),
        x[:, 4].view(B, 1, H, W, D),
        x[:, 5].view(B, 1, H, W, D),
        x[:, 6].view(B, 1, H, W, D),
        x[:, 7].view(B, 1, H, W, D),
    )


class WaveletFlowMatchingProstateMask:
    """
    WFM for prostate CE_T1 synthesis with mask conditioning and weighted loss.

    Key architectural change vs. wfm_prostate.py:
        cond = DWT(T2) + DWT(ADC) + DWT(DWI) + DWT(NC_T1) + DWT(mask)
             = 4×8 + 8 = 40 condition channels
        in_channels for UNet = 8 (noisy CE_T1) + 40 (cond) = 48

    Loss:
        Weighted MSE in wavelet space.
        Weight map = zone_weight(mask) → DWT → applied per-voxel.
    """

    def __init__(self, num_timesteps=1000, sigma_max=0.5,
                 tz_weight=5.0, pz_weight=8.0,
                 roi_normalize_mse_wav=False):
        self.num_timesteps = num_timesteps
        self.sigma_max     = sigma_max
        self.tz_weight     = tz_weight
        self.pz_weight     = pz_weight
        # Opt-in, default False (original plain-mean behavior unchanged) --
        # see wfm_prostate_run07.py's training_losses() for what this gates.
        # Kept as a flag rather than editing the shared mse_wav computation
        # unconditionally, so the frozen 64x64x32 legacy lineage
        # (run_prostate_run10_w_part2.sh) is completely unaffected unless it
        # explicitly opts in -- it currently doesn't.
        self.roi_normalize_mse_wav = roi_normalize_mse_wav

        t = np.linspace(0, 1, num_timesteps)
        self.sigmas = sigma_max * np.sqrt(t * (1 - t))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def get_sigma(self, t):
        return self.sigma_max * torch.sqrt(t * (1 - t) + 1e-8)

    def _scale_timesteps(self, t):
        return t * self.num_timesteps

    def _get_target_and_conditions(self, x_start):
        target     = x_start[PROSTATE_TARGET]
        conditions = [x_start[m] for m in PROSTATE_CONDITIONS]
        return target, conditions

    def _build_cond_dwt(self, conditions, mask, sdf=None):
        """
        DWT all 4 sequences + spatial prior, concatenate → [B, 40, H/2, W/2, D/2].

        If sdf is provided (preferred), DWT the SDF instead of the raw binary
        mask.  The SDF is a smooth continuous field (−1 outside, +1 inside WG),
        so its Haar DWT detail subbands encode the *actual spatial gradient* near
        the prostate capsule rather than binary step-function ringing artifacts.
        Falls back to raw mask if SDF is not in the batch (backward compatible).
        """
        dwt_list = [apply_dwt(c) for c in conditions]
        spatial_prior = sdf if sdf is not None else mask
        prior_dwt = apply_dwt(spatial_prior)
        cond_dwt = th.cat(dwt_list + [prior_dwt], dim=1)   # 32 + 8 = 40 ch
        return cond_dwt, dwt_list

    def _build_zone_weight_wav(self, mask):
        """
        Build a per-channel zone weight map at wavelet spatial resolution.

        Uses avg_pool3d (not DWT) to downsample so all weights remain positive.

        Channel 0  (LLL — low-frequency approximation): zone weights only.
        Channels 1-7 (detail subbands — edges/texture): zone weights × hf_boost
            inside the prostate.  This directly incentivises the model to
            reproduce fine structural detail inside the ROI, not just intensity.
        """
        import torch.nn.functional as F

        w = th.ones_like(mask)                                          # BG = 1.0
        w = th.where(mask == 1, th.full_like(w, self.tz_weight), w)    # TZ
        w = th.where(mask == 2, th.full_like(w, self.pz_weight), w)    # PZ

        # Downsample → wavelet spatial resolution [B, 1, X/2, Y/2, Z/2]
        w_wav = F.avg_pool3d(w, kernel_size=2, stride=2)

        # LLL channel: zone weights only (contrast learning unchanged)
        lll_w = w_wav                                                    # [B, 1, ...]

        # Detail channels: boost weight inside prostate (where w_wav > 1)
        # hf_boost=2.0 means detail subbands inside prostate get 2× more weight
        # than the LLL subband → model must reproduce edges/texture, not just intensity
        hf_boost    = 2.0
        inside_roi  = (w_wav > 1.0).float()                             # 1 inside prostate
        detail_w    = w_wav * (1.0 + (hf_boost - 1.0) * inside_roi)    # [B, 1, ...]
        detail_w_7  = detail_w.expand(-1, 7, -1, -1, -1)               # 7 detail channels

        return th.cat([lll_w, detail_w_7], dim=1)                       # [B, 8, ...]

    @staticmethod
    def _gradient_loss_roi(pred, target, mask):
        """
        L1 gradient magnitude agreement inside the prostate mask.

        Computes first-order finite differences along all three spatial axes,
        masks them to the WG region, and returns the mean absolute difference
        in gradient magnitude.  This directly penalises blurry boundaries
        without any architectural change.
        """
        import torch.nn.functional as F

        wg = (mask > 0).float()   # [B, 1, X, Y, Z]

        def masked_grad_l1(p, t, m):
            diff = (p - t).abs()
            return (m * diff).sum() / (m.sum() + 1e-8)

        # X-axis gradients (slice off last voxel to match diff size)
        gx = masked_grad_l1(
            pred[:, :, 1:] - pred[:, :, :-1],
            target[:, :, 1:] - target[:, :, :-1],
            wg[:, :, 1:],
        )
        # Y-axis gradients
        gy = masked_grad_l1(
            pred[:, :, :, 1:] - pred[:, :, :, :-1],
            target[:, :, :, 1:] - target[:, :, :, :-1],
            wg[:, :, :, 1:],
        )
        # Z-axis gradients (through-plane)
        gz = masked_grad_l1(
            pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1],
            target[:, :, :, :, 1:] - target[:, :, :, :, :-1],
            wg[:, :, :, :, 1:],
        )
        return (gx + gy + gz) / 3.0

    def q_sample(self, x_source, x_target, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_target)
        t_b     = t.view(-1, 1, 1, 1, 1)
        sigma_t = self.get_sigma(t).view(-1, 1, 1, 1, 1)
        return (1 - t_b) * x_source + t_b * x_target + sigma_t * noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Zone-weighted flow matching loss:
            L = E[ ||f_θ(x_t, cond, t) - (target - source)||² * zone_weight ]
        """
        if model_kwargs is None:
            model_kwargs = {}

        target, conditions = self._get_target_and_conditions(x_start)
        mask               = x_start[MASK_KEY]          # [B, 1, X, Y, Z]
        sdf                = x_start.get(SDF_KEY, None) # [B, 1, X, Y, Z] or None

        cond_dwt, dwt_list = self._build_cond_dwt(conditions, mask, sdf=sdf)
        zone_weight_dwt    = self._build_zone_weight_wav(mask)

        target_dwt = apply_dwt(target)
        source_dwt = sum(dwt_list) / len(dwt_list)       # mean of 4 seqs (no mask in prior)

        t_continuous = t.float() / self.num_timesteps

        if noise is None:
            noise = th.randn_like(target)
        noise_dwt = apply_dwt(noise)

        x_t      = self.q_sample(source_dwt, target_dwt, t_continuous, noise=noise_dwt)
        x_t_cond = th.cat([x_t, cond_dwt], dim=1)         # [B, 48, H/2, W/2, D/2]

        t_scaled     = self._scale_timesteps(t_continuous)
        model_output = model(x_t_cond, t_scaled, **model_kwargs)

        velocity = target_dwt - source_dwt

        # Zone-weighted MSE: each wavelet channel weighted by the DWT zone map
        squared_error = (velocity - model_output) ** 2    # [B, 8, H/2, W/2, D/2]
        weighted_error = squared_error * zone_weight_dwt   # element-wise

        # Per-channel loss returned to training script for backward().
        # grad_loss is returned as a separate scalar so the training loop can
        # log it independently and weight it (currently 0.1 × grad_loss).
        mse_per_ch = weighted_error.mean(dim=[0, 2, 3, 4])  # [8]

        pred_target       = source_dwt + model_output
        model_output_idwt = apply_idwt(pred_target)

        grad_loss = self._gradient_loss_roi(model_output_idwt, target, mask)

        # Full-resolution image-space ROI losses.
        # The wavelet MSE above grades 2×2×2 block summaries, which hides
        # sub-block detail inside the prostate. These L1 terms operate on the
        # IDWT'd prediction at full 160×160×32 resolution, giving the network
        # a direct gradient path for "which voxel" not just "which block".
        import torch.nn.functional as F
        wg   = (mask > 0).float()
        n_wg = wg.sum() + 1e-8
        img_l1_roi = ((model_output_idwt - target).abs() * wg).sum() / n_wg

        # 1-voxel-thick capsule ring (dilated WG minus WG itself) — targets
        # the boundary radiologists evaluate first.
        band = F.max_pool3d(wg, kernel_size=3, stride=1, padding=1) - wg
        n_b  = band.sum() + 1e-8
        edge_l1 = ((model_output_idwt - target).abs() * band).sum() / n_b

        # PZ-only L1 — PZ holds ~70 % of clinically significant cancers.
        pz   = (mask == 2).float()
        n_pz = pz.sum() + 1e-8
        pz_l1 = ((model_output_idwt - target).abs() * pz).sum() / n_pz

        return {
            "mse_wav":    mse_per_ch,
            "grad_loss":  grad_loss,
            "img_l1_roi": img_l1_roi,
            "edge_l1":    edge_l1,
            "pz_l1":      pz_l1,
        }, model_output, model_output_idwt

    # ------------------------------------------------------------------
    # Sampling  (identical to wfm_prostate.py — mask included in cond)
    # ------------------------------------------------------------------

    def p_sample_step_euler(self, model, x_t, t_curr, t_next, cond, source,
                            model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        B      = x_t.shape[0]
        device = x_t.device
        t_model  = torch.full((B,), t_curr * self.num_timesteps, device=device)
        x_t_cond = torch.cat([x_t, cond], dim=1)
        with torch.no_grad():
            velocity = model(x_t_cond, t_model, **model_kwargs)
        return x_t + (t_next - t_curr) * velocity

    def p_sample_step_heun(self, model, x_t, t_curr, t_next, cond, source,
                           model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        B      = x_t.shape[0]
        device = x_t.device
        dt     = t_next - t_curr
        t_curr_m = torch.full((B,), t_curr * self.num_timesteps, device=device)
        with torch.no_grad():
            v1 = model(torch.cat([x_t, cond], dim=1), t_curr_m, **model_kwargs)
        x_pred   = x_t + dt * v1
        t_next_m = torch.full((B,), t_next * self.num_timesteps, device=device)
        with torch.no_grad():
            v2 = model(torch.cat([x_pred, cond], dim=1), t_next_m, **model_kwargs)
        return x_t + dt * 0.5 * (v1 + v2)

    def p_sample_step(self, model, x_t, t_curr, t_next, cond, source,
                      model_kwargs=None, solver='euler'):
        if solver == 'heun':
            return self.p_sample_step_heun(model, x_t, t_curr, t_next, cond,
                                           source, model_kwargs)
        return self.p_sample_step_euler(model, x_t, t_curr, t_next, cond,
                                        source, model_kwargs)

    def direct_prediction(self, model, cond, source, device=None,
                          model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device
        B       = source.shape[0]
        t_model = torch.full((B,), self.num_timesteps, device=device,
                             dtype=torch.float32)
        with torch.no_grad():
            velocity = model(torch.cat([source, cond], dim=1), t_model,
                             **model_kwargs)
        return source + velocity

    def p_sample_loop(self, model, shape, cond, source, num_steps=50,
                      device=None, progress=True, clip_denoised=True,
                      model_kwargs=None, solver='euler'):
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device
        if solver == 'direct' or num_steps == 1:
            out = self.direct_prediction(model, cond, source, device, model_kwargs)
            return self._clip_wavelets(out) if clip_denoised else out
        x_t       = source.clone()
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
        iterator  = range(num_steps)
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator,
                            desc=f"WFM {solver.upper()} ({num_steps} steps)")
        for i in iterator:
            x_t = self.p_sample_step(
                model, x_t,
                t_curr=timesteps[i].item(), t_next=timesteps[i + 1].item(),
                cond=cond, source=source,
                model_kwargs=model_kwargs, solver=solver,
            )
        return self._clip_wavelets(x_t) if clip_denoised else x_t

    def sample_for_evaluation(self, model, batch, num_steps=50, device=None,
                               progress=True, clip_denoised=True):
        """Sample CE_T1 — mask is loaded from batch (available at inference)."""
        if device is None:
            device = next(model.parameters()).device

        target, conditions         = self._get_target_and_conditions(batch)
        mask                       = batch[MASK_KEY]
        sdf                        = batch.get(SDF_KEY, None)
        cond_dwt, dwt_list         = self._build_cond_dwt(conditions, mask, sdf=sdf)
        source_dwt                 = sum(dwt_list) / len(dwt_list)

        output_dwt = self.p_sample_loop(
            model=model, shape=source_dwt.shape,
            cond=cond_dwt, source=source_dwt,
            num_steps=num_steps, device=device,
            progress=progress, clip_denoised=clip_denoised,
        )
        output_img = apply_idwt(output_dwt).clamp(0., 1.)
        output_img[batch['nc_t1'] == 0] = 0
        return output_img, target

    def _clip_wavelets(self, x):
        return apply_dwt(apply_idwt(x).clamp(0., 1.))

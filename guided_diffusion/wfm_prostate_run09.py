"""
wfm_prostate_run09.py  —  Run 09: residual WFM + mask-conditional attention.

Extends Run 07's residual WFM (target = Δ = CE_T1 − NC_T1, source = 0)
by threading the prostate mask through to the model's forward pass so the
new mask-attention gates (in WavUNetModelRun09) can use it at each
encoder level.

Everything else is identical to Run 07 — same residual reframing, same loss
terms (wavelet MSE + image-space L1 + edge L1 + PZ L1), same flow source
(zeros), same reconstruction `pred_CE_T1 = NC_T1 + Δ̂`.

The only behavioural change is: at every model forward, mask is passed as
an extra kwarg.  Models that don't use it (vanilla WavUNetModel) will ignore
it via **_; models that do (WavUNetModelRun09) gate features by it.
"""

import torch as th

from .wfm_prostate_run07 import WaveletFlowMatchingProstateRun07
from .wfm_prostate_mask  import apply_dwt, MASK_KEY


class WaveletFlowMatchingProstateRun09(WaveletFlowMatchingProstateRun07):

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs = {**model_kwargs, 'mask': x_start[MASK_KEY]}
        return super().training_losses(model, x_start, t,
                                       model_kwargs=model_kwargs, noise=noise)

    # ------------------------------------------------------------------
    # Inference  (override predict_ce_t1 to thread mask through sampling)
    # ------------------------------------------------------------------
    def predict_ce_t1(self, model, batch, device=None, num_steps=1,
                      solver='heun', progress=False, clip_intermediate=False,
                      use_mask=True):
        if device is None:
            device = next(model.parameters()).device

        target_image, conditions = self._get_target_and_conditions(batch)
        nc_t1 = batch['nc_t1']
        mask  = batch[MASK_KEY]
        sdf   = batch.get('sdf', None)

        # No-mask inference (use_mask=False): present the exact regime the
        # mask-dropout model was trained for — null SDF prior + FiLM identity.
        # A non-mask-dropout model will degrade here (it never saw this state).
        if not use_mask:
            sdf  = th.zeros_like(sdf) if sdf is not None else th.zeros_like(nc_t1)
            mask = None

        cond_dwt, _ = self._build_cond_dwt(conditions, mask, sdf=sdf)

        delta_dwt_shape = apply_dwt(nc_t1).shape
        source_dwt = th.zeros(delta_dwt_shape, device=device)

        model_kwargs = {'mask': mask}

        if num_steps == 1:
            pred_delta_dwt = self.direct_prediction(
                model, cond=cond_dwt, source=source_dwt,
                device=device, model_kwargs=model_kwargs,
            )
        else:
            pred_delta_dwt = self.p_sample_loop(
                model=model, shape=source_dwt.shape,
                cond=cond_dwt, source=source_dwt,
                num_steps=num_steps, device=device,
                progress=progress, clip_denoised=clip_intermediate,
                solver=solver, model_kwargs=model_kwargs,
            )

        from .wfm_prostate_mask import apply_idwt
        pred_delta = apply_idwt(pred_delta_dwt)
        pred_ce_t1 = (nc_t1 + pred_delta).clamp(0., 1.)
        pred_ce_t1[nc_t1 == 0] = 0
        return pred_ce_t1, target_image

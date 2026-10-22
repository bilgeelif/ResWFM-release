"""
wfm_prostate_run10.py  —  Run 10: residual WFM + zone-FiLM + frequency-graded
                          subband-weighted loss, trained on prostate ROI crops.

Inherits the residual reframing (target = Δ = CE_T1 − NC_T1, source = 0) and
mask-threading from Run 09 (mask is passed to the model forward so the
zone-FiLM gates can use it).  The only methodological change at the loss level
is the wavelet zone-weight map.

Frequency-graded subband weighting
-----------------------------------
Run 03/07 boosted ALL detail subbands by a flat factor (hf_boost=2) inside the
prostate.  The L2/averaging bias that blurs synthesis is strongest at the
highest spatial frequencies, so a flat boost under-emphasises exactly the bands
that need help.  Run 10 grades the boost by the number of high-pass axes of
each Haar subband:

    subband (apply_dwt channel order):  LLH LHL LHH HLL HLH HHL HHH
    #high-pass axes (nH):                1   1   2   1   2   2   3
    inside-prostate boost:  hf_boost * (1 + hf_freq_gamma * (nH - 1))

With hf_boost=2.0, hf_freq_gamma=0.5 the inside-prostate detail weights are
≈ 2× (1H), 3× (2H), 4× (3H).  PZ vs TZ emphasis is unchanged (still driven by
pz_weight/tz_weight via the base zone map).

NOTE: subband reweighting concentrates loss energy in the right bands but each
wavelet coefficient still summarises a 2×2×2 block — it is complementary to the
full-resolution image-space L1/edge/gradient terms inherited from Run 07, not a
replacement for them.
"""

import torch as th
import torch.nn.functional as F

from .wfm_prostate_run09 import WaveletFlowMatchingProstateRun09


class WaveletFlowMatchingProstateRun10(WaveletFlowMatchingProstateRun09):

    def __init__(self, num_timesteps=1000, sigma_max=0.5,
                 tz_weight=5.0, pz_weight=8.0,
                 hf_boost=2.0, hf_freq_gamma=0.5,
                 roi_normalize_mse_wav=False):
        super().__init__(num_timesteps=num_timesteps, sigma_max=sigma_max,
                         tz_weight=tz_weight, pz_weight=pz_weight,
                         roi_normalize_mse_wav=roi_normalize_mse_wav)
        self.hf_boost      = float(hf_boost)
        self.hf_freq_gamma = float(hf_freq_gamma)

    def _build_zone_weight_wav(self, mask):
        """
        Per-channel zone weight map at wavelet resolution, [B, 8, X/2, Y/2, Z/2].

        Channel 0 (LLL): zone weights only (contrast learning unchanged).
        Channels 1-7 (detail): zone weights, plus a frequency-graded boost
            applied only inside the prostate (where the pooled zone weight > 1).
        """
        w = th.ones_like(mask)                                       # BG = 1.0
        w = th.where(mask == 1, th.full_like(w, self.tz_weight), w)  # TZ
        w = th.where(mask == 2, th.full_like(w, self.pz_weight), w)  # PZ

        # Downsample to wavelet spatial resolution (avg_pool keeps weights ≥ 0).
        w_wav  = F.avg_pool3d(w, kernel_size=2, stride=2)            # [B, 1, ...]
        inside = (w_wav > 1.0).float()                              # inside prostate

        lll_w = w_wav

        # #high-pass axes per detail subband, in apply_dwt() channel order:
        #   LLH, LHL, LHH, HLL, HLH, HHL, HHH
        n_hp = th.tensor([1, 1, 2, 1, 2, 2, 3],
                         device=mask.device, dtype=w_wav.dtype)
        band_boost = self.hf_boost * (1.0 + self.hf_freq_gamma * (n_hp - 1.0))  # [7]

        detail_channels = [
            w_wav * (1.0 + (band_boost[b] - 1.0) * inside)          # [B, 1, ...]
            for b in range(7)
        ]
        detail_w = th.cat(detail_channels, dim=1)                   # [B, 7, ...]

        return th.cat([lll_w, detail_w], dim=1)                     # [B, 8, ...]

"""
zone_film.py  —  Zone-conditional FiLM modulation for run_10.

Spatially-varying Feature-wise Linear Modulation (FiLM) keyed by anatomical
zone.  Each zone (BG=0, TZ=1, PZ=2) gets its own learned per-channel scale
(gamma) and shift (beta), applied as:

    feat = feat * (1 + alpha*gamma_zone) + alpha*beta_zone

Because the zone label is one-hot encoded and fed through a 1x1 conv, the
conv literally stores one (gamma, beta) vector per zone — i.e. the network
can weight feature channels differently inside PZ vs TZ vs background, which
mirrors how a radiologist leans on different sequences per zone (DWI/ADC in
PZ, T2 in TZ).

The modulation is the data-fusion mechanism: conditioning sequences enter as
input channels, and these gates re-route the resulting encoder features by
zone at every spatial scale.

Init is near-identity (conv weights and bias zeroed) so gamma=beta=0 at
step 0 → the gated model starts numerically equal to the ungated baseline,
and learns zone modulation only as training proceeds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZoneFiLM(nn.Module):
    """
    Zone-conditional, spatially-varying FiLM gate.

    Args:
        feat_channels : channels of the feature map to modulate
        num_zones     : number of discrete zone labels (default 3: BG/TZ/PZ)
        alpha         : modulation strength. alpha=0 → identity;
                        larger → stronger learned gamma/beta.
    """

    def __init__(self, feat_channels: int, num_zones: int = 3,
                 alpha: float = 1.0):
        super().__init__()
        self.num_zones     = int(num_zones)
        self.feat_channels = int(feat_channels)
        self.alpha         = float(alpha)

        # One-hot zone (num_zones) -> [gamma; beta]  (2*C channels), 1x1 conv.
        self.to_film = nn.Conv3d(self.num_zones, 2 * feat_channels,
                                 kernel_size=1, bias=True)

        # Near-identity init: gamma = beta = 0 at start.
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def forward(self, feat: torch.Tensor, zone: torch.Tensor) -> torch.Tensor:
        """
        feat : [B, C, X, Y, Z]
        zone : [B, 1, X, Y, Z] integer zone labels (already at feat's spatial
               resolution).  Values outside [0, num_zones-1] are clamped.
        returns: [B, C, X, Y, Z]
        """
        z = zone.long().clamp(0, self.num_zones - 1)[:, 0]      # [B, X, Y, Z]
        onehot = F.one_hot(z, num_classes=self.num_zones)        # [B, X, Y, Z, n]
        onehot = onehot.permute(0, 4, 1, 2, 3).to(feat.dtype)    # [B, n, X, Y, Z]

        gamma, beta = torch.chunk(self.to_film(onehot), 2, dim=1)
        return feat * (1.0 + self.alpha * gamma) + self.alpha * beta

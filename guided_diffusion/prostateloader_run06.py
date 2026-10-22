"""
prostateloader_run06.py  —  Run 06 dataloader

Same as ProstateVolumesMask (Run 05) but with training-time augmentation:

  - L/R flip (X axis) with 50 % probability — the prostate is approximately
    bilaterally symmetric, so flipping is anatomically valid and roughly
    doubles the effective dataset diversity.

  - Intensity jitter (×Uniform[0.9, 1.1]) applied independently to each
    conditioning modality (T2, ADC, DWI, NC_T1).  Simulates acquisition
    variability and forces the model to be robust to intensity scaling.

  - Target (CE_T1) and the mask are NEVER jittered (target is the supervision
    signal; mask must stay consistent with the SDF).

Mask, SDF, all sequences are flipped together to preserve spatial alignment.
"""

import numpy as np
import torch

from .prostateloader_mask import ProstateVolumesMask, PROSTATE_SEQTYPES_MASK


class ProstateVolumesRun06(ProstateVolumesMask):

    def __getitem__(self, x):
        sample = super().__getitem__(x)

        if self.mode == "train":
            # 50 % L/R flip — flip every spatial tensor on the X axis.
            # Tensor shape is [C, X, Y, Z], so X is dim=-3.
            if np.random.random() < 0.5:
                for key in PROSTATE_SEQTYPES_MASK + ["sdf"]:
                    if key in sample and isinstance(sample[key], torch.Tensor):
                        sample[key] = sample[key].flip(dims=[-3])

            # Intensity jitter on conditioning modalities only.
            # CE_T1 (target) and mask are untouched.
            for key in ["t2", "adc", "dwi", "nc_t1"]:
                if key in sample:
                    jitter = float(np.random.uniform(0.9, 1.1))
                    sample[key] = (sample[key] * jitter).clamp(0., 1.)

        return sample

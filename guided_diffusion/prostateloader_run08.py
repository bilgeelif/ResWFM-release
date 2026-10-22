"""
prostateloader_run08.py  —  Run 08 dataloader (Stage 2 of two-stage training)

Loads 96×96×32 prostate-centered ROI patches produced by
crop_bbox_roi_from_preprocessed.py.

Differences from ProstateVolumesMask (run_05 / run_07):
  - mask is INSIDE each subject directory as "mask.nii.gz" (not in a flat
    mask_dir alongside the sequences)
  - all volumes are 96×96×32 (already pre-cropped + zero-padded)
  - no mask_dir argument; just a single data_dir

Augmentation (train mode only, identical to ProstateVolumesRun06):
  - 50 % L/R flip across the X axis (prostate is approximately bilaterally
    symmetric — anatomically valid, doubles effective dataset diversity)
  - Intensity jitter ×Uniform[0.9, 1.1] on conditioning modalities only
    (T2, ADC, DWI, NC_T1). Target (CE_T1) and mask never jittered.

Expected layout:
    data_dir/NUPT_XXX/T2.nii.gz, ADC.nii.gz, DWI.nii.gz, NC_T1.nii.gz,
                      CE_T1.nii.gz, mask.nii.gz
"""

import os
import torch
import torch.utils.data
import numpy as np
import nibabel
from scipy.ndimage import distance_transform_edt

PROSTATE_SEQTYPES_RUN08 = ["t2", "adc", "dwi", "nc_t1", "ce_t1", "mask"]

_SEQ_DISK_TO_KEY = {
    "T2.nii.gz":    "t2",
    "ADC.nii.gz":   "adc",
    "DWI.nii.gz":   "dwi",
    "NC_T1.nii.gz": "nc_t1",
    "CE_T1.nii.gz": "ce_t1",
    "mask.nii.gz":  "mask",
}


class ProstateVolumesRun08(torch.utils.data.Dataset):

    def __init__(self, directory: str, mode: str = "train",
                 manifest_file: str = None):
        super().__init__()
        self.mode      = mode
        self.directory = os.path.expanduser(directory)
        self.database  = []

        if manifest_file is not None and os.path.isfile(manifest_file):
            with open(manifest_file) as f:
                subject_ids = [line.strip() for line in f if line.strip()]
        else:
            subject_ids = sorted(os.listdir(self.directory))

        n_no_mask     = 0
        n_no_sequence = 0
        for sid in subject_ids:
            subj_dir = os.path.join(self.directory, sid)
            if not os.path.isdir(subj_dir):
                continue

            datapoint = {"subj": sid}
            complete  = True
            for disk_name, key in _SEQ_DISK_TO_KEY.items():
                fpath = os.path.join(subj_dir, disk_name)
                if not os.path.isfile(fpath):
                    complete = False
                    if key == "mask":
                        n_no_mask += 1
                    else:
                        n_no_sequence += 1
                    break
                datapoint[key] = fpath

            if complete:
                self.database.append(datapoint)

        print(f"[ProstateVolumesRun08] mode={mode} | "
              f"subjects: {len(self.database)} "
              f"(missing mask: {n_no_mask}, missing sequence: {n_no_sequence})")

    def __len__(self) -> int:
        return len(self.database)

    def __getitem__(self, x: int) -> dict:
        d = self.database[x]

        sample = {}
        for key in PROSTATE_SEQTYPES_RUN08:
            arr = nibabel.load(d[key]).get_fdata(dtype=np.float32)
            sample[key] = torch.from_numpy(arr).unsqueeze(0)   # [1, X, Y, Z]

        # Soft spatial prior from binary mask.
        # SDF: −1 outside WG, +1 inside, smooth across the boundary.
        # Replaces raw binary mask in the DWT conditioning channel to avoid
        # Gibbs ringing in the wavelet detail subbands at zone boundaries.
        mask_np   = sample["mask"][0].numpy()
        wg_binary = (mask_np > 0).astype(np.float32)
        if wg_binary.sum() > 0:
            r     = max(distance_transform_edt(wg_binary).max(), 1.0)
            inner = distance_transform_edt(wg_binary)     / r
            outer = distance_transform_edt(1 - wg_binary) / r
            sdf   = np.clip(inner - outer, -1.0, 1.0)
        else:
            sdf = np.zeros_like(mask_np, dtype=np.float32)
        sample["sdf"] = torch.from_numpy(sdf.astype(np.float32)).unsqueeze(0)

        # Augmentation — train mode only.
        if self.mode == "train":
            # 50 % L/R flip. Tensor shape is [C, X, Y, Z], so X is dim=-3.
            if np.random.random() < 0.5:
                for key in PROSTATE_SEQTYPES_RUN08 + ["sdf"]:
                    if key in sample and isinstance(sample[key], torch.Tensor):
                        sample[key] = sample[key].flip(dims=[-3])

            # Intensity jitter on conditioning modalities only.
            # CE_T1 (target) and mask never jittered.
            for key in ["t2", "adc", "dwi", "nc_t1"]:
                if key in sample:
                    jitter = float(np.random.uniform(0.9, 1.1))
                    sample[key] = (sample[key] * jitter).clamp(0., 1.)

        sample["missing"] = "none"
        sample["subj"] = d["subj"] if self.mode in ("eval", "auto") else "dummy_string"
        return sample

"""
prostateloader_mask.py  —  Run 03 dataloader

Extends ProstateVolumes to also load prostate zone masks (label 1=TZ, 2=PZ).

Masks are stored in a separate flat directory as {subj_id}_mask.nii.gz,
produced by prepare_prostate_mask.py.  This avoids re-uploading the full
3 GB dataset to the VM — only the small mask directory (~250 MB) is uploaded.

Args:
    directory    : root with NUPT_XXX/ containing T2/ADC/DWI/NC_T1/CE_T1
    mask_dir     : flat directory containing NUPT_XXX_mask.nii.gz files
    manifest_file: optional subject list

The mask tensor has values {0, 1, 2} (float32) and shape [1, X, Y, Z].
It is DWT-ed in wfm_prostate_mask.py to give 8 extra conditioning channels,
bringing in_channels from 40 → 48.
"""

import os
import torch
import torch.utils.data
import numpy as np
import nibabel
from scipy.ndimage import distance_transform_edt

PROSTATE_SEQTYPES_MASK = ["t2", "adc", "dwi", "nc_t1", "ce_t1", "mask"]

_SEQ_DISK_TO_KEY = {
    "T2.nii.gz":    "t2",
    "ADC.nii.gz":   "adc",
    "DWI.nii.gz":   "dwi",
    "NC_T1.nii.gz": "nc_t1",
    "CE_T1.nii.gz": "ce_t1",
}


class ProstateVolumesMask(torch.utils.data.Dataset):
    """
    Prostate dataset with separate flat mask directory.

    Mask filename convention: {subj_id}_mask.nii.gz  (all in one flat dir).
    """

    def __init__(self, directory: str, mask_dir: str,
                 mode: str = "train", manifest_file: str = None):
        super().__init__()
        self.mode      = mode
        self.directory = os.path.expanduser(directory)
        self.mask_dir  = os.path.expanduser(mask_dir)
        self.database  = []

        if manifest_file is not None and os.path.isfile(manifest_file):
            with open(manifest_file) as f:
                subject_ids = [line.strip() for line in f if line.strip()]
        else:
            subject_ids = sorted(os.listdir(self.directory))

        n_no_mask = 0
        for sid in subject_ids:
            subj_dir = os.path.join(self.directory, sid)
            if not os.path.isdir(subj_dir):
                continue

            datapoint = {"subj": sid}
            complete  = True

            # Sequences from per-subject folder
            for disk_name, key in _SEQ_DISK_TO_KEY.items():
                fpath = os.path.join(subj_dir, disk_name)
                if not os.path.isfile(fpath):
                    complete = False
                    break
                datapoint[key] = fpath

            # Mask from flat mask_dir: {sid}_mask.nii.gz
            mask_path = os.path.join(self.mask_dir, f"{sid}_mask.nii.gz")
            if not os.path.isfile(mask_path):
                complete = False
                n_no_mask += 1
            else:
                datapoint["mask"] = mask_path

            if complete:
                self.database.append(datapoint)

        print(f"[ProstateVolumesMask] mode={mode} | "
              f"subjects found: {len(self.database)} "
              f"(excluded {n_no_mask} without mask)")

    def __len__(self) -> int:
        return len(self.database)

    def __getitem__(self, x: int) -> dict:
        d = self.database[x]

        sample = {}
        for key in PROSTATE_SEQTYPES_MASK:
            arr = nibabel.load(d[key]).get_fdata(dtype=np.float32)
            sample[key] = torch.from_numpy(arr).unsqueeze(0)  # [1, X, Y, Z]

        # Precompute soft spatial prior from the binary mask.
        # SDF encodes continuous distance to the WG capsule boundary:
        #   negative (−1 to 0) inside the gland, positive (0 to +1) outside.
        # Replacing the raw binary mask with SDF eliminates Haar DWT ringing
        # artifacts at zone boundaries in the high-frequency conditioning subbands.
        mask_np = sample["mask"][0].numpy()          # [X, Y, Z], values {0,1,2}
        wg_binary = (mask_np > 0).astype(np.float32)
        r = max(distance_transform_edt(wg_binary).max(), 1.0)   # prostate radius (voxels)
        inner = distance_transform_edt(wg_binary)   / r          # 0→1 inside
        outer = distance_transform_edt(1 - wg_binary) / r        # 0→1 outside
        sdf   = np.clip(inner - outer, -1.0, 1.0)               # −1 outside, +1 inside
        sample["sdf"] = torch.from_numpy(sdf.astype(np.float32)).unsqueeze(0)  # [1, X, Y, Z]

        sample["missing"] = "none"
        sample["subj"] = d["subj"] if self.mode in ("eval", "auto") else "dummy_string"
        return sample

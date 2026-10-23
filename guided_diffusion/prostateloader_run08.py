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
                 manifest_file: str = None, require_target: bool = True):
        """
        require_target: if False, a missing CE_T1.nii.gz (the training
            target) no longer excludes a subject from the dataset — it is
            loaded opportunistically if present (so evaluation/comparison
            still works when available) but never required. Training always
            needs the target, so every training call site leaves this at
            its default (True); inference (sample_wfm_prostate_run10.py)
            passes False.

        `directory` accepts either layout, auto-detected by content:
          - a single subject's own folder (NC_T1.nii.gz etc. directly
            inside it) -> one-subject dataset, subject ID = the folder's
            basename.
          - a folder of subject subfolders (the original layout) -> every
            subfolder is a candidate subject, optionally filtered by
            `manifest_file` (one subject ID per line).
        No manifest or other dataset-specific file is required either way.
        """
        super().__init__()
        self.mode      = mode
        self.directory = os.path.expanduser(directory)
        self.database  = []
        self.subj_dirs = {}   # subject ID -> the directory its files live in

        n_no_sequence = 0

        if os.path.isfile(os.path.join(self.directory, "NC_T1.nii.gz")):
            # Single-subject layout: `directory` itself holds the files.
            sid = os.path.basename(self.directory.rstrip(os.sep)) or "subject"
            datapoint = self._collect_datapoint(self.directory, sid, require_target)
            if datapoint is not None:
                self.database.append(datapoint)
                self.subj_dirs[sid] = self.directory
            else:
                n_no_sequence += 1
        else:
            # Multi-subject layout: `directory` holds one subfolder per subject.
            if manifest_file is not None and os.path.isfile(manifest_file):
                with open(manifest_file) as f:
                    subject_ids = [line.strip() for line in f if line.strip()]
            else:
                subject_ids = sorted(os.listdir(self.directory))

            for sid in subject_ids:
                subj_dir = os.path.join(self.directory, sid)
                if not os.path.isdir(subj_dir):
                    continue
                datapoint = self._collect_datapoint(subj_dir, sid, require_target)
                if datapoint is not None:
                    self.database.append(datapoint)
                    self.subj_dirs[sid] = subj_dir
                else:
                    n_no_sequence += 1  # approximate; see _collect_datapoint

        print(f"[ProstateVolumesRun08] mode={mode} | "
              f"subjects: {len(self.database)} "
              f"(missing/incomplete: {n_no_sequence})")

    @staticmethod
    def _collect_datapoint(subj_dir: str, sid: str, require_target: bool):
        """Build one subject's file-path dict, or None if incomplete."""
        datapoint = {"subj": sid}
        for disk_name, key in _SEQ_DISK_TO_KEY.items():
            fpath = os.path.join(subj_dir, disk_name)
            if key == "ce_t1" and not require_target:
                if os.path.isfile(fpath):
                    datapoint[key] = fpath
                continue  # optional: never blocks completeness
            if not os.path.isfile(fpath):
                return None
            datapoint[key] = fpath
        return datapoint

    def __len__(self) -> int:
        return len(self.database)

    def __getitem__(self, x: int) -> dict:
        d = self.database[x]

        sample = {}
        for key in PROSTATE_SEQTYPES_RUN08:
            if key not in d:
                continue  # e.g. ce_t1 absent when require_target=False
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

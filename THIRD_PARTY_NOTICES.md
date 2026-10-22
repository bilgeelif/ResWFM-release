# Third-party notices

This repository is MIT-licensed (see `LICENSE`) **except** for the component
listed below, which carries a different license and is not relicensed by the
`LICENSE` file at the repo root.

## DWT_IDWT/ (both DWT_IDWT_Functions.py and DWT_IDWT_layer.py)

```
Copyright (c) 2019, Adobe Inc. All rights reserved.
Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International Public License (CC BY-NC-SA 4.0).
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
```

See `DWT_IDWT/LICENSE` — the whole `DWT_IDWT/` directory is scoped under
this license, not relicensed by the repo-root MIT `LICENSE`.
`DWT_IDWT_Functions.py` carries Adobe's notice directly and unmodified.
`DWT_IDWT_layer.py` is included in the same scope: it is a thin wrapper
around `DWT_IDWT_Functions.py`'s autograd `Function`s with no independent
existence apart from them, so it is treated as part of the same licensed
work rather than as separately-MIT code that merely imports it.
`guided_diffusion/unet.py` and `guided_diffusion/wfm_prostate_mask.py`
depend on this directory for the Haar wavelet transform at the core of the
method. Inherited via the upstream WFM repository, which does not
separately disclose this exception in its own blanket MIT `LICENSE` — this
notice (and `DWT_IDWT/LICENSE`) closes that gap for this repository.

**Practical consequence:** this component (and anything that would count as
a derivative of it) is restricted to **non-commercial** use and, if
redistributed in modified form, must stay under CC BY-NC-SA 4.0 — it cannot
be sublicensed as MIT. Academic/research use of this repository as a whole is
unaffected; downstream commercial use of the DWT/IDWT component specifically
is not permitted under its original license.

If a fully-permissive (MIT-only) release is needed later, this component can
be replaced with a clean-room 3D Haar DWT/IDWT implementation (a fixed-kernel
`conv3d`/`conv_transpose3d` pair is mathematically sufficient for Haar) that
does not reuse Adobe's code — not done in this pass.

## guided_diffusion/logger.py

```
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/master/baselines/logger.py
```

OpenAI baselines is MIT-licensed; compatible with this repo's MIT license.

## Everything else under guided_diffusion/

Vendored (and in the case of `script_util.py`, trimmed) from WFM
(https://github.com/yalcintur/WFM, MIT License, Tur/Stojkovic/Bagci, MIDL
2026), which itself builds on cWDM/wdm-3d (https://github.com/pfriedri/wdm-3d,
MIT License, Friedrich et al.). See `LICENSE` for the full MIT text.

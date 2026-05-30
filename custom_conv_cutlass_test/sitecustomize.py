"""Automatically route copied benchmark models through the custom Conv2d path.

Python imports this module automatically when this directory is on sys.path.
The custom_conv runner sets PYTHONPATH accordingly, so the original eval/profile
scripts can stay mostly unchanged while still using StandaloneCutlassConv2d.
"""

from __future__ import annotations

import os

_CUTLASS_DIR = "/nethome/yjia305/USERSCRATCH/cutlass_test"

os.environ.setdefault("USE_STANDALONE_CUTLASS_CONV", "1")
os.environ.setdefault("USE_TORCH_COMPILE", "0")
os.environ.setdefault("STANDALONE_CUTLASS_STRICT", "1")
os.environ.setdefault("CUTLASS_DIR", _CUTLASS_DIR)
os.environ.setdefault("STANDALONE_CUTLASS_DIR", _CUTLASS_DIR)
os.environ.setdefault(
    "STANDALONE_CUTLASS_BUILD_DIR",
    "/tmp/ml_bench_cutlass_test_standalone_conv_build",
)

try:
    import torch

    from cutlass_standalone_conv import replace_supported_conv2d_modules

    _ORIGINAL_TORCH_HUB_LOAD = torch.hub.load

    def _custom_conv_hub_load(*args, **kwargs):
        model = _ORIGINAL_TORCH_HUB_LOAD(*args, **kwargs)
        if os.environ.get("USE_STANDALONE_CUTLASS_CONV", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            model = replace_supported_conv2d_modules(model)
        return model

    if not getattr(torch.hub.load, "_custom_conv_wrapped", False):
        _custom_conv_hub_load._custom_conv_wrapped = True
        torch.hub.load = _custom_conv_hub_load
except Exception:
    # Keep import side effects non-fatal; explicit strict mode in the extension
    # will still surface real custom-conv failures during model execution.
    pass

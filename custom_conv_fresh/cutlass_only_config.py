#!/usr/bin/env python3
"""Shared non-cuDNN CUTLASS-oriented setup for new_example_cutlass_modified."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_LOCAL_CUTLASS_DIR = Path(__file__).resolve().parents[4] / "cutlass_test"


def configured_cutlass_dir() -> Path | None:
    """Return the editable CUTLASS checkout these scripts should use, if any."""
    raw = os.environ.get("CUTLASS_DIR") or os.environ.get("LOCAL_CUTLASS_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    if DEFAULT_LOCAL_CUTLASS_DIR.exists():
        return DEFAULT_LOCAL_CUTLASS_DIR
    return None


def configured_backend_env(name: str, default: str) -> str:
    """Return a backend preference string, honoring caller-provided env overrides."""
    return os.environ.get(name, default)


def configure_env() -> None:
    """Set backend preferences before torch is imported."""
    os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "1")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS", "CUTLASS,ATEN")
    # Some small GEMMs (for example the classifier addmm) have no valid CUTLASS lowering.
    # Keep CUTLASS preferred, but allow an ATen fallback so torch.compile() succeeds.
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "CUTLASS,ATEN")


def configure_torch(torch, device) -> None:
    """Configure CUDA libraries after torch is imported."""
    if getattr(device, "type", str(device)) != "cuda":
        raise RuntimeError("new_example_cutlass_modified requires CUDA to run CUTLASS kernels.")

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return

    cutlass_dir = configured_cutlass_dir()
    if cutlass_dir is not None:
        if not cutlass_dir.exists():
            raise FileNotFoundError(f"CUTLASS_DIR does not exist: {cutlass_dir}")
        if not (cutlass_dir / "include").is_dir():
            raise FileNotFoundError(f"CUTLASS_DIR is missing include/: {cutlass_dir}")
        if not (cutlass_dir / "python").is_dir():
            raise FileNotFoundError(f"CUTLASS_DIR is missing python/: {cutlass_dir}")
        inductor_config.cuda.cutlass_dir = str(cutlass_dir)

    for name, value in (
        ("max_autotune", True),
        (
            "max_autotune_conv_backends",
            configured_backend_env("TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS", "CUTLASS,ATEN"),
        ),
        (
            "max_autotune_gemm_backends",
            configured_backend_env("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "CUTLASS,ATEN"),
        ),
    ):
        if hasattr(inductor_config, name):
            setattr(inductor_config, name, value)


def move_model_to_cutlass_layout(model, torch, device):
    """Move a plain nn.Module to CUDA channels-last layout."""
    if os.environ.get("USE_STANDALONE_CUTLASS_CONV", "0").strip().lower() in {"1", "true", "yes", "on"}:
        from cutlass_standalone_conv import replace_supported_conv2d_modules

        model = replace_supported_conv2d_modules(model)
    return model.to(device=device, memory_format=torch.channels_last).eval()


def move_input_to_cutlass_layout(tensor, torch, device):
    """Move NCHW image tensors to CUDA channels-last layout."""
    return tensor.to(device=device, memory_format=torch.channels_last)


def configure_yolo_model_layout(model, torch):
    """Apply channels-last layout to the wrapped PyTorch YOLO module when present."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "to"):
        if os.environ.get("USE_STANDALONE_CUTLASS_CONV", "0").strip().lower() in {"1", "true", "yes", "on"}:
            from cutlass_standalone_conv import replace_supported_conv2d_modules

            inner = replace_supported_conv2d_modules(inner)
            model.model = inner
        inner.to(memory_format=torch.channels_last)
    return model


def assert_cutlass_only_kernel_info(path) -> None:
    """Fail if any profiled HMMA kernel is not a non-cuDNN CUTLASS kernel."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    bad = []
    current = None
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("KERNEL "):
            parts = line.split()
            current = {"id": int(parts[1]), "has_target": parts[2] == "1", "name": ""}
        elif current and line.startswith("NAME "):
            current["name"] = line[5:]
            lower = current["name"].lower()
            if not current["has_target"]:
                continue
            if "cutlass" not in lower:
                bad.append((current["id"], "not cutlass", current["name"]))

    if bad:
        preview = "\n".join(
            f"  kernel {kid} ({reason}): {name}" for kid, reason, name in bad[:10]
        )
        suffix = "" if len(bad) <= 10 else f"\n  ... {len(bad) - 10} more"
        raise RuntimeError(
            f"{path} contains {len(bad)} disallowed HMMA kernels:\n{preview}{suffix}"
        )

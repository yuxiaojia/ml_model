#!/usr/bin/env python3
"""Standalone CUTLASS Conv2d bridge for new_example_cutlass_modified."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torch._dynamo as _dynamo
except Exception:
    _dynamo = None


_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None
_USAGE_STATS = {
    "modules_replaced": 0,
    "standalone_calls": 0,
    "direct_calls": 0,
    "fallback_calls": 0,
}

_DEFAULT_CUTLASS_DIR = Path(__file__).resolve().parents[4] / "cutlass_test"
_DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / ".cutlass_test_standalone_conv_build"
_BASE_EXTENSION_NAME = "torch_cutlass_test_conv_ext_v1"


def _dynamo_disable(fn):
    if _dynamo is not None and hasattr(_dynamo, "disable"):
        return _dynamo.disable(fn)
    return fn


def cutlass_standalone_conv_enabled() -> bool:
    raw = os.environ.get("USE_STANDALONE_CUTLASS_CONV", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_enabled(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _strict_mode() -> bool:
    return _env_enabled("STANDALONE_CUTLASS_STRICT")


def _tail_duplicate_print_enabled() -> bool:
    return _env_enabled("STANDALONE_CUTLASS_TAIL_DUP_PRINT")


def _tail_duplicate_enabled() -> bool:
    return _env_enabled("STANDALONE_CUTLASS_TAIL_DUP") or _tail_duplicate_print_enabled()


def _extension_name() -> str:
    if _tail_duplicate_enabled() and not _tail_duplicate_print_enabled():
        return f"{_BASE_EXTENSION_NAME}_taildup"
    if _tail_duplicate_print_enabled():
        return f"{_BASE_EXTENSION_NAME}_taildup_print"
    return _BASE_EXTENSION_NAME


def _cutlass_dir() -> Path:
    raw = os.environ.get("STANDALONE_CUTLASS_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_CUTLASS_DIR.resolve()


def _extension_sources() -> list[str]:
    base = _cutlass_dir() / "tools" / "torch_cutlass_conv"
    return [
        str(base / "cutlass_conv2d_ext.cpp"),
        str(base / "cutlass_conv2d_kernel.cu"),
    ]


def runtime_config() -> dict[str, str]:
    """Return paths that prove which standalone CUTLASS extension is active."""
    cutlass_dir = _cutlass_dir()
    build_dir = Path(os.environ.get("STANDALONE_CUTLASS_BUILD_DIR", _DEFAULT_BUILD_DIR)).expanduser()
    return {
        "cutlass_dir": str(cutlass_dir),
        "build_dir": str(build_dir),
        "extension_name": _extension_name(),
        "kernel_source": str(cutlass_dir / "tools" / "torch_cutlass_conv" / "cutlass_conv2d_kernel.cu"),
        "tail_duplicate": str(_tail_duplicate_enabled()),
        "tail_duplicate_print": str(_tail_duplicate_print_enabled()),
    }


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError("CUTLASS standalone conv extension is unavailable") from _EXTENSION_ERROR

    cutlass_dir = _cutlass_dir()
    if not cutlass_dir.exists():
        raise FileNotFoundError(f"Standalone CUTLASS directory not found: {cutlass_dir}")

    try:
        from torch.utils.cpp_extension import load

        build_dir = Path(os.environ.get("STANDALONE_CUTLASS_BUILD_DIR", _DEFAULT_BUILD_DIR))
        build_dir.mkdir(parents=True, exist_ok=True)
        extra_cuda_cflags = [
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
        ]
        if _tail_duplicate_enabled():
            extra_cuda_cflags.extend(
                [
                    "-DCUTLASS_ENABLE_IMPLICIT_GEMM_MULTISTAGE_TAIL_DUPLICATE=1",
                    "-DCUTLASS_ENABLE_MMA_MULTISTAGE_TAIL_DUPLICATE=1",
                ]
            )
        if _tail_duplicate_print_enabled():
            extra_cuda_cflags.extend(
                [
                    "-DCUTLASS_TAIL_DUPLICATE_ENABLE_PRINTF=1",
                    "-DCUTLASS_IMPLICIT_GEMM_TAIL_DUPLICATE_PRINT_REGS=1",
                ]
            )

        _EXTENSION = load(
            name=_extension_name(),
            sources=_extension_sources(),
            extra_include_paths=[str(cutlass_dir / "include")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=str(build_dir),
            verbose=os.environ.get("STANDALONE_CUTLASS_VERBOSE", "0") == "1",
        )
        return _EXTENSION
    except Exception as exc:  # pragma: no cover - depends on local toolchain
        _EXTENSION_ERROR = exc
        raise


def reset_usage_stats() -> None:
    for key in _USAGE_STATS:
        _USAGE_STATS[key] = 0


def usage_stats() -> dict[str, int]:
    return dict(_USAGE_STATS)


class StandaloneCutlassConv2d(torch.nn.Module):
    """Inference-only Conv2d wrapper backed by a standalone CUTLASS kernel."""

    def __init__(self, conv: torch.nn.Conv2d):
        super().__init__()
        if not isinstance(conv, torch.nn.Conv2d):
            raise TypeError(f"Expected Conv2d, got {type(conv)!r}")

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = tuple(conv.kernel_size)
        self.stride = tuple(conv.stride)
        self.padding = tuple(conv.padding)
        self.dilation = tuple(conv.dilation)
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode

        self.register_buffer("weight", conv.weight.detach().contiguous())
        bias = conv.bias.detach().contiguous() if conv.bias is not None else None
        self.register_buffer("bias", bias)
        packed_weight = conv.weight.detach().permute(0, 2, 3, 1).contiguous().to(dtype=torch.float16)
        self.register_buffer("weight_krsc_half", packed_weight)

    def _supports_cutlass(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and self.groups == 1
            and self.padding_mode == "zeros"
            and x.ndim == 4
        )

    def _supports_direct(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and self.groups > 1
            and self.padding_mode == "zeros"
            and x.ndim == 4
        )

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        _USAGE_STATS["fallback_calls"] += 1
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @_dynamo_disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self._supports_cutlass(x) or self._supports_direct(x)):
            return self._fallback_forward(x)

        try:
            ext = _load_extension()
            input_nhwc = x.permute(0, 2, 3, 1).contiguous().to(dtype=torch.float16)
            weight_krsc = self.weight_krsc_half.to(device=x.device, dtype=torch.float16).contiguous()
            if self.groups == 1:
                output_nhwc = ext.conv2d_forward(
                    input_nhwc,
                    weight_krsc,
                    self.stride[0],
                    self.stride[1],
                    self.padding[0],
                    self.padding[1],
                    self.dilation[0],
                    self.dilation[1],
                )
                _USAGE_STATS["standalone_calls"] += 1
            else:
                output_nhwc = ext.grouped_conv2d_forward(
                    input_nhwc,
                    weight_krsc,
                    self.groups,
                    self.stride[0],
                    self.stride[1],
                    self.padding[0],
                    self.padding[1],
                    self.dilation[0],
                    self.dilation[1],
                )
                _USAGE_STATS["direct_calls"] += 1
            if self.bias is not None:
                output_nhwc = output_nhwc + self.bias.to(output_nhwc.dtype).view(1, 1, 1, -1)
            output = output_nhwc.permute(0, 3, 1, 2).contiguous()
            if output.dtype != x.dtype:
                output = output.to(dtype=x.dtype)
            return output
        except Exception:
            if _strict_mode():
                raise
            return self._fallback_forward(x)

    @classmethod
    def from_conv2d(cls, conv: torch.nn.Conv2d) -> "StandaloneCutlassConv2d":
        return cls(conv)


def replace_supported_conv2d_modules(model: torch.nn.Module) -> torch.nn.Module:
    """Replace Conv2d leaves with the standalone CUTLASS wrapper."""
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Conv2d):
            setattr(model, name, StandaloneCutlassConv2d.from_conv2d(child))
            _USAGE_STATS["modules_replaced"] += 1
        else:
            replace_supported_conv2d_modules(child)
    return model

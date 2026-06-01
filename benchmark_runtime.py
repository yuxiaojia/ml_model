#!/usr/bin/env python3
"""Measure CIFAR model forward runtime across the three ml_bench setups."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path


SETUPS = ("pytorch_original", "custom_conv_fresh", "custom_conv_cutlass_test")
MODELS = ("resnet", "mobilenet", "shufflenet")
HUB_IDS = {
    "resnet": "cifar100_resnet20",
    "mobilenet": "cifar100_mobilenetv2_x1_0",
    "shufflenet": "cifar100_shufflenetv2_x1_0",
}
DEFAULT_CUTLASS_DIRS = {
    "custom_conv_fresh": "/net/netscratch/yjia305/cutlass_fresh",
    "custom_conv_cutlass_test": "/net/netscratch/yjia305/cutlass_test",
}


def configure_env(
    setup: str,
    cutlass_dir: Path | None,
    build_dir: Path,
    strict: bool,
    tail_dup: bool,
    tail_dup_print: bool,
    cta_dup: bool,
) -> None:
    os.environ["USE_TORCH_COMPILE"] = "0"
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    if setup == "pytorch_original":
        os.environ["USE_STANDALONE_CUTLASS_CONV"] = "0"
        return

    if cutlass_dir is None:
        raise ValueError(f"--cutlass-dir is required for {setup}")
    os.environ["USE_STANDALONE_CUTLASS_CONV"] = "1"
    os.environ["STANDALONE_CUTLASS_STRICT"] = "1" if strict else "0"
    os.environ["CUTLASS_DIR"] = str(cutlass_dir)
    os.environ["STANDALONE_CUTLASS_DIR"] = str(cutlass_dir)
    os.environ["STANDALONE_CUTLASS_BUILD_DIR"] = str(build_dir)
    os.environ["STANDALONE_CUTLASS_TAIL_DUP"] = "1" if (tail_dup or tail_dup_print) else "0"
    os.environ["STANDALONE_CUTLASS_TAIL_DUP_PRINT"] = "1" if tail_dup_print else "0"
    os.environ["STANDALONE_CUTLASS_CTA_DUP"] = "1" if cta_dup else "0"


def import_setup(bench_root: Path, setup: str) -> None:
    setup_dir = bench_root / setup
    sys.path.insert(0, str(setup_dir))
    if setup != "pytorch_original":
        import sitecustomize  # noqa: F401


def build_loader(torch, data_root: Path, batch_size: int, workers: int):
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    dataset = datasets.CIFAR100(
        root=str(data_root),
        train=False,
        download=False,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "timed_batches": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "total_forward_ms": sum(values),
    }


def run_one(args: argparse.Namespace) -> dict[str, object]:
    cutlass_dir = args.cutlass_dir
    if cutlass_dir is None and args.setup in DEFAULT_CUTLASS_DIRS:
        cutlass_dir = Path(DEFAULT_CUTLASS_DIRS[args.setup])
    cutlass_dir = cutlass_dir.resolve() if cutlass_dir is not None else None
    build_dir = (args.build_root / f"{args.setup}_{args.model}").resolve()

    configure_env(args.setup, cutlass_dir, build_dir, args.strict, args.tail_dup, args.tail_dup_print, args.cta_dup)
    import_setup(args.bench_root, args.setup)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16)

    loader = build_loader(torch, args.data_root, args.batch_size, args.workers)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", HUB_IDS[args.model], pretrained=True)
    model = model.to(device).eval()

    usage: dict[str, object] = {}
    try:
        from cutlass_standalone_conv import reset_usage_stats, runtime_config

        reset_usage_stats()
        usage["runtime_config"] = runtime_config()
    except Exception as exc:
        if args.setup != "pytorch_original":
            usage["runtime_config_error"] = repr(exc)

    first_batch = next(iter(loader))
    with torch.no_grad():
        for _ in range(args.warmup):
            x = first_batch[0].to(device, non_blocking=True)
            with amp_ctx:
                _ = model(x)
            torch.cuda.synchronize()

    timings_ms: list[float] = []
    correct = 0
    total = 0
    wall_start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if args.batches is not None and batch_idx >= args.batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(f"{args.setup}:{args.model}:batch_{batch_idx:03d}")
            try:
                with amp_ctx:
                    start.record()
                    logits = model(x)
                    end.record()
            finally:
                torch.cuda.nvtx.range_pop()
            end.synchronize()

            timings_ms.append(start.elapsed_time(end))
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    torch.cuda.synchronize()

    try:
        from cutlass_standalone_conv import usage_stats

        usage["usage_stats"] = usage_stats()
    except Exception:
        pass

    result: dict[str, object] = {
        "setup": args.setup,
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "strict": args.strict,
        "tail_dup": args.tail_dup or args.tail_dup_print,
        "tail_dup_print": args.tail_dup_print,
        "cta_dup": args.cta_dup,
        "samples": total,
        "accuracy_pct": 100.0 * correct / max(total, 1),
        "wall_s": time.perf_counter() - wall_start,
        **summarize(timings_ms),
        **usage,
    }
    return result


def write_outputs(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result['model']}_{result['setup']}"
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    flat = dict(result)
    flat.pop("runtime_config", None)
    flat.pop("usage_stats", None)
    flat.pop("runtime_config_error", None)
    with (output_dir / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat))
        writer.writeheader()
        writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--setup", choices=SETUPS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/net/netscratch/yjia305/nvbit_release_x86_64/TensorDynamic_own_version/examples/.data"),
    )
    parser.add_argument("--cutlass-dir", type=Path)
    parser.add_argument("--build-root", type=Path, default=Path("/tmp/ml_bench_runtime_build"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime_results"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batches", type=int, help="Timed batches. Omit to run the full dataset.")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--tail-dup", action="store_true", help="Enable tail-duplicate HMMA/check without device printf.")
    parser.add_argument("--tail-dup-print", action="store_true")
    parser.add_argument("--cta-dup", action="store_true", help="Enable CTA/launch-level duplicate compare without device printf.")
    args = parser.parse_args()

    args.bench_root = args.bench_root.resolve()
    args.data_root = args.data_root.resolve()
    args.build_root = args.build_root.resolve()
    args.output_dir = args.output_dir.resolve()

    result = run_one(args)
    write_outputs(result, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

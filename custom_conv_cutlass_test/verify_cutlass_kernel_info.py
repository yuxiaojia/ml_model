#!/usr/bin/env python3
"""Verify that profiled HMMA kernels in kernel_info.txt are CUTLASS-backed."""

from __future__ import annotations

import argparse
from pathlib import Path

from cutlass_only_config import assert_cutlass_only_kernel_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(p) for p in Path(__file__).parent.glob("*/kernel_info.txt")],
        help="kernel_info.txt files to verify. Defaults to */kernel_info.txt.",
    )
    args = parser.parse_args()

    for path in args.paths:
        assert_cutlass_only_kernel_info(path)
        print(f"OK: {path}")


if __name__ == "__main__":
    main()

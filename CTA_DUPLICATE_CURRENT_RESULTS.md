# Current cutlass_intra_warp CTA Duplicate Results

This note documents the current duplicate implementation measured from:

```text
CUTLASS repo: /nethome/yjia305/USERSCRATCH/cutlass_intra_warp
remote: https://github.com/yuxiaojia/cutlass_test.git
branch: intra-warp-duplication
commit: b2f085c9 Add CTA duplicate implicit GEMM wrapper
```

Important: despite the branch name, the current implementation is not warp-level
`warp_mma` duplication. The active duplicate mechanism is CTA/kernel-launch
level DMR in:

```text
include/cutlass/conv/device/implicit_gemm_convolution.h
```

It is compiled by enabling:

```text
-DCUTLASS_ENABLE_IMPLICIT_GEMM_CTA_DUPLICATE=1
```

In `ml_bench`, this is exposed as:

```bash
python -B benchmark_runtime.py ... --cta-dup
```

## Implementation

When `CUTLASS_ENABLE_IMPLICIT_GEMM_CTA_DUPLICATE=1` is enabled, the CUTLASS
device wrapper does the following for each implicit GEMM convolution:

1. Launch the normal authoritative CUTLASS kernel with the original params.
   This writes the real output tensor `ptr_D`.
2. Allocate a shadow output buffer and a one-word mismatch flag.
3. Copy the same kernel params, changing only the output pointer to the shadow
   buffer.
4. Launch the shadow CUTLASS kernel on a second CUDA stream when stream creation
   succeeds.
5. Synchronize both launches.
6. Run `implicit_gemm_cta_duplicate_compare_kernel` to compare real output vs
   shadow output element by element.
7. Copy the mismatch flag back to host and free scratch memory.

The authoritative output is still the normal CUTLASS output. The duplicate path
checks that a second launch on the same inputs produced the same output.

This is DMR-style duplication at the kernel/output level. It is not:

- intra-warp HMMA duplication,
- tail-only K-loop duplication,
- helper-warp duplication,
- or duplicate `warp_mma` inside the same mainloop.

## How To Run

From `ml_bench`:

```bash
cd /nethome/yjia305/USERSCRATCH/ml_bench
source /net/netscratch/yjia305/setup_env.sh

python -B benchmark_runtime.py \
  --setup custom_conv_cutlass_test \
  --model resnet \
  --warmup 1 \
  --cutlass-dir /nethome/yjia305/USERSCRATCH/cutlass_intra_warp \
  --build-root /tmp/ml_bench_intra_warp_current_full \
  --output-dir runtime_results_intra_warp_current_full/cta_duplicate \
  --cta-dup
```

For MobileNet, use `--model mobilenet`.

For ShuffleNet, add `--no-strict`:

```bash
python -B benchmark_runtime.py \
  --setup custom_conv_cutlass_test \
  --model shufflenet \
  --warmup 1 \
  --no-strict \
  --cutlass-dir /nethome/yjia305/USERSCRATCH/cutlass_intra_warp \
  --build-root /tmp/ml_bench_intra_warp_current_full \
  --output-dir runtime_results_intra_warp_current_full/cta_duplicate \
  --cta-dup
```

To run the matching baseline, omit `--cta-dup`.

## Validation Fields

The duplicate JSON should contain:

```json
{
  "cta_dup": true,
  "tail_dup": false,
  "tail_dup_print": false,
  "runtime_config": {
    "cutlass_dir": "/net/netscratch/yjia305/cutlass_intra_warp",
    "extension_name": "torch_cutlass_test_conv_ext_v1_ctadup",
    "cta_duplicate": "True",
    "tail_duplicate": "False",
    "tail_duplicate_print": "False"
  }
}
```

The baseline JSON should contain:

```json
{
  "cta_dup": false,
  "runtime_config": {
    "extension_name": "torch_cutlass_test_conv_ext_v1",
    "cta_duplicate": "False"
  }
}
```

## CUDA Event Timing

Full CIFAR test set:

```text
timed_batches = 100
batch_size = 100
samples = 10000
device = NVIDIA A100-PCIE-40GB
```

Results from `runtime_results_intra_warp_current_full/summary.md`:

| model | baseline ms/batch | CTA duplicate ms/batch | absolute slowdown | overhead |
|---|---:|---:|---:|---:|
| resnet | 6.141 | 16.427 | +10.287 | +167.52% |
| mobilenet | 13.308 | 29.091 | +15.783 | +118.60% |
| shufflenet | 13.783 | 26.124 | +12.341 | +89.54% |

Use these CUDA-event numbers as the primary performance overhead numbers.

## Nsight Timing

Profiles were also collected with:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
```

Results from `runtime_results_nsys_intra_warp_current_full/summary.md`:

| model | baseline event ms under nsys | CTA duplicate event ms under nsys | absolute slowdown | overhead |
|---|---:|---:|---:|---:|
| resnet | 8.290 | 20.976 | +12.687 | +153.04% |
| mobilenet | 18.698 | 35.452 | +16.754 | +89.60% |
| shufflenet | 21.729 | 31.637 | +9.908 | +45.60% |

Nsight changes timing, so use it for trace validation and kernel inspection, not
as the primary overhead claim.

## Correctness Check

A direct ResNet first-batch comparison between normal standalone CUTLASS and the
CTA duplicate extension produced identical logits:

```text
max_abs_diff = 0.0
num_different_elements = 0
argmax_equal_count = 100 / 100
```

Saved artifact:

```text
runtime_results_cta_dup_compile_test/resnet_cta_correctness_compare.json
```

## Result Artifacts

Primary CUDA-event summary:

```text
runtime_results_intra_warp_current_full/summary.md
runtime_results_intra_warp_current_full/summary.csv
```

Nsight summary and reports:

```text
runtime_results_nsys_intra_warp_current_full/summary.md
runtime_results_nsys_intra_warp_current_full/summary.csv
runtime_results_nsys_intra_warp_current_full/*.nsys-rep
```

Initial compile smoke test:

```text
runtime_results_cta_dup_compile_test/summary.md
```


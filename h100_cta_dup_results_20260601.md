# H100 CTA Duplication Results

New CUTLASS checkout:

```text
/storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_intra_warp_dup
branch: intra-warp-duplication
commit: b2f085c9 Add CTA duplicate implicit GEMM wrapper
```

Important correction:

- The current tree implements CTA/launch-level duplication in `include/cutlass/conv/device/implicit_gemm_convolution.h`.
- It is guarded by `CUTLASS_ENABLE_IMPLICIT_GEMM_CTA_DUPLICATE=1`.
- It is not the old sequential tail HMMA duplication path.
- It is not enabled by `benchmark_runtime.py --tail-dup` in this `ml_bench` checkout.

The corrected H100 run used:

```text
NVCC_PREPEND_FLAGS=-DCUTLASS_ENABLE_IMPLICIT_GEMM_CTA_DUPLICATE=1
tail_duplicate=False
tail_duplicate_print=False
```

Benchmark shape:

```text
GPU: NVIDIA H100 80GB HBM3
batch size: 100
timed batches: 100
samples: 10,000
benchmark: benchmark_runtime.py
```

ShuffleNet uses `--no-strict`, matching prior runs.

## CUDA Event Timing

Mean ms/batch from clean CUDA-event timing.

| model | clean cutlass_test | previous taildup | CTA duplication | CTA vs clean cutlass_test | CTA vs previous taildup |
|---|---:|---:|---:|---:|---:|
| resnet | 6.760 | 19.271 | 15.929 | +135.65% | -17.34% |
| mobilenet | 13.144 | 28.802 | 25.824 | +96.47% | -10.34% |
| shufflenet | 7.204 | 16.092 | 15.240 | +111.55% | -5.30% |

Sources:

```text
clean cutlass_test: runtime_results_h100_20260531_redo_clean/
previous taildup: runtime_results_h100_20260531_taildup_cuda_event_1747/
CTA duplication: runtime_results_h100_20260601_cta_dup_cuda_event/
```

## Coverage

| model | standalone calls | direct calls | fallback calls | strict |
|---|---:|---:|---:|---|
| resnet | 2121 | 0 | 0 | true |
| mobilenet | 3535 | 1717 | 0 | true |
| shufflenet | 2828 | 1919 | 909 | false |

The coverage pattern matches the previous H100 taildup runs.

## Nsight Systems

Full 100-batch Nsight profiles were also generated with the CTA macro enabled.

| model | CUDA-event mean under Nsight ms/batch | report |
|---|---:|---|
| resnet | 16.722 | `resnet_custom_conv_cutlass_test_cta_dup.nsys-rep` |
| mobilenet | 28.465 | `mobilenet_custom_conv_cutlass_test_cta_dup.nsys-rep` |
| shufflenet | 19.577 | `shufflenet_custom_conv_cutlass_test_cta_dup.nsys-rep` |

Nsight output directory:

```text
runtime_results_h100_20260601_cta_dup_nsys_100b/
```

## Notes

An earlier low-overhead run pointed at the new checkout but used `--tail-dup`, which did not enable `CUTLASS_ENABLE_IMPLICIT_GEMM_CTA_DUPLICATE`. That result measured the normal kernel path from the new checkout and has been removed from the committed result set.

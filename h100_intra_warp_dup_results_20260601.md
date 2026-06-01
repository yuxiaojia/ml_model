# H100 Intra-Warp Duplication Results

New CUTLASS checkout:

```text
/storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_intra_warp_dup
branch: intra-warp-duplication
commit: b2f085c9 Add CTA duplicate implicit GEMM wrapper
```

Benchmark shape:

```text
GPU: NVIDIA H100 80GB HBM3
batch size: 100
timed batches: 100
samples: 10,000
benchmark: benchmark_runtime.py
flag: --tail-dup
print flag: disabled
```

ShuffleNet uses `--no-strict`, matching prior runs.

## CUDA Event Timing

Mean ms/batch from clean CUDA-event timing.

| model | clean cutlass_test | previous taildup | intra-warp duplication | vs clean cutlass_test | vs previous taildup |
|---|---:|---:|---:|---:|---:|
| resnet | 6.760 | 19.271 | 6.393 | -5.42% | -66.82% |
| mobilenet | 13.144 | 28.802 | 12.963 | -1.37% | -54.99% |
| shufflenet | 7.204 | 16.092 | 6.657 | -7.59% | -58.63% |

Sources:

```text
clean cutlass_test: runtime_results_h100_20260531_redo_clean/
previous taildup: runtime_results_h100_20260531_taildup_cuda_event_1747/
intra-warp duplication: runtime_results_h100_20260601_intra_warp_dup_cuda_event/
```

## Coverage

| model | standalone calls | direct calls | fallback calls | strict |
|---|---:|---:|---:|---|
| resnet | 2121 | 0 | 0 | true |
| mobilenet | 3535 | 1717 | 0 | true |
| shufflenet | 2828 | 1919 | 909 | false |

The coverage pattern matches the previous H100 taildup runs.

## Nsight Systems

Full 100-batch Nsight profiles were also generated.

| model | CUDA-event mean under Nsight ms/batch | report |
|---|---:|---|
| resnet | 6.326 | `resnet_custom_conv_cutlass_test_intra_warp_dup.nsys-rep` |
| mobilenet | 13.131 | `mobilenet_custom_conv_cutlass_test_intra_warp_dup.nsys-rep` |
| shufflenet | 9.363 | `shufflenet_custom_conv_cutlass_test_intra_warp_dup.nsys-rep` |

Nsight output directory:

```text
runtime_results_h100_20260601_intra_warp_dup_nsys_100b/
```

## Notes

- The smoke run with `--warmup 0 --batches 2` showed a large first timed batch from first-use/JIT overhead, so the full result uses the normal warmup path.
- The intra-warp duplication branch is much faster than the previous diagnostic taildup path and is close to clean `cutlass_test` in CUDA-event timing.

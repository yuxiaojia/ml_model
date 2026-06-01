# H100 CUDA Event vs Nsight Variant Comparison

All rows use batch size 100 and 100 timed batches, for 10,000 CIFAR-100 images.

## CUDA Event Timing

Clean runtime from `benchmark_runtime.py`, without Nsight profiler attached.

| model | PyTorch ms/batch | cutlass_fresh ms/batch | cutlass_test ms/batch | corrected taildup ms/batch | taildup overhead vs cutlass_test |
|---|---:|---:|---:|---:|---:|
| resnet | 1.780 | 6.290 | 6.760 | 19.271 | +185.09% |
| mobilenet | 3.667 | 13.009 | 13.144 | 28.802 | +119.13% |
| shufflenet | 5.201 | 7.088 | 7.204 | 16.092 | +123.39% |

Sources:
- Clean variants: `runtime_results_h100_20260531_redo_clean/`
- Corrected taildup: `runtime_results_h100_20260531_taildup_cuda_event_1747/`

## Nsight Systems Timing

Same benchmark shape, but run under:

```text
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
```

The numbers below are CUDA-event means recorded while Nsight was attached. Use these for within-Nsight comparison only, not as clean runtime.

| model | PyTorch under nsys ms/batch | cutlass_fresh under nsys ms/batch | cutlass_test under nsys ms/batch | corrected taildup under nsys ms/batch | taildup overhead vs cutlass_test under nsys |
|---|---:|---:|---:|---:|---:|
| resnet | 6.095 | 7.992 | 7.736 | 19.078 | +146.60% |
| mobilenet | 15.635 | 18.438 | 18.618 | 28.503 | +53.10% |
| shufflenet | 16.872 | 22.042 | 21.622 | 17.014 | -21.31% |

Sources:
- Clean variants under Nsight: `runtime_results_nsys_after_print_removal/`
- Corrected taildup under Nsight: `runtime_results_h100_20260531_taildup_nsys_100b_2130/`

Generated Nsight reports:

```text
runtime_results_h100_20260531_taildup_nsys_100b_2130/resnet_custom_conv_cutlass_test_taildup.nsys-rep
runtime_results_h100_20260531_taildup_nsys_100b_2130/mobilenet_custom_conv_cutlass_test_taildup.nsys-rep
runtime_results_h100_20260531_taildup_nsys_100b_2130/shufflenet_custom_conv_cutlass_test_taildup.nsys-rep
```

## Interpretation

The CUDA-event table is the cleaner performance comparison because it avoids profiler perturbation. The Nsight table is for trace validation and profiler-context comparison.

Nsight changes the denominator: PyTorch and clean CUTLASS runs are already slower under profiling, so the apparent percentage overhead can shrink or even change direction. For example, ShuffleNet taildup is slower than clean `cutlass_test` in the clean CUDA-event run, but lower than the clean `cutlass_test` row under Nsight in this run.

Note: the current corrected taildup logs still contain `IMPLICIT_GEMM_PIPELINED_TAIL_CHECK_V2` diagnostic output even though `tail_dup_print` is `false` in the JSON metadata. Treat these taildup timings as measuring the currently built diagnostic taildup path.

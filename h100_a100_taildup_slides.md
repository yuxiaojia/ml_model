---
marp: true
title: A100/H100 Conv2D Tail-Duplication Evaluation
paginate: true
---

# A100/H100 Conv2D Tail-Duplication Evaluation

CUDA-event timing and Nsight Systems validation for CIFAR-100 Conv2D workloads.

**Models:** ResNet20, MobileNetV2, ShuffleNetV2  
**GPUs:** A100-PCIE-40GB, H100 80GB HBM3  
**Batch size:** 100  
**Timed batches:** 100  
**Images per run:** 10,000

---

# Goal

Evaluate the runtime impact of a CUTLASS Conv2D tail-duplication path.

Key questions:

- What are the A100 results across PyTorch and CUTLASS setups?
- What are the H100 results across PyTorch, CUTLASS, and taildup setups?
- How much overhead comes from the standalone CUTLASS wrapper?
- How much additional overhead comes from enabling taildup?
- Which models fall back to PyTorch conv paths?
- What does Nsight Systems validate?

---

# Experimental Setups

| setup | meaning |
|---|---|
| PyTorch | Original framework conv path |
| cutlass_fresh | Standalone CUTLASS Conv2D wrapper, no taildup modification |
| cutlass_test | Modified CUTLASS checkout, taildup-capable but taildup disabled |
| cutlass_test + previous taildup | Same `cutlass_test` checkout with `--tail-dup` enabled |
| new intra-warp branch | `intra-warp-duplication` branch; current measured implementation uses CTA duplicate macro |

Previous taildup is **not** a separate repo. It is:

```text
cutlass_test + --tail-dup
```

The new intra-warp branch data is enabled by:

```text
NVCC_PREPEND_FLAGS=-DCUTLASS_ENABLE_IMPLICIT_GEMM_CTA_DUPLICATE=1
```

---

# Measurement Method

Primary runtime metric:

```text
benchmark_runtime.py CUDA events around model forward
```

Profiler validation:

```text
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
```

Interpretation:

- CUDA-event timing is the main performance number.
- Nsight Systems is for trace validation and launch/kernel inspection.
- Nsight timing is perturbed by profiler overhead.

---

# A100 Clean CUDA Event Timing

Non-Nsight `benchmark_runtime.py` means, ms/batch.

| model | PyTorch | cutlass_fresh | cutlass_test |
|---|---:|---:|---:|
| resnet | 3.813 | 6.085 | 6.115 |
| mobilenet | 9.858 | 13.535 | 13.863 |
| shufflenet | 9.551 | 14.837 | 13.716 |

---

# A100 Setup Takeaways

Within the A100 run:

- PyTorch is fastest for all three models.
- `cutlass_fresh` and `cutlass_test` are close.
- `cutlass_test` is slightly slower than `cutlass_fresh` for ResNet and MobileNet.
- ShuffleNet `cutlass_test` is slightly faster than `cutlass_fresh`.

These rows are useful as the A100 baseline/control setup results.

---

# H100 Setup Comparison

CUDA-event mean ms/batch, 100 timed batches.

| model | PyTorch | naive CUTLASS / fresh | cutlass_test | intra_warp current |
|---|---:|---:|---:|---:|
| resnet | 3.932 | 4.730 | 5.847 | 16.427 |
| mobilenet | 9.753 | 13.656 | 13.974 | 29.091 |
| shufflenet | 10.055 | 14.113 | 13.559 | 26.124 |

Main observations:

- `cutlass_fresh` and clean `cutlass_test` are close.
- `intra_warp current` adds large overhead relative to clean `cutlass_test`.
- PyTorch remains fastest for these CIFAR models on H100.

---

# H100 Previous Taildup Rerun

Previous taildup overhead relative to clean `cutlass_test`.

| model | clean cutlass_test | taildup | overhead |
|---|---:|---:|---:|
| resnet | 6.760 | 19.264 | +184.99% |
| mobilenet | 13.144 | 28.922 | +120.04% |
| shufflenet | 7.204 | 15.940 | +121.27% |

Interpretation:

- The rerun reproduces the earlier high overhead.
- Rerun vs old run changed by less than 1% for all three models.
- The old taildup path still emitted diagnostics during the rerun even with `tail_dup_print=False`.

---

# H100 New Intra-Warp Branch

CUDA-event results for the current `intra-warp-duplication` branch.

| model | cutlass_test | intra_warp current | overhead |
|---|---:|---:|---:|
| resnet | 5.847 | 16.427 | +180.95% |
| mobilenet | 13.974 | 29.091 | +108.18% |
| shufflenet | 13.559 | 26.124 | +92.67% |

Interpretation:

- This is the new `intra-warp-duplication` branch result.
- In the current checked-out branch, the measured mechanism is CTA/launch-level duplication.
- It is still far slower than clean `cutlass_test`.
- It is not yet true warp-level HMMA duplication in the mainloop.

---

# Fallback and Coverage

100 timed batches, H100 previous-taildup rerun.

| model | standalone calls | direct calls | fallback calls | strict |
|---|---:|---:|---:|---|
| resnet | 2121 | 0 | 0 | true |
| mobilenet | 3535 | 1717 | 0 | true |
| shufflenet | 2828 | 1919 | 909 | false |

Meaning:

- ResNet and MobileNet fully avoid fallback.
- ShuffleNet uses `--no-strict` and has 909 fallback calls.
- ShuffleNet timing mixes custom CUTLASS path and PyTorch fallback path.

---

# Why ShuffleNet Uses `--no-strict`

`strict=true` means unsupported or failing CUTLASS convs abort the run.

`strict=false` means unsupported/problematic convs fall back to PyTorch.

ShuffleNet strict mode hits a CUTLASS misaligned operand case, so the completed runs use:

```text
--tail-dup --no-strict
```

This is why ShuffleNet reports:

```text
fallback_calls = 909
```

---

# Nsight Systems Validation

Nsight confirms that the H100 duplication paths run and produce trace artifacts.

H100 full 100-batch Nsight reports:

```text
runtime_results_h100_20260531_taildup_nsys_100b_2130/
runtime_results_h100_20260601_cta_dup_nsys_100b/
```

Reports:

```text
resnet_custom_conv_cutlass_test_taildup.nsys-rep
mobilenet_custom_conv_cutlass_test_taildup.nsys-rep
shufflenet_custom_conv_cutlass_test_taildup.nsys-rep
resnet_custom_conv_cutlass_test_cta_dup.nsys-rep
mobilenet_custom_conv_cutlass_test_cta_dup.nsys-rep
shufflenet_custom_conv_cutlass_test_cta_dup.nsys-rep
```

---

# H100 Nsight Timing

CUDA-event means recorded while Nsight Systems was attached.

| model | PyTorch | naive CUTLASS / fresh | cutlass_test | intra_warp current |
|---|---:|---:|---:|---:|
| resnet | 6.530 | 8.324 | 8.409 | 20.976 |
| mobilenet | 14.933 | 19.005 | 17.941 | 35.452 |
| shufflenet | 17.829 | 21.788 | 21.475 | 31.637 |

Interpretation:

- These are H100 profiler-context timings, not clean runtime timings.
- Nsight shows the same qualitative result as CUDA-event timing: `intra_warp current` is slower than clean `cutlass_test`.
- Use CUDA-event timing for the main overhead claim; use Nsight for profiler validation.

---

# Nsight Timing Caveat

Nsight changes timing:

- Adds profiling overhead.
- Perturbs synchronization and launch behavior.
- Makes clean baseline and taildup both slower.
- Should not be the primary overhead claim.

Use Nsight for:

- Kernel launch sequence
- NVTX range validation
- CUDA API behavior
- Confirming taildup code path

Use CUDA events for:

- Main runtime overhead numbers

---

# A100 Taildup Caveat

The available full-batch A100 taildup run in this checkout is print-validation:

```text
runtime_results_taildup_all_batches/
```

It used:

```text
--tail-dup-print
```

So it includes CUDA device `printf` overhead. I present it as an A100 validation artifact, not as a direct counterpart to the H100 no-print taildup timing.

Clean A100 taildup performance needs a separate rerun:

```text
A100 + cutlass_test + --tail-dup
```

without:

```text
--tail-dup-print
```

---

# Summary

Main conclusions:

- A100 clean setup results are available for PyTorch, `cutlass_fresh`, and `cutlass_test`.
- H100 clean, previous-taildup rerun, and new intra-warp branch results are available.
- Custom CUTLASS wrapper behavior is model-dependent.
- Clean `cutlass_fresh` and clean `cutlass_test` are close on both GPUs.
- Previous taildup currently adds large overhead on H100, and the rerun confirms it.
- The new intra-warp branch reduces overhead relative to previous taildup, but remains slower than clean `cutlass_test`.
- ResNet and MobileNet have zero fallback; ShuffleNet has fallback.
- Nsight validates behavior, but CUDA events should be used for overhead claims.

---

# Next Steps

Recommended follow-up experiments:

- Rerun A100 taildup with `--tail-dup` and no `--tail-dup-print`.
- Rerun H100 clean PyTorch/fresh/cutlass_test under Nsight for a complete H100 profiler-context table.
- Remove or disable diagnostic output path from taildup builds.
- Implement true warp-level HMMA duplication if that is the intended next mechanism.
- Optimize H100-specific CUTLASS configuration.
- Break down overhead by layer/kernel type.

---

# Result Files

Committed branch:

```text
H100_data
```

Main summaries:

```text
a100_h100_comparison_20260531.md
h100_cuda_nsys_variant_comparison_20260531.md
h100_cta_dup_results_20260601.md
h100_previous_taildup_rerun_20260601.md
```

H100 previous-taildup rerun CUDA-event results:

```text
runtime_results_h100_20260601_previous_taildup_rerun_cuda_event/
```

H100 previous-taildup old CUDA-event results:

```text
runtime_results_h100_20260531_taildup_cuda_event_1747/
```

H100 new intra-warp branch CUDA-event results:

```text
runtime_results_h100_20260601_cta_dup_cuda_event/
```

H100 taildup Nsight results:

```text
runtime_results_h100_20260531_taildup_nsys_100b_2130/
```

H100 new intra-warp branch Nsight results:

```text
runtime_results_h100_20260601_cta_dup_nsys_100b/
```

---

# Backup: Commands

Taildup no-print run:

```bash
python -B benchmark_runtime.py \
  --setup custom_conv_cutlass_test \
  --model resnet \
  --data-root /storage/ice1/0/3/yjia305/.data \
  --cutlass-dir /storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_test \
  --build-root /tmp/ml_bench_taildup_no_print \
  --output-dir runtime_results_taildup_no_print \
  --tail-dup
```

ShuffleNet additionally needs:

```text
--no-strict
```

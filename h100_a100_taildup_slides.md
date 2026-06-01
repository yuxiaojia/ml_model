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

- How do A100 and H100 compare on the same benchmark?
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
| cutlass_test + taildup | Same `cutlass_test` checkout with `--tail-dup` enabled |

Taildup is **not** a separate repo. It is:

```text
cutlass_test + --tail-dup
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

# A100 vs H100: Clean CUDA Event Timing

Non-Nsight `benchmark_runtime.py` means, ms/batch.

| model | setup | A100 | H100 | H100 vs A100 |
|---|---|---:|---:|---:|
| resnet | PyTorch | 3.813 | 1.780 | -53.3% |
| resnet | cutlass_fresh | 6.085 | 6.290 | +3.4% |
| resnet | cutlass_test | 6.115 | 6.760 | +10.5% |
| mobilenet | PyTorch | 9.858 | 3.667 | -62.8% |
| mobilenet | cutlass_fresh | 13.535 | 13.009 | -3.9% |
| mobilenet | cutlass_test | 13.863 | 13.144 | -5.2% |
| shufflenet | PyTorch | 9.551 | 5.201 | -45.5% |
| shufflenet | cutlass_fresh | 14.837 | 7.088 | -52.2% |
| shufflenet | cutlass_test | 13.716 | 7.204 | -47.5% |

---

# A100 vs H100: Takeaways

H100 improves PyTorch baseline substantially:

- ResNet: 3.813 -> 1.780 ms
- MobileNet: 9.858 -> 3.667 ms
- ShuffleNet: 9.551 -> 5.201 ms

Custom CUTLASS behavior is more model-dependent:

- MobileNet and ShuffleNet improve on H100.
- ResNet custom CUTLASS paths are slightly slower on H100 in these runs.
- The custom kernels/wrapper are not automatically H100-optimized.

---

# H100 Setup Comparison

CUDA-event mean ms/batch, 100 timed batches.

| model | PyTorch | cutlass_fresh | cutlass_test | cutlass_test + taildup |
|---|---:|---:|---:|---:|
| resnet | 1.780 | 6.290 | 6.760 | 19.271 |
| mobilenet | 3.667 | 13.009 | 13.144 | 28.802 |
| shufflenet | 5.201 | 7.088 | 7.204 | 16.092 |

Main observations:

- `cutlass_fresh` and clean `cutlass_test` are close.
- Taildup adds substantial overhead in the current implementation.
- PyTorch remains fastest for these small CIFAR models on H100.

---

# H100 Taildup Overhead

Taildup overhead relative to clean `cutlass_test`.

| model | clean cutlass_test | taildup | overhead |
|---|---:|---:|---:|
| resnet | 6.760 | 19.271 | +185.09% |
| mobilenet | 13.144 | 28.802 | +119.13% |
| shufflenet | 7.204 | 16.092 | +123.39% |

Interpretation:

- ResNet has the largest relative overhead.
- MobileNet and ShuffleNet also roughly double.
- These numbers reflect the currently built diagnostic taildup path.

---

# Fallback and Coverage

100 timed batches, H100 taildup run.

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

Nsight confirms that the corrected taildup path runs and produces trace artifacts.

H100 full 100-batch taildup Nsight reports:

```text
runtime_results_h100_20260531_taildup_nsys_100b_2130/
```

Reports:

```text
resnet_custom_conv_cutlass_test_taildup.nsys-rep
mobilenet_custom_conv_cutlass_test_taildup.nsys-rep
shufflenet_custom_conv_cutlass_test_taildup.nsys-rep
```

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

So it includes CUDA device `printf` overhead and should not be compared directly against H100 no-print taildup performance.

Clean A100-vs-H100 taildup comparison needs:

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

- H100 is much faster than A100 for PyTorch baselines.
- Custom CUTLASS wrapper performance is model-dependent.
- Clean `cutlass_fresh` and clean `cutlass_test` are close.
- Taildup currently adds large overhead on H100.
- ResNet and MobileNet have zero fallback; ShuffleNet has fallback.
- Nsight validates behavior, but CUDA events should be used for overhead claims.

---

# Next Steps

Recommended follow-up experiments:

- Rerun A100 taildup with `--tail-dup` and no `--tail-dup-print`.
- Rerun H100 clean PyTorch/fresh/cutlass_test under Nsight for same-GPU Nsight comparison.
- Remove or disable diagnostic output path from taildup builds.
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
```

H100 taildup CUDA-event results:

```text
runtime_results_h100_20260531_taildup_cuda_event_1747/
```

H100 taildup Nsight results:

```text
runtime_results_h100_20260531_taildup_nsys_100b_2130/
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


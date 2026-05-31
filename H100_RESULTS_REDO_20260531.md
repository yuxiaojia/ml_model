# H100 Tail Duplication Redo Results - 2026-05-31

## Branch and source state

- Branch: `h100-taildup-redo-20260531`
- `ml_bench` base commit after repull: `aa1f273 redo the experiment`
- Baseline CUTLASS tree: `/storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_fresh`
- Test CUTLASS tree: `/storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_test`
- GPU: NVIDIA H100 80GB HBM3
- Python: `/storage/ice1/0/3/yjia305/model_env/bin/python`
- Dataset root: `/storage/ice1/0/3/yjia305/.data`

The repulled `ml_bench` adds a separate `--tail-dup` option. The redo used that option for the tail duplication run instead of the older `--tail-dup-print` flag.

## Result directories

- Clean comparison: `runtime_results_h100_20260531_redo_clean`
- Tail duplication comparison: `runtime_results_h100_20260531_redo_taildup`
- Summary CSV: `h100_results_redo_20260531_summary.csv`

## Important print-path caveat

The `--tail-dup` harness path records `tail_duplicate=true` and `tail_duplicate_print=false` in the JSON runtime config. However, the current local `cutlass_test` header still prints whenever tail duplication is enabled.

In `/storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_test/include/cutlass/conv/threadblock/implicit_gemm_pipelined.h`, the mismatch and register `printf` block is guarded by:

```cpp
#if CUTLASS_ENABLE_IMPLICIT_GEMM_PIPELINED_TAIL_DUPLICATE
```

It is not additionally gated by `CUTLASS_TAIL_DUPLICATE_ENABLE_PRINTF` or `CUTLASS_IMPLICIT_GEMM_TAIL_DUPLICATE_PRINT_REGS`. Therefore the `--tail-dup` timings below still include device `printf` overhead from the CUTLASS implementation. They should not be treated as true silent tail-duplication overhead until the CUTLASS header gates those prints separately.

The printed sampled checks reported `mismatch=0`.

## Clean comparison

| Model | Setup | Mean ms | Median ms | P95 ms | Accuracy | Calls |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ResNet | PyTorch original | 1.780 | 1.620 | 2.134 | 68.83 | - |
| ResNet | CUTLASS fresh | 6.290 | 5.957 | 7.256 | 68.85 | direct=0, standalone=2121, fallback=0 |
| ResNet | CUTLASS test clean | 6.760 | 5.909 | 14.735 | 68.85 | direct=0, standalone=2121, fallback=0 |
| MobileNet | PyTorch original | 3.667 | 3.258 | 4.623 | 74.30 | - |
| MobileNet | CUTLASS fresh | 13.009 | 12.993 | 13.065 | 74.30 | direct=1717, standalone=3535, fallback=0 |
| MobileNet | CUTLASS test clean | 13.144 | 12.996 | 13.166 | 74.30 | direct=1717, standalone=3535, fallback=0 |
| ShuffleNet | PyTorch original | 5.201 | 3.670 | 14.191 | 72.64 | - |
| ShuffleNet | CUTLASS fresh | 7.088 | 6.187 | 15.496 | 72.63 | direct=1919, standalone=2828, fallback=909 |
| ShuffleNet | CUTLASS test clean | 7.204 | 6.218 | 15.387 | 72.63 | direct=1919, standalone=2828, fallback=909 |

## Tail duplication comparison

| Model | Clean CUTLASS test mean ms | Tail-dup mean ms | Ratio | Delta |
| --- | ---: | ---: | ---: | ---: |
| ResNet | 6.760 | 19.314 | 2.86x | +185.7% |
| MobileNet | 13.144 | 28.895 | 2.20x | +119.8% |
| ShuffleNet | 7.204 | 16.249 | 2.26x | +125.5% |

Because the CUTLASS tree still prints on the `--tail-dup` path, these ratios are dominated by the debug-printing path and are not a clean measurement of sequential duplicated-HMMA compute overhead.

## Fresh vs test clean

| Model | CUTLASS fresh mean ms | CUTLASS test clean mean ms | Delta |
| --- | ---: | ---: | ---: |
| ResNet | 6.290 | 6.760 | +7.5% |
| MobileNet | 13.009 | 13.144 | +1.0% |
| ShuffleNet | 7.088 | 7.204 | +1.6% |

## Commands used

Clean matrix:

```bash
OUT=runtime_results_h100_20260531_redo_clean
BUILD=/tmp/ml_bench_h100_20260531_redo_clean
DATA=/storage/ice1/0/3/yjia305/.data
PY=/storage/ice1/0/3/yjia305/model_env/bin/python
export PATH=/storage/ice1/0/3/yjia305/model_env/bin:$PATH
export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=1

for model in resnet mobilenet shufflenet; do
  extra=()
  if [ "$model" = shufflenet ]; then extra=(--no-strict); fi

  "$PY" -B benchmark_runtime.py --setup pytorch_original --model "$model" \
    --warmup 1 --data-root "$DATA" --output-dir "$OUT" "${extra[@]}"

  "$PY" -B benchmark_runtime.py --setup custom_conv_fresh --model "$model" \
    --warmup 1 --cutlass-dir /storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_fresh \
    --build-root "$BUILD" --data-root "$DATA" --output-dir "$OUT" "${extra[@]}"

  "$PY" -B benchmark_runtime.py --setup custom_conv_cutlass_test --model "$model" \
    --warmup 1 --cutlass-dir /storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_test \
    --build-root "$BUILD" --data-root "$DATA" --output-dir "$OUT" "${extra[@]}"
done
```

Tail duplication matrix:

```bash
OUT=runtime_results_h100_20260531_redo_taildup
BUILD=/tmp/ml_bench_h100_20260531_redo_taildup
DATA=/storage/ice1/0/3/yjia305/.data
PY=/storage/ice1/0/3/yjia305/model_env/bin/python
export PATH=/storage/ice1/0/3/yjia305/model_env/bin:$PATH
export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=1

for model in resnet mobilenet shufflenet; do
  extra=()
  if [ "$model" = shufflenet ]; then extra=(--no-strict); fi

  "$PY" -B benchmark_runtime.py --setup custom_conv_cutlass_test --model "$model" \
    --warmup 1 --tail-dup \
    --cutlass-dir /storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_test \
    --build-root "$BUILD" --data-root "$DATA" --output-dir "$OUT" "${extra[@]}"
done
```

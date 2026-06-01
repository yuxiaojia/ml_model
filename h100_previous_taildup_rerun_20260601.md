# H100 Previous Taildup Rerun

Rerun target:

```text
/storage/ice1/0/3/yjia305/2026_SUMMER/cutlass_test
commit: 2fefaa45 Add standalone CUTLASS Conv2D extension and pipelined tail check
source branch: taildup-debug-restart
```

Benchmark shape:

```text
GPU: NVIDIA H100 80GB HBM3
batch size: 100
timed batches: 100
samples: 10,000
benchmark: benchmark_runtime.py
timing: CUDA events
```

The rerun used `--tail-dup` and did not use `--tail-dup-print`.
ShuffleNet used `--no-strict`, matching the earlier run.

## CUDA Event Timing

Mean ms/batch.

| model | clean cutlass_test | previous taildup old run | previous taildup rerun | rerun overhead vs clean | rerun vs old run |
|---|---:|---:|---:|---:|---:|
| resnet | 6.760 | 19.271 | 19.264 | +184.99% | -0.03% |
| mobilenet | 13.144 | 28.802 | 28.922 | +120.04% | +0.42% |
| shufflenet | 7.204 | 16.092 | 15.940 | +121.27% | -0.95% |

Sources:

```text
clean cutlass_test: runtime_results_h100_20260531_redo_clean/
previous taildup old run: runtime_results_h100_20260531_taildup_cuda_event_1747/
previous taildup rerun: runtime_results_h100_20260601_previous_taildup_rerun_cuda_event/
```

## Coverage

| model | standalone calls | direct calls | fallback calls | strict |
|---|---:|---:|---:|---|
| resnet | 2121 | 0 | 0 | true |
| mobilenet | 3535 | 1717 | 0 | true |
| shufflenet | 2828 | 1919 | 909 | false |

## Notes

The rerun reproduces the high overhead from the earlier previous-taildup measurement.
The old taildup path still printed `IMPLICIT_GEMM_PIPELINED_TAIL_CHECK_V2` diagnostics during the rerun even though `tail_dup_print=False`, so the measured overhead includes that behavior from the old checkout.

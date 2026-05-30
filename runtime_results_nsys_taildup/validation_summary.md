# Tail-duplicate print validation

Command profile:

```text
nsys profile --trace=cuda,nvtx,osrt ... python -B benchmark_runtime.py --setup custom_conv_cutlass_test --model resnet --batches 1 --warmup 0 --tail-dup-print ...
```

Validation result:

```text
IMPLICIT_GEMM_PIPELINED_TAIL_CHECK_V2 lines: 21
runtime_config.cutlass_dir: /net/netscratch/yjia305/cutlass_test
runtime_config.extension_name: torch_cutlass_test_conv_ext_v1_taildup_print
runtime_config.tail_duplicate_print: True
usage_stats.standalone_calls: 21
usage_stats.fallback_calls: 0
```

This run is for path validation only. CUDA device `printf` changes runtime, so
the non-print `runtime_results_nsys/` profiles should be used for performance.

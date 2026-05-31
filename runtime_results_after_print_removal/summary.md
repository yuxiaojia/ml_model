# Runtime comparison after CUTLASS print removal

`cutlass_test` HEAD: `0acc588f comment out printing`

Plain CUDA-event timing, no Nsight profiler. Full CIFAR-100 test loader: 100 timed batches, batch size 100. `tail_dup_print` is false for all rows.

| model | setup | mean ms | median ms | p95 ms | overhead vs pytorch | standalone | direct | fallback | strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet | pytorch_original | 3.8127 | 3.7743 | 3.9168 | +0.00% |  |  |  | True |
| resnet | custom_conv_fresh | 6.0850 | 5.8565 | 6.5712 | +59.60% | 2121 | 0 | 0 | True |
| resnet | custom_conv_cutlass_test | 6.1149 | 6.1303 | 6.4969 | +60.39% | 2121 | 0 | 0 | True |
| mobilenet | pytorch_original | 9.8582 | 8.0256 | 20.2978 | +0.00% |  |  |  | True |
| mobilenet | custom_conv_fresh | 13.5354 | 12.5278 | 21.7415 | +37.30% | 3535 | 1717 | 0 | True |
| mobilenet | custom_conv_cutlass_test | 13.8628 | 12.5744 | 21.8228 | +40.62% | 3535 | 1717 | 0 | True |
| shufflenet | pytorch_original | 9.5514 | 9.3016 | 10.7696 | +0.00% |  |  |  | True |
| shufflenet | custom_conv_fresh | 14.8372 | 12.4650 | 25.4680 | +55.34% | 2828 | 1919 | 909 | False |
| shufflenet | custom_conv_cutlass_test | 13.7156 | 12.2468 | 23.1052 | +43.60% | 2828 | 1919 | 909 | False |

Notes:
- `custom_conv_cutlass_test` was rebuilt from `/net/netscratch/yjia305/cutlass_test` at commit `0acc588f comment out printing` using `/tmp/ml_bench_after_print_removal_build`.
- ShuffleNet custom runs use `--no-strict` so the full 100 batches complete; strict mode still fails on CUTLASS `Error Misaligned Operand`.

# ResNet Nsight Systems summary

Full CIFAR-100 test split: 100 timed batches, batch size 100. Timings below are from Nsight-profiled runs, so they include profiler overhead and should be compared within this table only.

| label | setup | event mean ms | NVTX batch mean ms | GPU kernel total ms | CUTLASS conv total ms | CUTLASS conv calls | overhead vs pytorch, NVTX | fallbacks | report |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pytorch_original | pytorch_original | 7.0967 | 7.2001 | 163.9420 | 0.0000 | 0 | +0.00% |  | resnet_pytorch_original.nsys-rep |
| custom_conv_fresh | custom_conv_fresh | 8.0795 | 8.1074 | 605.0680 | 454.3902 | 2121 | +12.60% | 0 | resnet_custom_conv_fresh.nsys-rep |
| custom_conv_cutlass_test_rerun | custom_conv_cutlass_test | 7.8840 | 8.0288 | 582.6004 | 438.1169 | 2121 | +11.51% | 0 | resnet_custom_conv_cutlass_test_rerun.nsys-rep |
| custom_conv_cutlass_test_first | custom_conv_cutlass_test | 9.8178 | 9.8788 | 456.1795 | 342.8477 | 2121 | +37.20% | 0 | resnet_custom_conv_cutlass_test.nsys-rep |

Validation notes:
- `custom_conv_cutlass_test_rerun` is the primary modified-CUTLASS ResNet profile; `custom_conv_cutlass_test_first` is kept as the first trace and was noisier at the NVTX batch level.
- `custom_conv_cutlass_test_rerun` used `/net/netscratch/yjia305/cutlass_test` with `tail_duplicate_print=False`, strict mode, 2121 standalone calls, and 0 fallbacks.
- Separate tail-dup validation is in `../runtime_results_nsys_taildup/resnet_custom_conv_cutlass_test_taildup.log`; it has 21 `IMPLICIT_GEMM_PIPELINED_TAIL_CHECK_V2` lines for one batch, all showing `mismatch=0` in the sampled lines.
- Tail-dup print mode is intentionally not used for the performance table because CUDA device `printf` changes runtime substantially.

# CTA Duplicate Compile Smoke Results

These are one-batch compile/run smoke tests for `/net/netscratch/yjia305/cutlass_intra_warp` with `-DCUTLASS_ENABLE_IMPLICIT_GEMM_CTA_DUPLICATE=1`, exposed through `benchmark_runtime.py --cta-dup`.

Device: NVIDIA A100-PCIE-40GB on Slurm job 137409 / node frozone2.

| model | mean ms/batch | batches | samples | standalone calls | direct | fallback | strict | cta_dup | runtime cta_duplicate |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| resnet | 56503.051 | 1 | 100 | 21 | 0 | 0 | true | true | True |
| mobilenet | 55906.801 | 1 | 100 | 35 | 17 | 0 | true | true | True |
| shufflenet | 55999.754 | 1 | 100 | 28 | 19 | 9 | false | true | True |

Validation notes:
- All rows compiled extension `torch_cutlass_test_conv_ext_v1_ctadup`.
- All rows report `cutlass_dir=/net/netscratch/yjia305/cutlass_intra_warp`, `cta_dup=true`, and runtime config `cta_duplicate=True`.
- The CUDA compile line included `-DCUTLASS_ENABLE_IMPLICIT_GEMM_CTA_DUPLICATE=1` in the verbose ResNet build.
- These timings are intentionally not performance-clean: this CTA duplicate implementation calls `cudaMalloc/cudaFree` and synchronizes inside `run()`, so one batch takes about 56 seconds.

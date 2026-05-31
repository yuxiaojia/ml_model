# Corrected CUTLASS Test Taildup Runtime Comparison

This is the corrected run using `/net/netscratch/yjia305/cutlass_test` with `--tail-dup`. That tree contains the actual `tail_duplicate_accum` code and `CUTLASS_ENABLE_*_TAIL_DUPLICATE` macro guards. The previous `intra_warp_taildup` rows used `/net/netscratch/yjia305/cutlass_intra_warp`, which currently differs from `origin/main` only by `INTRA_WARP_TAIL_DUP_README.md`, so those earlier rows should not be treated as proof of duplicate-HMMA execution.

`cutlass_test`: `taildup-debug-restart`, `0acc588f comment out printing`
`cutlass_intra_warp`: `intra-warp-duplication`, diff vs `origin/main`: `INTRA_WARP_TAIL_DUP_README.md`

Device: NVIDIA A100-PCIE-40GB on Slurm job 136969 / node frozone2.

| model | normal cutlass_test ms/batch | wrong intra_warp row ms/batch | corrected cutlass_test taildup ms/batch | corrected overhead vs normal | wrong overhead vs normal | batches | samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet | 5.847 | 6.069 | 10.389 | +77.69% | +3.79% | 100 | 10000 |
| mobilenet | 13.974 | 14.262 | 18.695 | +33.79% | +2.06% | 100 | 10000 |
| shufflenet | 13.559 | 13.698 | 14.965 | +10.37% | +1.03% | 100 | 10000 |

Validation: all corrected rows report `cutlass_dir=/net/netscratch/yjia305/cutlass_test`, extension `torch_cutlass_test_conv_ext_v1_taildup`, `tail_dup=true`, `tail_dup_print=false`, runtime `tail_duplicate=True`, and 100 batches / 10,000 samples.

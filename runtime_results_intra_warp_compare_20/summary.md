# Intra-warp tail-dup 20-batch smoke comparison

This is a 20-batch GPU smoke benchmark for the new `/net/netscratch/yjia305/cutlass_intra_warp` tree. It uses `--tail-dup-print` only to enable the historical bridge macro path; the intra-warp tree disables CUDA device printf by default, so these timings are not dominated by print output.

| model | variant | mean ms | median ms | p95 ms | overhead vs cutlass_test | standalone | direct | fallback | tail flag | strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet | fresh | 6.6619 | 6.4893 | 7.6421 | +1.83% | 441 | 0 | 0 | False | True |
| resnet | cutlass_test | 6.5424 | 6.4655 | 6.9933 | +0.00% | 441 | 0 | 0 | False | True |
| resnet | intra_warp_taildup | 12.4177 | 12.7922 | 13.0932 | +89.80% | 441 | 0 | 0 | True | True |
| mobilenet | fresh | 18.3708 | 17.5374 | 23.1419 | +0.60% | 735 | 357 | 0 | False | True |
| mobilenet | cutlass_test | 18.2613 | 21.7854 | 21.8118 | +0.00% | 735 | 357 | 0 | False | True |
| mobilenet | intra_warp_taildup | 21.0337 | 17.5901 | 29.2672 | +15.18% | 735 | 357 | 0 | True | True |
| shufflenet | fresh | 14.1448 | 12.7208 | 19.9858 | -4.32% | 588 | 399 | 189 | False | False |
| shufflenet | cutlass_test | 14.7836 | 12.4116 | 24.9499 | +0.00% | 588 | 399 | 189 | False | False |
| shufflenet | intra_warp_taildup | 14.8696 | 13.3241 | 20.2235 | +0.58% | 588 | 399 | 189 | True | False |

Key checks:
- ResNet intra-warp run: 441 standalone calls, 0 fallbacks over 20 batches.
- MobileNet intra-warp run: 735 standalone calls, 357 direct calls, 0 fallbacks over 20 batches.
- ShuffleNet intra-warp run: 588 standalone calls, 399 direct calls, 189 fallbacks over 20 batches; it uses `--no-strict` like the prior custom ShuffleNet runs.
- Intra-warp rows use `cutlass_dir=/net/netscratch/yjia305/cutlass_intra_warp` and `extension_name=torch_cutlass_test_conv_ext_v1_taildup_print`.

This is not yet a full 100-batch/repeated benchmark; it is the requested 20-batch build/run validation plus matched 20-batch comparison.

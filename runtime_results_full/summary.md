# ml_bench runtime summary

| model | setup | strict | batches | mean ms | median ms | p95 ms | overhead vs pytorch | fallback calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mobilenet | custom_conv_cutlass_test | True | 100 | 13.4653 | 12.5333 | 21.7328 | +57.26% | 0 |
| mobilenet | custom_conv_fresh | True | 100 | 14.3641 | 12.5628 | 21.4373 | +67.75% | 0 |
| mobilenet | pytorch_original | True | 100 | 8.5625 | 8.1006 | 10.1714 | +0.00% |  |
| resnet | custom_conv_cutlass_test | True | 100 | 5.9546 | 5.7131 | 6.4996 | +57.97% | 0 |
| resnet | custom_conv_fresh | True | 100 | 6.2137 | 5.8948 | 6.4701 | +64.85% | 0 |
| resnet | pytorch_original | True | 100 | 3.7694 | 3.7185 | 3.9776 | +0.00% |  |
| shufflenet | custom_conv_cutlass_test | False | 100 | 13.4791 | 12.4245 | 20.9375 | +14.78% | 909 |
| shufflenet | custom_conv_fresh | False | 100 | 13.7161 | 12.3352 | 24.7561 | +16.80% | 909 |
| shufflenet | pytorch_original | True | 100 | 11.7436 | 9.4316 | 22.0238 | +0.00% |  |

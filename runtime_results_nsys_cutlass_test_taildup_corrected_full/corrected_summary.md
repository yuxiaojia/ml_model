# Corrected CUTLASS Test Taildup Nsight Results

Profiles used `nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none` for the corrected `/net/netscratch/yjia305/cutlass_test --tail-dup` rows.

| model | corrected taildup NVTX ms/batch | corrected event ms under nsys | baseline event ms under nsys | event overhead vs baseline | GPU kernel total ms | standalone conv total ms | standalone conv launches | batches | samples | report |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| resnet | 8.9964 | 10.9899 | 8.3493 | +31.63% | 910.3088 | 807.2341 | 2121 | 100 | 10000 | resnet_custom_conv_cutlass_test_taildup.nsys-rep |
| mobilenet | 17.6471 | 20.3001 | 18.4078 | +10.28% | 1837.2051 | 873.2092 | 3535 | 100 | 10000 | mobilenet_custom_conv_cutlass_test_taildup.nsys-rep |
| shufflenet | 21.3859 | 21.3949 | 21.4132 | -0.09% | 849.8820 | 381.7486 | 2828 | 100 | 10000 | shufflenet_custom_conv_cutlass_test_taildup.nsys-rep |

Validation: all corrected Nsight JSON rows report `tail_dup=true`, `tail_dup_print=false`, and 100 timed batches / 10,000 samples.

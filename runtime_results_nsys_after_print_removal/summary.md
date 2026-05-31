# Nsight Systems comparison after CUTLASS print removal

`cutlass_test` HEAD: `0acc588f comment out printing`

Full CIFAR-100 test loader: 100 timed batches, batch size 100. These runs are profiled with `nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none`; compare within this table because Nsight adds overhead.

| model | setup | NVTX batch mean ms | event mean ms under nsys | GPU kernel total ms | standalone conv total ms | standalone | direct | fallback | overhead vs pytorch, NVTX | report |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| resnet | pytorch_original | 6.1824 | 6.0948 | 148.6564 | 0.0000 |  |  |  | +0.00% | resnet_pytorch_original.nsys-rep |
| resnet | custom_conv_fresh | 8.0220 | 7.9922 | 601.0516 | 452.2307 | 2121 | 0 | 0 | +29.76% | resnet_custom_conv_fresh.nsys-rep |
| resnet | custom_conv_cutlass_test | 7.7649 | 7.7363 | 622.1579 | 467.5413 | 2121 | 0 | 0 | +25.60% | resnet_custom_conv_cutlass_test.nsys-rep |
| mobilenet | pytorch_original | 15.7493 | 15.6351 | 1087.0069 | 0.0000 |  |  |  | +0.00% | mobilenet_pytorch_original.nsys-rep |
| mobilenet | custom_conv_fresh | 18.0873 | 18.4380 | 1309.9231 | 344.6787 | 3535 | 1717 | 0 | +14.85% | mobilenet_custom_conv_fresh.nsys-rep |
| mobilenet | custom_conv_cutlass_test | 18.0532 | 18.6177 | 1342.1874 | 354.0010 | 3535 | 1717 | 0 | +14.63% | mobilenet_custom_conv_cutlass_test.nsys-rep |
| shufflenet | pytorch_original | 17.0724 | 16.8725 | 486.7205 | 0.0000 |  |  |  | +0.00% | shufflenet_pytorch_original.nsys-rep |
| shufflenet | custom_conv_fresh | 22.0976 | 22.0416 | 823.7535 | 247.0719 | 2828 | 1919 | 909 | +29.43% | shufflenet_custom_conv_fresh.nsys-rep |
| shufflenet | custom_conv_cutlass_test | 21.6805 | 21.6216 | 836.5365 | 250.3664 | 2828 | 1919 | 909 | +26.99% | shufflenet_custom_conv_cutlass_test.nsys-rep |

Notes:
- All custom rows have `tail_dup_print=False`; the removed-print `cutlass_test` build is measured without device printf overhead.
- SQLite exports are generated locally from `.nsys-rep` for this summary and can be regenerated with `nsys stats`; they are not intended as the primary artifact.

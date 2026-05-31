# Nsight Systems Decoupled Tail-Duplicate Full Results

Full CIFAR test loader: 100 timed batches, batch size 100, 10,000 images. Profiles used `nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none`. Compare rows within this table because Nsight adds overhead.

Device: NVIDIA A100-PCIE-40GB on Slurm job 136966 / node frozone2.

| model | variant | NVTX mean ms/batch | event mean ms under nsys | GPU kernel total ms | standalone conv total ms | standalone conv launches | overhead vs cutlass_test, NVTX | tail_dup | tail_dup_print | batches | samples | report |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| resnet | pytorch_original | 6.5299 | 6.4351 | 159.4968 | 0.0000 | 0 | -22.34% | false | false | 100 | 10000 | resnet_pytorch_original.nsys-rep |
| resnet | custom_conv_fresh | 8.3244 | 8.2683 | 578.9779 | 436.0023 | 2121 | -1.00% | false | false | 100 | 10000 | resnet_custom_conv_fresh.nsys-rep |
| resnet | custom_conv_cutlass_test | 8.4085 | 8.3493 | 586.3900 | 441.5171 | 2121 | +0.00% | false | false | 100 | 10000 | resnet_custom_conv_cutlass_test.nsys-rep |
| resnet | intra_warp_taildup | 8.9078 | 8.7430 | 613.7635 | 463.0862 | 2121 | +5.94% | true | false | 100 | 10000 | resnet_intra_warp_taildup.nsys-rep |
| mobilenet | pytorch_original | 14.9329 | 14.8258 | 1063.4564 | 0.0000 | 0 | -16.77% | false | false | 100 | 10000 | mobilenet_pytorch_original.nsys-rep |
| mobilenet | custom_conv_fresh | 19.0045 | 19.0991 | 1245.0857 | 325.7912 | 3535 | +5.93% | false | false | 100 | 10000 | mobilenet_custom_conv_fresh.nsys-rep |
| mobilenet | custom_conv_cutlass_test | 17.9413 | 18.4078 | 1313.5841 | 345.5699 | 3535 | +0.00% | false | false | 100 | 10000 | mobilenet_custom_conv_cutlass_test.nsys-rep |
| mobilenet | intra_warp_taildup | 17.6525 | 18.1875 | 1333.2633 | 351.2132 | 3535 | -1.61% | true | false | 100 | 10000 | mobilenet_intra_warp_taildup.nsys-rep |
| shufflenet | pytorch_original | 17.8291 | 17.7085 | 470.8723 | 0.0000 | 0 | -16.98% | false | false | 100 | 10000 | shufflenet_pytorch_original.nsys-rep |
| shufflenet | custom_conv_fresh | 21.7880 | 21.7216 | 808.5775 | 243.5482 | 2828 | +1.46% | false | false | 100 | 10000 | shufflenet_custom_conv_fresh.nsys-rep |
| shufflenet | custom_conv_cutlass_test | 21.4750 | 21.4132 | 824.2023 | 248.1427 | 2828 | +0.00% | false | false | 100 | 10000 | shufflenet_custom_conv_cutlass_test.nsys-rep |
| shufflenet | intra_warp_taildup | 21.9826 | 21.9118 | 619.0053 | 185.7020 | 2828 | +2.36% | true | false | 100 | 10000 | shufflenet_intra_warp_taildup.nsys-rep |

Validation notes:
- All JSON outputs and NVTX traces contain 100 timed batches / 10,000 samples.
- Intra-warp rows use `--tail-dup` only: `tail_dup=true`, `tail_dup_print=false`, runtime config `tail_duplicate=True`, extension `_taildup`.
- Normal `custom_conv_cutlass_test` rows keep `tail_duplicate=False`; this is the clean comparison baseline.
- `standalone_cutlass_conv_count` matches `usage_stats.standalone_calls` for custom conv rows, confirming the Nsight kernel count lines up with the wrapper counters.

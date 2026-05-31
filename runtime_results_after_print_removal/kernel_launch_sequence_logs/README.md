# Kernel launch sequence logs

These files are one row per CUDA kernel launch, ordered by Nsight start timestamp. They are not grouped by kernel name.

## resnet

- fresh launch rows: `16459`
- cutlass_test launch rows: `16459`
- sequence name diffs: `0`
- fresh sequence log: `resnet_custom_conv_fresh_kernel_launch_sequence.csv.gz`
- cutlass_test sequence log: `resnet_custom_conv_cutlass_test_kernel_launch_sequence.csv.gz`
- sequence diff log: `resnet_fresh_vs_cutlass_test_launch_sequence_diff.csv`

## mobilenet

- fresh launch rows: `35346`
- cutlass_test launch rows: `35346`
- sequence name diffs: `0`
- fresh sequence log: `mobilenet_custom_conv_fresh_kernel_launch_sequence.csv.gz`
- cutlass_test sequence log: `mobilenet_custom_conv_cutlass_test_kernel_launch_sequence.csv.gz`
- sequence diff log: `mobilenet_fresh_vs_cutlass_test_launch_sequence_diff.csv`

## shufflenet

- fresh launch rows: `39833`
- cutlass_test launch rows: `39833`
- sequence name diffs: `0`
- fresh sequence log: `shufflenet_custom_conv_fresh_kernel_launch_sequence.csv.gz`
- cutlass_test sequence log: `shufflenet_custom_conv_cutlass_test_kernel_launch_sequence.csv.gz`
- sequence diff log: `shufflenet_fresh_vs_cutlass_test_launch_sequence_diff.csv`


To inspect a compressed launch log:

```bash
gzip -dc resnet_custom_conv_fresh_kernel_launch_sequence.csv.gz | head
```

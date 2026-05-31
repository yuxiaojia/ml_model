# Kernel launch sequence logs

These files are one row per CUDA kernel launch, ordered by Nsight start timestamp. They are not grouped by kernel name.

Each `*_kernel_launch_sequence.csv.gz` has:

```text
launch_index,start_ns,end_ns,duration_us,kernel_name
```

## resnet

- pytorch_original launch rows: `16330`
- custom_conv_fresh launch rows: `16459`
- custom_conv_cutlass_test launch rows: `16459`
- fresh_vs_cutlass_test sequence name diffs: `0` (`resnet_fresh_vs_cutlass_test_launch_sequence_diff.csv`)
- pytorch_vs_fresh sequence name diffs: `16088` (`resnet_pytorch_vs_fresh_launch_sequence_diff.csv.gz`)
- pytorch_vs_cutlass_test sequence name diffs: `16088` (`resnet_pytorch_vs_cutlass_test_launch_sequence_diff.csv.gz`)
- pytorch sequence log: `resnet_pytorch_original_kernel_launch_sequence.csv.gz`
- fresh sequence log: `resnet_custom_conv_fresh_kernel_launch_sequence.csv.gz`
- cutlass_test sequence log: `resnet_custom_conv_cutlass_test_kernel_launch_sequence.csv.gz`

## mobilenet

- pytorch_original launch rows: `49704`
- custom_conv_fresh launch rows: `35346`
- custom_conv_cutlass_test launch rows: `35346`
- fresh_vs_cutlass_test sequence name diffs: `0` (`mobilenet_fresh_vs_cutlass_test_launch_sequence_diff.csv`)
- pytorch_vs_fresh sequence name diffs: `49387` (`mobilenet_pytorch_vs_fresh_launch_sequence_diff.csv.gz`)
- pytorch_vs_cutlass_test sequence name diffs: `49387` (`mobilenet_pytorch_vs_cutlass_test_launch_sequence_diff.csv.gz`)
- pytorch sequence log: `mobilenet_pytorch_original_kernel_launch_sequence.csv.gz`
- fresh sequence log: `mobilenet_custom_conv_fresh_kernel_launch_sequence.csv.gz`
- cutlass_test sequence log: `mobilenet_custom_conv_cutlass_test_kernel_launch_sequence.csv.gz`

## shufflenet

- pytorch_original launch rows: `32320`
- custom_conv_fresh launch rows: `39833`
- custom_conv_cutlass_test launch rows: `39833`
- fresh_vs_cutlass_test sequence name diffs: `0` (`shufflenet_fresh_vs_cutlass_test_launch_sequence_diff.csv`)
- pytorch_vs_fresh sequence name diffs: `37185` (`shufflenet_pytorch_vs_fresh_launch_sequence_diff.csv.gz`)
- pytorch_vs_cutlass_test sequence name diffs: `37185` (`shufflenet_pytorch_vs_cutlass_test_launch_sequence_diff.csv.gz`)
- pytorch sequence log: `shufflenet_pytorch_original_kernel_launch_sequence.csv.gz`
- fresh sequence log: `shufflenet_custom_conv_fresh_kernel_launch_sequence.csv.gz`
- cutlass_test sequence log: `shufflenet_custom_conv_cutlass_test_kernel_launch_sequence.csv.gz`

To inspect a compressed launch log:

```bash
gzip -dc resnet_pytorch_original_kernel_launch_sequence.csv.gz | head
```

To inspect a compressed PyTorch-vs-custom diff:

```bash
gzip -dc resnet_pytorch_vs_fresh_launch_sequence_diff.csv.gz | head
```

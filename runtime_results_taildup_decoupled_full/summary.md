# Decoupled Tail-Duplicate Full Runtime Results

Command shape: `python -B benchmark_runtime.py --warmup 1` with `--batches` omitted, so each row is the full CIFAR test set: 100 timed batches at batch size 100, 10,000 images total. The intra-warp rows use `--tail-dup` only, not `--tail-dup-print`.

Device: NVIDIA A100-PCIE-40GB on Slurm job 136966 / node frozone2.

| model | variant | mean ms/batch | total forward ms | overhead vs cutlass_test | overhead vs fresh | tail_dup | tail_dup_print | runtime tail_duplicate | batches | samples |
|---|---|---:|---:|---:|---:|---|---|---|---:|---:|
| resnet | pytorch_original | 3.932 | 393.201 | -32.75% | -16.87% | false | false |  | 100 | 10000 |
| resnet | custom_conv_fresh | 4.730 | 472.994 | -19.10% | +0.00% | false | false | False | 100 | 10000 |
| resnet | custom_conv_cutlass_test | 5.847 | 584.682 | +0.00% | +23.61% | false | false | False | 100 | 10000 |
| resnet | intra_warp_taildup | 6.069 | 606.864 | +3.79% | +28.30% | true | false | True | 100 | 10000 |
| mobilenet | pytorch_original | 9.753 | 975.334 | -30.20% | -28.58% | false | false |  | 100 | 10000 |
| mobilenet | custom_conv_fresh | 13.656 | 1365.598 | -2.28% | +0.00% | false | false | False | 100 | 10000 |
| mobilenet | custom_conv_cutlass_test | 13.974 | 1397.393 | +0.00% | +2.33% | false | false | False | 100 | 10000 |
| mobilenet | intra_warp_taildup | 14.262 | 1426.206 | +2.06% | +4.44% | true | false | True | 100 | 10000 |
| shufflenet | pytorch_original | 10.055 | 1005.514 | -25.84% | -28.75% | false | false |  | 100 | 10000 |
| shufflenet | custom_conv_fresh | 14.113 | 1411.266 | +4.08% | +0.00% | false | false | False | 100 | 10000 |
| shufflenet | custom_conv_cutlass_test | 13.559 | 1355.944 | +0.00% | -3.92% | false | false | False | 100 | 10000 |
| shufflenet | intra_warp_taildup | 13.698 | 1369.850 | +1.03% | -2.93% | true | false | True | 100 | 10000 |

Validation notes:
- All rows report `timed_batches=100` and `samples=10000`.
- Normal `custom_conv_cutlass_test` rows report `tail_duplicate=False`.
- `intra_warp_taildup` rows report extension `torch_cutlass_test_conv_ext_v1_taildup`, `tail_dup=true`, `tail_dup_print=false`, and runtime config `tail_duplicate=True`, so this is the decoupled duplicate-HMMA/check path without device printf.
- ShuffleNet custom conv rows run with `strict=false`; the other rows run with strict mode enabled.

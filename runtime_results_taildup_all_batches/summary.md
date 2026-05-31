# Tail-duplicate all-batch validation

This validates the modified `cutlass_test` Conv2D path with `--tail-dup-print` over the full CIFAR-100 test loader. These runtimes include CUDA device `printf` overhead and should not be used as performance numbers.

| model | timed batches | samples | tail-check prints | standalone calls | direct calls | fallback calls | mean ms with printf | strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet | 100 | 10000 | 2100 | 2100 | 0 | 0 | 587.3058 | True |
| mobilenet | 100 | 10000 | 3500 | 3500 | 1700 | 0 | 589.4044 | True |
| shufflenet | 100 | 10000 | 2800 | 2800 | 1900 | 900 | 618.4166 | False |

All three runs used:

```text
cutlass_dir=/net/netscratch/yjia305/cutlass_test
extension_name=torch_cutlass_test_conv_ext_v1_taildup_print
tail_duplicate_print=True
```

ShuffleNet was run with `--no-strict` because strict mode fails on a CUTLASS `Error Misaligned Operand`; it completed all 100 batches with fallbacks reported above.

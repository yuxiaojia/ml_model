# Fresh vs cutlass_test kernel/fallback check

This check compares `custom_conv_fresh` and `custom_conv_cutlass_test` from the
post-print-removal run.

`cutlass_test` HEAD:

```text
0acc588f comment out printing
```

## Fallback and call counters

The model-level bridge counters are exactly the same between the fresh and
`cutlass_test` custom-conv runs:

| model | setup | standalone calls | direct calls | fallback calls | strict |
|---|---|---:|---:|---:|---:|
| resnet | custom_conv_fresh | 2121 | 0 | 0 | True |
| resnet | custom_conv_cutlass_test | 2121 | 0 | 0 | True |
| mobilenet | custom_conv_fresh | 3535 | 1717 | 0 | True |
| mobilenet | custom_conv_cutlass_test | 3535 | 1717 | 0 | True |
| shufflenet | custom_conv_fresh | 2828 | 1919 | 909 | False |
| shufflenet | custom_conv_cutlass_test | 2828 | 1919 | 909 | False |

ShuffleNet uses `--no-strict` in both custom runs so all 100 batches complete.
Both fresh and `cutlass_test` have exactly 909 fallbacks.

## Nsight CUDA kernel names and counts

From the generated Nsight SQLite exports:

| model | distinct CUDA kernel names, fresh | distinct CUDA kernel names, cutlass_test | total kernel instances, fresh | total kernel instances, cutlass_test | name/count diff |
|---|---:|---:|---:|---:|---:|
| resnet | 13 | 13 | 16459 | 16459 | 0 |
| mobilenet | 15 | 15 | 35346 | 35346 | 0 |
| shufflenet | 36 | 36 | 39833 | 39833 | 0 |

The CUDA kernel name set and per-name instance counts are identical for fresh
and `cutlass_test` for all three models.

Standalone CUTLASS Conv2D kernel instances:

| model | fresh instances | cutlass_test instances | fresh total ms | cutlass_test total ms |
|---|---:|---:|---:|---:|
| resnet | 2121 | 2121 | 452.2307 | 467.5413 |
| mobilenet | 3535 | 3535 | 344.6787 | 354.0010 |
| shufflenet | 2828 | 2828 | 247.0719 | 250.3664 |

## Binary identity

The compiled extension binaries are not byte-identical. That is expected because
the same extension source is built against different CUTLASS include trees:

```text
/net/netscratch/yjia305/cutlass_fresh/include
/net/netscratch/yjia305/cutlass_test/include
```

SHA256 of the compiled `.so` files:

| model | setup | sha256 |
|---|---|---|
| resnet | custom_conv_fresh | c08c368a94a809173657b965b64a0e5bf1ef470a08787c4ed43c9cee8699dfef |
| resnet | custom_conv_cutlass_test | 26f8bbc7a80194542fd2fa02a6fea6d27c7639c9a42c08c9527f7faf2a4ddaf8 |
| mobilenet | custom_conv_fresh | 12a243debc3017094e5fe0c0eed776fdbb21f70895b42bbcfed6469c87d22add |
| mobilenet | custom_conv_cutlass_test | b3db9a6f201623e95af395e37ec806eef88b579adc6e59e890d9332d78a46962 |
| shufflenet | custom_conv_fresh | d821c58f502ab4b4bd7fe00114965fb57ef60bcd290535dabcc5e4b1ea98779b |
| shufflenet | custom_conv_cutlass_test | 95462d8ccb256056f34990d83623b05ec766417cc56f2b5d8324913534e58529 |

Conclusion: fresh and `cutlass_test` have identical fallback behavior and
identical Nsight CUDA kernel names/counts, but they are not the exact same
compiled binary implementation.

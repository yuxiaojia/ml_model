# ML Bench

This folder has three separate benchmark trees with the same four model
families:

```text
pytorch_original/          # original PyTorch model path from new_test examples
custom_conv_fresh/         # custom Conv2d bridge backed by cutlass_fresh
custom_conv_cutlass_test/  # custom Conv2d bridge backed by cutlass_test
```

The copied trees are based on:

```text
/nethome/yjia305/USERSCRATCH/new_test/nvbit_release_x86_64/TensorDynamic_own_version/examples
```

Generated caches/logs and the old tensor-dump `new_example/` folder were removed.

## Run Original PyTorch

```bash
cd /nethome/yjia305/USERSCRATCH/ml_bench

./run_original.sh resnet
./run_original.sh mobilenet
./run_original.sh shufflenet
./run_original.sh yolo
```

## Run Custom Conv With Baseline CUTLASS

```bash
cd /nethome/yjia305/USERSCRATCH/ml_bench

./run_custom_conv.sh resnet
./run_custom_conv.sh mobilenet
./run_custom_conv.sh shufflenet
./run_custom_conv.sh yolo
```

`run_custom_conv.sh` uses:

```text
/nethome/yjia305/USERSCRATCH/cutlass_fresh
```

## Run Custom Conv With Modified CUTLASS

```bash
cd /nethome/yjia305/USERSCRATCH/ml_bench

./run_custom_conv_cutlass_test.sh resnet
./run_custom_conv_cutlass_test.sh mobilenet
./run_custom_conv_cutlass_test.sh shufflenet
./run_custom_conv_cutlass_test.sh yolo
```

`run_custom_conv_cutlass_test.sh` uses:

```text
/nethome/yjia305/USERSCRATCH/cutlass_test
```

The two custom folders use the same bridge files and the same model scripts.
The intended difference is only the CUTLASS tree selected by
`CUTLASS_DIR` / `STANDALONE_CUTLASS_DIR`.

Each custom folder has a `sitecustomize.py` that enables the standalone custom
Conv2d path by default and wraps models loaded through `torch.hub.load`. YOLO is
handled by the copied `cutlass_only_config.configure_yolo_model_layout()`.

## Current CTA Duplicate Experiment

The current `/nethome/yjia305/USERSCRATCH/cutlass_intra_warp` branch implements
CTA/kernel-launch-level DMR in CUTLASS Conv2D. It is enabled from this benchmark
with:

```bash
python -B benchmark_runtime.py \
  --setup custom_conv_cutlass_test \
  --model resnet \
  --cutlass-dir /nethome/yjia305/USERSCRATCH/cutlass_intra_warp \
  --build-root /tmp/ml_bench_intra_warp_current_full \
  --output-dir runtime_results_intra_warp_current_full/cta_duplicate \
  --warmup 1 \
  --cta-dup
```

This is not warp-level `warp_mma` duplication. See
[`CTA_DUPLICATE_CURRENT_RESULTS.md`](CTA_DUPLICATE_CURRENT_RESULTS.md) for the
implementation details, exact commands, correctness check, CUDA-event overhead,
and Nsight results.
# ml_model

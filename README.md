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
# ml_model

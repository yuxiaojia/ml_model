# Current cutlass_intra_warp Duplicate Nsight Results

Nsight profiles used `nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none`. These are profiler-attached event timings; use the CUDA-event summary for the primary overhead claim.

| model | baseline event ms under nsys | duplicate event ms under nsys | overhead under nsys | batches | samples | reports |
|---|---:|---:|---:|---:|---:|---|
| resnet | 8.290 | 20.976 | +153.04% | 100 | 10000 | `resnet_baseline.nsys-rep`, `resnet_cta_duplicate.nsys-rep` |
| mobilenet | 18.698 | 35.452 | +89.60% | 100 | 10000 | `mobilenet_baseline.nsys-rep`, `mobilenet_cta_duplicate.nsys-rep` |
| shufflenet | 21.729 | 31.637 | +45.60% | 100 | 10000 | `shufflenet_baseline.nsys-rep`, `shufflenet_cta_duplicate.nsys-rep` |

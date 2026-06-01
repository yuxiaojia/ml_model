# Current cutlass_intra_warp Duplicate Runtime

CUTLASS tree: `/nethome/yjia305/USERSCRATCH/cutlass_intra_warp`, branch `intra-warp-duplication`, HEAD `b2f085c9 Add CTA duplicate implicit GEMM wrapper`.

Important: this tree currently does **not** contain warp-level `warp_mma` duplication in the threadblock mainloop. The active duplicate implementation is CTA/launch-level duplication in `include/cutlass/conv/device/implicit_gemm_convolution.h`, enabled through `benchmark_runtime.py --cta-dup`, which passes `-DCUTLASS_ENABLE_IMPLICIT_GEMM_CTA_DUPLICATE=1`.

CUDA-event timing, full CIFAR test set: 100 timed batches, batch size 100, 10,000 images. Warmup is 1 batch.

| model | baseline ms/batch | duplicate ms/batch | overhead | baseline total ms | duplicate total ms | standalone conv calls | direct | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet | 6.141 | 16.427 | +167.52% | 614.070 | 1642.750 | 2121 | 0 | 0 |
| mobilenet | 13.308 | 29.091 | +118.60% | 1330.811 | 2909.108 | 3535 | 1717 | 0 |
| shufflenet | 13.783 | 26.124 | +89.54% | 1378.290 | 2612.434 | 2828 | 1919 | 909 |

Implementation summary:
- The normal authoritative CUTLASS convolution launch writes the real output `ptr_D`.
- A shadow output buffer and mismatch flag are allocated with `cudaMalloc`.
- The same CUTLASS kernel is launched again with identical parameters except `ptr_D` points to the shadow buffer, on a separate CUDA stream when stream creation succeeds.
- The wrapper synchronizes both launches, runs an element-wise compare kernel over real output vs shadow output, copies the mismatch flag to host, then frees the scratch allocations.
- This preserves the authoritative GEMM output, but the measured overhead includes the second full kernel launch, scratch allocation/free, stream synchronization, compare kernel, and host mismatch readback.

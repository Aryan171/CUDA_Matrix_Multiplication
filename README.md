# A custom GEMM implementation using CUDA

This repository showcases the progression of performance optimization in CUDA, implementing various techniques to overcome the memory and compute bottlenecks.


| Kernel name | Key optimization | Description |
| --- | --- | --- |
| Naive | Baleline GPU | One thread per output element with global memory access. |
| Tiled | Shared Memory | Uses __shared__ memory tiles to reduce redundant VRAM reads. |
| 1D Block Tiling | Thread Level Reuse | Each thread calculates a vertical strip (4 elements) of the output. |
| 2D Block Tiling | Register Tiling | Each thread calculates a 4x4 patch, maximizing register reuse. |
| Double Buffering | Latency Hiding | Overlaps computation of Tile N with the global memory fetch of Tile N+1. |
| Most Optimal | Full Integration | Combines 2D Register Tiling, Double Buffering, and Bank Conflict Padding. |
| cuBlas | Reference | Uses NVIDIA's proprietary library as the performance benchmark |

## Key Optimizations

1. Register Tiling & 2D Blocking
The "Most Optimal" kernel utilizes a 4x4 micro-kernel per thread. By keeping the result fragments in registers, the kernel significantly reduces the pressure on the Shared Memory bandwidth.

2. Double Buffering (Prefetching)
By using two sets of Shared Memory buffers (tileA[2][...]), the kernel hides the high latency of Global Memory. While the ALUs are performing Fused Multiply-Add (FMA) operations on the current tile, the Load/Store units are already fetching the next tile from VRAM.

3. Bank Conflict Avoidance
All Shared Memory tiles use Padding (BLOCK_DIM + 1) to ensure that vertical column accesses across threads do not hit the same memory bank, preventing serialization of memory requests.

## Configuration

You can toggle specific behaviors at the top of kernel.cu:

#define TAKE_USER_INPUT: Manually enter dimensions.

#define PERFORM_CPU_MATRIX_MULTIPLICATION: Enable slow CPU baseline for comparison.

## Benchmarking Results

For a 5000 X 5000 matrix on an RTX 2050:  
Naive GPU: ~95 GFLOPS  
Most Optimal: ~272 GFLOPS  
cuBlas: ~578 GFLOPS  

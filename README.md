# wgpu-compute-benchmarks

Resources:

- [Mark Harris' slides on optimizing parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf),
as well as accompanying [video](https://www.youtube.com/watch?v=NrWhZMHrP4w&t=1236s)


## Conversion between CUDA and WGSL

The following table shows the mapping between CUDA and WGSL builtins:

| CUDA | WGSL |
|------|------|
| `__syncthreads()` | `workgroupBarrier()` |
| `blockIdx` | `workgroup_id` |
| `blockDim` | `workgroup_size` |
| `threadIdx` | `local_invocation_id` |

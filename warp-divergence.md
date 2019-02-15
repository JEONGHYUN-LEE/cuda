# Warp Divergence

## Warp Divergence

Warp divergence is generated when the branch granularity is different with multiple of warp size. The below kernel function is example of bad branch efficiency.

```cpp
__global__ void mathKernel1(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;
  if (tid % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}
```

To achieve 100% branch efficiency, the branch granularity have to be the multiple of warp size.

```cpp
__global__ void mathKernel2(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;
  if ((tid / warpSize) % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}
```


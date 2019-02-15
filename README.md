---
description: 'reference : Professional CUDA C Programming'
---

# Basic Patterns

## Generating GPU Vector Pattern

Allocate host memory and fill the values.

```cpp
int numberOfElement = 1024;
size_t numberOfBytes = numberOfElement*sizeof(float);
hostA = (float *)malloc(numberOfBytes);

for(int i=0;i<1024;i++){
    hostA[i] = i;
}
```

Allocate device memory.

```cpp
float * deviceA;
cudaMalloc((float**)&deviceA, numberOfBytes);
```

Copy host to device and make the host memory free.

```cpp
cudaMemcpy(deviceA, hostA, numberOfBytes, cudaMemcpyHostToDevice);
free(hostA);
```

The `cudaMemcpy` API is designed as below.

```cpp
​cudaError_t cudaMemcpy ( 
    void* dst, 
    const void* src, 
    size_t count, 
    cudaMemcpyKind kind 
)
```

Make the device memory free.

```cpp
cudaFree(deviceA);
```

## Sum of Two GPU Vector

Define function for GPU vector sum. We can get the global index of thread by 

$$
(\text{Block Dimension})*\text{(Block Index)}+\text{(Thread Index)}
$$

Sum of vector function is 

```cpp
__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  C[i] = A[i] + B[i];
}
```

Following the vector generating pattern, we can set the each three device and host vectors.

```cpp
int numberOfElement = 16;
size_t numberOfBytes = numberOfElement*sizeof(float);
float * hostA = (float *)malloc(numberOfBytes);
float * hostB = (float *)malloc(numberOfBytes);
float * hostC = (float *)malloc(numberOfBytes);


for(int i=0;i<16;i++){
  hostA[i] = i;
  hostB[i] = i;
  hostC[i] = 0;
}

float * deviceA;
float * deviceB;
float * deviceC;

cudaMalloc((float**)&deviceA, numberOfBytes);
cudaMalloc((float**)&deviceB, numberOfBytes);
cudaMalloc((float**)&deviceC, numberOfBytes);

cudaMemcpy(deviceA, hostA, numberOfBytes, cudaMemcpyHostToDevice);
cudaMemcpy(deviceB, hostB, numberOfBytes, cudaMemcpyHostToDevice);
free(hostA);
free(hostB);
```

Calculate appropriate block and grid dimension.

```cpp
dim3 block (numberOfElement);
dim3 grid (numberOfElement/block.x);
```

Call the kernel function with calculated dimension.

```cpp
sumArraysOnGPU<<< grid, block >>>(deviceA, deviceB, deviceC);
```

Copy the calculated result on device to host.

```cpp
cudaMemcpy(hostC, deviceC, numberOfBytes, cudaMemcpyDeviceToHost);
```

Check the result

```cpp
for(int i=0;i<16;i++){
    std::cout<<hostC[i]<<std::endl;
}
```

```bash
# result:
0
2
4
6
8
10
12
14
16
18
20
22
24
26
28
30
```



## 2D Block and 2D Grid For Matrix Summation

Define matrix summation kernel function.

```cpp
__global__ void sumMatrixOnGPU2D(
        float *MatA, 
        float *MatB, 
        float *MatC,
        int nx, // matrix size1
        int ny  // matrix size2               
        ) 
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy*nx + ix;
  if (ix < nx && iy < ny)
    MatC[idx] = MatA[idx] + MatB[idx];
}
```

Following gpu vector generating pattern

```cpp
int nx = 1000;
int ny = 1000;
int nxy = nx * ny;
int numberOfBytes = nxy * sizeof(float);


float *hostA, *hostB, *hostC;
hostA = (float *) malloc(numberOfBytes);
hostB = (float *) malloc(numberOfBytes);
hostC = (float *) malloc(numberOfBytes);

for (int i = 0; i < nxy; i++) {
  hostA[i] = 1;
  hostB[i] = 1;
  hostC[i] = 0;
}

float *deviceA, *deviceB, *deviceC;
cudaMalloc((void **) &deviceA, numberOfBytes);
cudaMalloc((void **) &deviceB, numberOfBytes);
cudaMalloc((void **) &deviceC, numberOfBytes);


// transfer data from host to device
cudaMemcpy(deviceA, hostA, numberOfBytes, cudaMemcpyHostToDevice);
cudaMemcpy(deviceB, hostB, numberOfBytes, cudaMemcpyHostToDevice);
free(hostA);
free(hostB);
```

Set grid and block dimension. We first designate block dimension then calculate corresponding grid dimension.

```cpp
int dimx = 32;
int dimy = 32;
dim3 block(dimx, dimy);
dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
```

Run kernel function and copy result from device to host.

```cpp
sumMatrixOnGPU2D <<< grid, block >>>(deviceA, deviceB, deviceC, nx, ny);
cudaMemcpy(hostC, deviceC, numberOfBytes, cudaMemcpyDeviceToHost);
```

The grid and block dimension is 

```cpp
// grid  = (32,32)
// block = (32,32)
```

## 1D Block and 1D Grid For Matrix Summation

Now each thread have to process one whole column of matrix.

```cpp
__global__ void sumMatrixOnGPU1D(
        float *MatA, 
        float *MatB, 
        float *MatC,
        int nx, // matrix size1
        int ny  // matrix size2               
        ) 
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix < nx ) {
    for (int iy=0; iy<ny; iy++) {
      int idx = iy*nx + ix;
      MatC[idx] = MatA[idx] + MatB[idx];
    } 
  }
}  
```

GPU vector generating code is same. Then generate 1-dimensional grid and block.

```cpp
dim3 block(32,1);
dim3 grid((nx+block.x-1)/block.x,1);
sumMatrixOnGPU1D <<< grid, block >>>(deviceA, deviceB, deviceC, nx, ny);
```

The grid and block dimension is

```cpp
// grid  = (32,1)
// block = (32,1)
```

## 1D Block and 2D Grid For Matrix Summation

The kernel function is defined by

```cpp
__global__ void sumMatrixOnGPUMix(
        float *MatA,
        float *MatB,
        float *MatC,
        int nx, //matrix size1
        int ny  //matrix size2
        ) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; 
  unsigned int iy = blockIdx.y;
  unsigned int idx = iy*nx + ix;
  if (ix < nx && iy < ny)
    MatC[idx] = MatA[idx] + MatB[idx];
}
```

Then call the kernel function with 2D-grid, 1D-block

```cpp
dim3 block(32);
dim3 grid((nx + block.x - 1) / block.x,ny);
sumMatrixOnGPUMIx <<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
```

```cpp
// grid  = (32,1000)
// block = (32,1)
```










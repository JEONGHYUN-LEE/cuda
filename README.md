# CUDA

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


































# CUDA

## Generating GPU Array Pattern

* Allocate host memory and fill the values

```cpp
int numberOfElement = 1024;
size_t nBytes = numberOfElement*sizeof(float);
hostA = (float *)malloc(nBtytes);

for(int i=0;i<1024;i++){
    hostA[i] = i;
}
```

* Allocate device memory 

```cpp
float * deviceA;
cudaMalloc((float**)&deviceA, numberOfBytes);
```

* Copy host to device

```cpp
cudaMemcpy(deviceA, hostA, numberOfBytes, cudaMemcpyHostToDevice);
```

The `cudaMemcpy` API is designed as

```cpp
​cudaError_t cudaMemcpy ( 
    void* dst, 
    const void* src, 
    size_t count, 
    cudaMemcpyKind kind 
)
```




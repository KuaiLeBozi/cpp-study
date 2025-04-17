# cuda实战代码

```cpp
dim3 threadsPerBlock(16, 16);//256 threads per block
dim3 blocksPerGrid(64, 64);//4096 blocks in the grid网格
_global_ void matrixAdd(float *A, float *B, float *C, int width){//A、B、C是指向矩阵A B C的指针，而width是矩阵的宽度（如果是方阵则是高和宽）
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;//计算出当前线程处理的矩阵元素的行和列索引
    if(row < width && col < width){//检查该索引是否在矩阵范围内
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}
//核函数的编写和执行的示例
//调用核函数
matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, 1024);
第一个参数为网格参数，代表有多少个线程块number_of_blocks
第二个参数为线程块参数，代表有多少个线程thread_per_block
```

**threadIdx**用来表示每个核（其实就是thread）的位置，而其为三维，固分有x、y、z

而**blockDim**也是**dim3**类型，x、y、z分别表示该block三个维度上各有**多少个核**，故这个block中核的总数为blockDim.x * blockDIm.y * blockDIm.z

而对于gird与block的关于与block与thread的关系类似，**blockIdx**也表示每个block在**grid**中的位置（与blockDim区分！！），**gridDim**与blockDim**同理**

---

**注意：**

- 在调用内核函数时，需要指定网格和线程块的维度，例如：

```cpp
dim3 threadsPerBlock(16, 16);
dim3 blocksPerGrid((width + 15) / 16, (width + 15) / 16);//
matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, width);
```

为什么使用 `(width + 15) / 16`？

假设矩阵的宽度为 `width`，并且每个线程块处理 16 个元素（因为 `threadsPerBlock` 是 (16, 16)）。为了确保**所有元素都被处理**，即使 `width` 不是 16 的倍数，我们需要计算至少多少个线程块才能覆盖整个矩阵。

`(width + 15) / 16` 的计算方式确保了即使 `width` 不是 16 的倍数，**也会分配足够的线程块**。例如：

- 如果 `width = 1024`，则 `(1024 + 15) / 16 = 64`，需要 64 个线程块。
- 如果 `width = 1025`，则 `(1025 + 15) / 16 = 64`，仍然需要 64 个线程块，因为需要**向上取整**。

这种计算方法确保了网格中有足够的线程块来**处理所有矩阵元素**，即使矩阵的尺寸不是 16 的倍数。

---

```cpp
// CUDA核函数，计算两个矩阵的和  
__global__ void matrixAdd(float *A, float *B, float *C, int width, int height) {  
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程的x坐标  
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 计算当前线程的y坐标  
    int index = x + y * width; // 计算当前线程在矩阵中的索引  
  
    if (x < width && y < height) { // 确保索引在矩阵范围内  
        C[index] = A[index] + B[index]; // 计算矩阵和  
    }  
}  
  
// 在主机代码中调用核函数  
dim3 blockSize(16, 16); // 每个Block的大小为16x16  
dim3 gridSize((width + blockSize.x - 1) / blockSize.x,   
              (height + blockSize.y - 1) / blockSize.y); // 根据矩阵大小和Block大小计算Grid大小  
  
matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height); // 调用核函数
```

以上为二维矩阵加法

---

假设我们有两个大型的1024×1024浮点数矩阵A和B，我们的目标是求和得到一个新的矩阵C，其中每个元素C[i][j]是A[i][j]和B[i][j]的和。为了并行化此操作，我们可以为每个矩阵元素分配一个线程。考虑到硬件的限制，我们选择每个线程块的大小为16×16，即每个线程块有256个线程。那么，我们需要64×64=4096个线程块来覆盖整个1024×1024的矩阵。这意味着我们的网格将是一个64×64的线程块集合。

在CUDA编程中，我们可以这样定义网格和线程块，并调用核函数：
```cpp
dim3 threadsPerBlock(16, 16); // 256 threads per block  
dim3 blocksPerGrid(64, 64); // 4096 blocks in the grid  
  
__global__ void matrixAdd(float *A, float *B, float *C, int width) {  
    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  
    if (row < width && col < width) {  
        int index = row * width + col;  
        C[index] = A[index] + B[index];  
    }  
}  
  
// 调用核函数  
matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, 1024);
```

---

`dim3` 结构体的使用

在CUDA编程中，`dim3` 用于定义线程块和线程网格的尺寸。例如，如果你想创建一个二维的线程网格，每个线程块有8x2个线程，而整个网格有2x2个这样的块，你可以这样定义：

```cpp
int nx = 16;
int ny = 4;
dim3 block(8, 2); // z默认为1
dim3 grid(nx/8, ny/2);
addKernel << <grid, block >> >(c, a, b);
```

 这一示例中创建了一个有(2*2)个block的grid，每个block中有(8*2)个thread，下图给出了更直观的表述

![img](https://i-blog.csdnimg.cn/direct/92b34e9d99da4ea99f94586251c3ee9b.webp)

要注意的是，对block、grid的尺寸定义并不是没有限制的，一个GPU中的核的数量同样是有限制的。对于一个block来说，总的核数不得超过1024，x、y维度都不得超过1024，z维度不得超过64，如下图

![img](https://i-blog.csdnimg.cn/direct/295523cc10fb4e6b915f460c9d70fdf9.webp)

对于整个grid而言，x维度上不得有超过232−1个thread，注意这里是thread而不是block，在其y维度和z维度上thread数量不得超过65536.

![img](https://i-blog.csdnimg.cn/direct/54ab2423cc0042dcacf33f9b95cedc33.webp)

然后，你可以使用这些 `dim3` 变量来配置内核（kernel）的启动参数：

```cpp
myKernel<<<gridSize, blockSize>>>(...);
```

这里，myKernel 是你的CUDA内核函数，<<<gridSize, blockSize>>> 指定了内核的启动配置，包括网格和块的维度。

注意事项

    在定义 dim3 变量时，如果某个维度不需要，可以将其设为1，但通常不建议省略该维度，以保持代码的一致性和可读性。
    
    CUDA编程中的线程是多维的，可以灵活地定义为一维、二维或三维结构，以适应不同的并行计算需求。
    
    使用 dim3 结构体时，需要包含CUDA的头文件（如 cuda_runtime.h），并确保你的开发环境已经正确配置了CUDA工具链。

综上所述，dim 在CUDA中通常指的是定义线程块和线程网格维度的结构体，但标准API中使用的是 dim3 结构体，而不是 dim 或 dim2。

---

#### 编程步骤

    配置开发环境：
        安装CUDA Toolkit。
        配置IDE（如Visual Studio、CLion等）以使用CUDA编译器（nvcc）。
    编写CUDA代码：
        使用CUDA C/C++编写代码，包括主机代码和设备代码。
        主机代码在CPU上执行，设备代码在GPU上执行。
    内存管理：
        使用cudaMalloc和cudaFree在GPU上分配和释放内存。
        使用cudaMemcpy在主机和设备之间传输数据。
    内核函数：
        使用__global__关键字定义内核函数。
        内核函数的参数包括线程索引，这些索引用于确定每个线程处理的数据。
    启动内核：
        使用<<<gridSize, blockSize>>>语法启动内核，指定网格和线程块的大小。
    同步和错误检查：
        使用cudaDeviceSynchronize等待GPU完成所有操作。
        使用cudaGetLastError和cudaPeekAtLastError检查错误。
![img](https://i-blog.csdnimg.cn/direct/447cccbf9e914ada8beebb2f06d1dea9.png)

简单说流程

1. 在CPU中初始化数据
2. 将输入传入GPU中
3. 利用分配好的grid和block启动kernel函数
4. 将计算结果传入cpu中
5. 释放申请的内存空间

一个CUDA程序主要分为两部分，第一部分运行在CPU上，称之为Host code,主要负责完成复杂的指令，第二部分运行在GPU上，称之为Device code，主要负责并行地完成指令。

```cpp
#include <iostream>  
#include <cuda_runtime.h>  
  
// CUDA内核函数，计算两个向量的加法  
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    if (i < numElements) {  
        C[i] = A[i] + B[i];  
    }  
}  
  
int main() {  
    int numElements = 50000;  
    size_t size = numElements * sizeof(float);  
  
    // 在主机上分配内存  
    float *h_A = (float *)malloc(size);  
    float *h_B = (float *)malloc(size);  
    float *h_C = (float *)malloc(size);  
  
    // 初始化向量A和B  
    for (int i = 0; i < numElements; ++i) {  
        h_A[i] = static_cast<float>(i);//static_cast用于显示类型转换
        h_B[i] = static_cast<float>(2 * i);  
    }  
  
    // 在设备上分配内存  
    float *d_A = nullptr;  
    float *d_B = nullptr;  
    float *d_C = nullptr;  
    cudaMalloc((void **)&d_A, size);  
    cudaMalloc((void **)&d_B, size);  
    cudaMalloc((void **)&d_C, size);  
  
    // 将数据从主机复制到设备  
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);  
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);  
  
    // 配置内核参数并启动内核  
    int threadsPerBlock = 256;  
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;  
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);  
  
    // 将结果从设备复制回主机  
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);  
  
    // 验证结果  
    bool success = true;  
    for (int i = 0; i < numElements; ++i) {//浮点数不能精确！！！只能减小误差！！  
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {  
            success = false;  
            break;  
        }  
    }  
  
    // 打印结果  
    if (success) {  
        std::cout << "Test PASSED" << std::endl;  
    } else {  
        std::cout << "Test FAILED" << std::endl;  
    }  
  
    // 释放内存  
    cudaFree(d_A);  
    cudaFree(d_B);  
    cudaFree(d_C);  
    free(h_A);  
    free(h_B);  
    free(h_C);  
  
    return 0;  
}
```

---

数据传输的性能优化

由于host与device之间的数据传输通常受到PCIe总线带宽的限制，因此优化数据传输性能对于提高CUDA程序的整体性能至关重要。以下是一些优化数据传输性能的建议：

* 减少数据传输量：尽可能减少在host与device之间传输的数据量。这可以通过在GPU上执行更多的计算来减少数据传输的需求，或者通过优化数据结构和算法来减少不必要的数据传输。

* 使用锁页内存（Pinned Memory）：锁页内存是一种特殊的内存分配方式，它允许GPU直接访问host内存，而无需通过系统内存进行中转。这可以显著提高数据传输的速度。在CUDA中，可以使用**cudaMallocHost**或**cudaHostAlloc**函数来分配锁页内存。

* 批量传输：将**多个小的数据传输合并为一个大的传输**可以显著提高性能，因为这样可以减少每次传输的开销。

* 重叠数据传输与计算：利用CUDA的**异步传输**功能，可以在数据传输的同时执行CUDA内核，从而隐藏数据传输的延迟。

* 内存对齐：确保数据的**内存对齐**可以提高传输性能。特别是对于二维和三维数据，使用**cudaMallocPitch**来分配内存可以确保每行的字节数对齐到适当的边界。

`cudaMallocPitch` 是 CUDA 编程模型中的一个函数，用于在设备（GPU）上分配二维数组的内存，并确保每一行的起始地址满足对齐要求，从而提高内存访问效率。[CSDN博客+2博客园+2掘金+2](https://www.cnblogs.com/csyisong/archive/2010/01/10/1643519.html?utm_source=chatgpt.com)

**函数原型：**

```cpp
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t widthInBytes, size_t height);
```

**参数说明：**

- `devPtr`：指向设备内存指针的指针，用于返回分配的内存地址。[博客园+3CSDN博客+3掘金+3](https://blog.csdn.net/jdhanhua/article/details/4813725?utm_source=chatgpt.com)
- `pitch`：指向 `size_t` 类型的变量，用于返回每一行的内存跨度（即步幅），单位为字节。
- `widthInBytes`：二维数组每一行的字节数。[知乎专栏+5NVIDIA+5掘金+5](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf?utm_source=chatgpt.com)
- `height`：二维数组的行数。[掘金](https://juejin.cn/post/7312404578959491110?utm_source=chatgpt.com)

**使用步骤：**

1. **分配内存：**

   使用 `cudaMallocPitch` 在设备上分配二维数组的内存，并获取每行的内存跨度。

```cpp
float* d_array;
size_t pitch;
size_t width = 1024;  // 假设每行有1024个浮点数
size_t height = 512;  // 假设有512行
cudaMallocPitch((void**)&d_array, &pitch, width * sizeof(float), height);
```

**访问内存：**

在设备代码（如核函数）中，使用返回的 `pitch` 值来计算每一行的起始地址，确保内存访问对齐。

```cpp
__global__ void kernel(float* devPtr, size_t pitch, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        float* rowPtr = (float*)((char*)devPtr + row * pitch);
        float value = rowPtr[col];
        // 进行计算...
    }
}
```

**释放内存：**

使用 `cudaFree` 释放之前分配的内存。

```cpp
cudaFree(d_array);
```

**注意事项：**

- `cudaMallocPitch` 会根据设备的对齐要求，可能在每一行末尾添加填充字节，以确保下一行的起始地址满足对齐要求。返回的 `pitch` 值表示每一行的实际字节数，包括填充字节。[NVIDIA+3博客园+3CSDN博客+3](https://www.cnblogs.com/csyisong/archive/2010/01/10/1643519.html?utm_source=chatgpt.com)
- 在访问二维数组时，必须使用返回的 `pitch` 值来计算每一行的起始地址，以确保正确访问数据。
- `cudaMallocPitch` 适用于需要在设备上处理二维数组的场景，特别是当数据需要在设备内存的不同区域之间进行复制时。[CSDN博客+1博客园+1](https://blog.csdn.net/jdhanhua/article/details/4813725?utm_source=chatgpt.com)

通过使用 `cudaMallocPitch`，可以有效地管理二维数组在设备内存中的布局，确保内存访问的高效性和正确性。

---

数据传输的示例代码

以下是一个简单的示例代码，演示了如何在CUDA中使用cudaMemcpy函数来传输数据：

```cpp
#include <stdio.h>  
  
__global__ void myKernel(int *data) {  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    data[idx] = idx; // 简单的内核操作，将索引值赋给数组元素  
}  
  
int main() {  
    const int arraySize = 1024;  
    int *h_data = (int *)malloc(arraySize * sizeof(int)); // 分配host内存  
    int *d_data;  
  
    // 分配device内存  
    cudaMalloc((void **)&d_data, arraySize * sizeof(int));  
  
    // 初始化host数据  
    for (int i = 0; i < arraySize; i++) {  
        h_data[i] = 0;  
    }  
  
    // 将数据从host传输到device  
    cudaMemcpy(d_data, h_data, arraySize * sizeof(int), cudaMemcpyHostToDevice);  
  
    // 启动CUDA内核  
    dim3 blockSize(256);  
    dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x);  
    myKernel<<<gridSize, blockSize>>>(d_data);  
  
    // 将数据从device传输回host  
    cudaMemcpy(h_data, d_data, arraySize * sizeof(int), cudaMemcpyDeviceToHost);  
  
    // 处理host数据（这里只是简单地打印前几个元素）  
    for (int i = 0; i < 10; i++) {  
        printf("%d ", h_data[i]);  
    }  
    printf("\n");  
  
    // 释放内存  
    free(h_data);  
    cudaFree(d_data);  
  
    return 0;  
}
```

---

以下是一个简单的**CUDA内存分配**示例：

```cpp
int main()
{
    const int arraySize = 64;
    const int byteSize = arraySize * sizeof(int);
 
    int *h_input,*d_input;
    h_input = (int*)malloc(byteSize);
    // 在GPU上分配全局内存 
    cudaMalloc((void **)&d_input,byteSize);
    // 检查内存分配是否成功  
    if (cudaGetLastError() != cudaSuccess) {  
        std::cerr << "Failed to allocate device memory" << std::endl;  
        return -1;  
    }  
 
    srand((unsigned)time(NULL));
    for (int i = 0; i < 64; ++i)
    {
        if(h_input[i] != NULL)h_input[i] = (int)rand()& 0xff;
    }
 
    cudaMemcpy(d_input, h_input, byteSize, cudaMemcpyHostToDevice);
 
    int nx = 4, ny = 4, nz = 4;
    dim3 block(2, 2, 2);
    dim3 grid(nx/2, ny/2, nz/2);
    print_array << < grid, block >> > (d_input);
    cudaDeviceSynchronize();
 
    cudaFree(d_input);
    free(h_input);
 
    return 0;
}
```

void**:在一些函数中，需要返回一个指向指针的指针，这样可以在函数内部分配内存并将地址返回给调用者。例如，CUDA 的 `cudaMalloc` 等函数经常使用 `void**` 来返回分配的内存地址。在这种情况下，`(void**)&devPtr` 将 `devPtr` 的地址传递给 `cudaMalloc`，以便 `cudaMalloc` 可以修改 `devPtr`，将其指向分配的设备内存。
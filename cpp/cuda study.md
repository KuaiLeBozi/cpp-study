# cuda study

典型的CUDA程序的执行流程如下：

1. 分配host内存，并进行数据初始化；
2. 分配device内存，并从host将数据拷贝到device上；
3. 调用CUDA的[核函数](https://zhida.zhihu.com/search?content_id=6024941&content_type=Article&match_order=1&q=核函数&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NDMzMTk1NTQsInEiOiLmoLjlh73mlbAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjo2MDI0OTQxLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.opjsYoRqF4knmmIj-5J0UzwUZefcrel8YbwHQnVa2Uo&zhida_source=entity)

在device上完成指定的运算；

将device上的运算结果拷贝到host上；

释放device和host上分配的内存。

---

[kernel](https://link.zhihu.com/?target=http%3A//docs.nvidia.com/cuda/cuda-c-programming-guide/index.html%23kernels)是CUDA中一个重要的概念，kernel是在device上线程中并行执行的函数，核函数用`__global__`符号声明，在调用时需要用`<<<grid, block>>>`来指定kernel要执行的线程数量，在CUDA中，每一个线程都要执行核函数，并且每个线程会分配一个唯一的线程号thread ID，这个ID值可以通过核函数的内置变量`threadIdx`来获得。

---

由于GPU实际上是异构模型，所以需要区分host和device上的代码，在CUDA中是通过函数类型限定词开区别host和device上的函数，主要的三个函数类型限定词如下：

- `__global__`：在device上执行，从host中调用（一些特定的GPU也可以从device上调用），返回类型必须是`void`，不支持可变参数参数，不能成为类成员函数。注意用`__global__`定义的kernel是异步的，这意味着host不会等待kernel执行完就执行下一步。
- `__device__`：在device上执行，单仅可以从device中调用，不可以和`__global__`同时用。
- `__host__`：在host上执行，仅可以从host上调用，一般省略不写，不可以和`__global__`同时用，但可和`__device__`，此时函数会在device和host都编译。

---

cuda中dim3类型的变量是什么意思:

在 CUDA 编程中，`dim3` 是一种内置的数据类型，用于表示三维向量。它由 NVIDIA 定义，通常用于指定线程块（block）和网格（grid）的维度，以便在 GPU 上进行并行计算。

### `dim3` 的结构

`dim3` 类型实际上是一个包含三个 `unsigned int` 成员的结构体，分别表示三维空间中的 x、y 和 z 维度：[chhzh123.github.io](https://chhzh123.github.io/summary/parallel-computing/cuda/?utm_source=chatgpt.com)

```cpp
typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;
```

在使用 `dim3` 时，如果在初始化时未指定某个维度的值，该维度会默认初始化为 1。[博客园](https://www.cnblogs.com/chuaner/p/15100409.html?utm_source=chatgpt.com)

### 使用示例

在 CUDA 中，`dim3` 常用于设置线程块和网格的尺寸。例如，定义一个二维的线程块和网格：

```cpp
dim3 block(16, 16); // 线程块大小为 16x16
dim3 grid(32, 32);   // 网格大小为 32x32
```

这表示网格由 32x32 个线程块组成，每个线程块包含 16x16 个线程，共有 1024 个线程块，每个线程块有 256 个线程，总计 262,144 个线程。

### 相关内置变量

在 CUDA 中，除了 `dim3` 类型外，还有一些与线程和块相关的内置变量：

- `threadIdx.x`, `threadIdx.y`, `threadIdx.z`: 表示当前线程在线程块中的索引。[简书+2知乎专栏+2chhzh123.github.io+2](https://zhuanlan.zhihu.com/p/455134132?utm_source=chatgpt.com)
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z`: 表示当前线程块在网格中的索引。[CSDN博客+3CSDN博客+3知乎专栏+3](https://blog.csdn.net/weixin_34279184/article/details/93902570?utm_source=chatgpt.com)
- `blockDim.x`, `blockDim.y`, `blockDim.z`: 表示线程块在各维度上的大小。
- `gridDim.x`, `gridDim.y`, `gridDim.z`: 表示网格在各维度上的大小。

这些变量帮助开发者在编写内核函数时确定每个线程的全局唯一标识，从而进行数据访问和计算。

通过使用 `dim3` 类型和相关的内置变量，CUDA 编程模型能够灵活地在一维、二维或三维空间中组织线程和线程块，以充分利用 GPU 的并行计算能力。[chhzh123.github.io](https://chhzh123.github.io/summary/parallel-computing/cuda/?utm_source=chatgpt.com)

---

CUDA线程是GPU执行并行计算任务的**最小单元**

**线程束**是GPU**的基本执行单元** 通常包含32线程

---

CUDA中的Block（线程块）是一个重要的概念，它是CUDA并行计算模型中的基本组织单元。以下是对CUDA Block的详细解释：

1、定义与组成

    定义：Block是CUDA编程模型中的一个基本概念，它代表了一组同时启动的线程，这些线程可以协同工作并通过共享内存进行通信。
    
    组成：每个Block内部包含了一定数量的线程（Thread），这些线程以三维结构组织，但实践中常用的是一维或二维结构。

2、特点与功能

    共享内存：Block内的线程可以共享一块内存区域，即共享内存（Shared Memory）。这使得线程间的数据交换和通信变得高效。
    
    同步机制：Block内的线程可以通过同步原语（如__syncthreads()）进行同步，确保所有线程在执行到某个点时都达到一致状态。
    
    并行执行：虽然Block内的线程是并行执行的，但不同Block之间的线程是独立执行的，它们之间没有直接的通信和同步机制。
    
    执行效率：Block的设计考虑了GPU的物理架构，以实现最大的并行性和效率。一个Block中的线程通常会被映射到同一个GPU的流多处理器（Streaming Multiprocessor，SM）上执行。

3、配置与执行

    配置：在调用CUDA核函数（Kernel）时，需要指定网格（Grid）和Block的大小。Grid是由多个Block组成的，而每个Block则包含了一定数量的线程。这些大小可以通过dim3类型的变量来指定，其中dim3是一个表示三维向量的数据类型。
    
    执行：当核函数被调用时，它会根据指定的Grid和Block大小，在GPU上启动相应数量的线程。这些线程会并行地执行核函数中的代码，直到所有线程都完成计算。

[https://blog.csdn.net/dcrmg/article/details/54867507?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task](公式)

---

==cuda有很多**公式**可以套==

---

网格（Grid）、线程块（Block）和线程（Thread）的组织关系

1. 三者关系

 三者关系： 一个CUDA的并行程序会被以许多个thread来执行，数个thread会被群组成一个block，同一个block中的thread可以同步，也可以通过shared memory进行通信，多个block则会再构成grid。
![img](https://i-blog.csdnimg.cn/direct/f7e8a88f6e634a2eba8d4f5e070a9f33.webp)

---

#### 线程索引的计算公式

一个Grid可以包含多个Blocks，Blocks的组织方式可以是一维的，二维或者三维的。block包含多个Threads，这些Threads的组织方式也可以是一维，二维或者三维的。
CUDA中每一个线程都有一个唯一的标识ID—ThreadIdx，这个ID随着Grid和Block的划分方式的不同而变化，这里给出Grid和Block不同划分方式下线程索引ID的计算公式。

在CUDA编程中，为了唯一标识和定位每个线程，通常使用以下变量：

```cpp
threadIdx：表示线程在其所在线程块中的索引，是一个三维向量（threadIdx.x, threadIdx.y, threadIdx.z），分别表示线程在x、y、z三个维度上的位置。

blockIdx：表示线程块在其所在线程网格中的索引，同样是一个三维向量（blockIdx.x, blockIdx.y, blockIdx.z）。

blockDim：表示线程块的大小，即线程块中线程的数量，也是一个三维向量（blockDim.x, blockDim.y, blockDim.z）。

gridDim：表示线程网格的大小，即线程网格中线程块的数量，同样是一个三维向量（gridDim.x, gridDim.y, gridDim.z）。
```

通过这些变量，可以唯一地标识和定位每个线程，从而实现精确的并行计算任务分配。

```cpp
 grid划分成1维，block划分为1维

int threadId = blockIdx.x *blockDim.x + threadIdx.x;  
```


```cpp
grid划分成1维，block划分为2维  

int threadId = blockIdx.x * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x + threadIdx.x;  

grid划分成1维，block划分为3维  

int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z  
                   + threadIdx.z * blockDim.y * blockDim.x  
                   + threadIdx.y * blockDim.x + threadIdx.x;  

grid划分成2维，block划分为1维  

int blockId = blockIdx.y * gridDim.x + blockIdx.x;  
int threadId = blockId * blockDim.x + threadIdx.x;  
```


```cpp
grid划分成2维，block划分为2维 

int blockId = blockIdx.x + blockIdx.y * gridDim.x;  
int threadId = blockId * (blockDim.x * blockDim.y)  
                   + (threadIdx.y * blockDim.x) + threadIdx.x;  
```


    grid划分成2维，block划分为3维
    
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;  
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)  
                       + (threadIdx.z * (blockDim.x * blockDim.y))  
                       + (threadIdx.y * blockDim.x) + threadIdx.x;  


```cpp
grid划分成3维，block划分为1维 

int blockId = blockIdx.x + blockIdx.y * gridDim.x  
                 + gridDim.x * gridDim.y * blockIdx.z;  
int threadId = blockId * blockDim.x + threadIdx.x;  
```


```cpp
grid划分成3维，block划分为2维  

int blockId = blockIdx.x + blockIdx.y * gridDim.x  
                 + gridDim.x * gridDim.y * blockIdx.z;  
int threadId = blockId * (blockDim.x * blockDim.y)  
                   + (threadIdx.y * blockDim.x) + threadIdx.x;  
```


```cpp
grid划分成3维，block划分为3维

int blockId = blockIdx.x + blockIdx.y * gridDim.x  
                 + gridDim.x * gridDim.y * blockIdx.z;  
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)  
                   + (threadIdx.z * (blockDim.x * blockDim.y))  
                   + (threadIdx.y * blockDim.x) + threadIdx.x;     
```
---

CUDA主机端与设备端之间的数据传输

在CUDA编程模型中，host（主机端，即CPU）与device（设备端，即GPU）之间的数据传输是一个至关重要的环节。以下是对CUDA中host与device数据传输的详细分析：

1、数据传输的基本方式

CUDA提供了多种方式来在host与device之间传输数据，主要包括以下几种：

    cudaMemcpy：这是CUDA中最常用的数据传输函数，用于将数据从host端拷贝到device端，或者从device端拷贝到host端。它支持一维数据的传输，并且可以通过指定不同的拷贝方向（如cudaMemcpyHostToDevice、cudaMemcpyDeviceToHost等）来实现数据的双向传输。
    
    cudaMemcpy2D/cudaMemcpy3D：这些函数用于传输二维或三维数据。对于二维数据，cudaMemcpy2D允许指定每行的字节数（pitch），这对于内存对齐和性能优化非常重要。
    
    异步传输：CUDA还支持异步数据传输，即cudaMemcpyAsync、cudaMemcpy2DAsync和cudaMemcpy3DAsync等函数。这些函数允许数据传输与CUDA内核的执行并行进行，从而进一步提高程序的性能。

2、数据传输的性能优化

由于host与device之间的数据传输通常受到PCIe总线带宽的限制，因此优化数据传输性能对于提高CUDA程序的整体性能至关重要。以下是一些优化数据传输性能的建议：

    减少数据传输量：尽可能减少在host与device之间传输的数据量。这可以通过在GPU上执行更多的计算来减少数据传输的需求，或者通过优化数据结构和算法来减少不必要的数据传输。
    
    使用锁页内存（Pinned Memory）：锁页内存是一种特殊的内存分配方式，它允许GPU直接访问host内存，而无需通过系统内存进行中转。这可以显著提高数据传输的速度。在CUDA中，可以使用cudaMallocHost或cudaHostAlloc函数来分配锁页内存。
    
    批量传输：将多个小的数据传输合并为一个大的传输可以显著提高性能，因为这样可以减少每次传输的开销。
    
    重叠数据传输与计算：利用CUDA的异步传输功能，可以在数据传输的同时执行CUDA内核，从而隐藏数据传输的延迟。
    
    内存对齐：确保数据的内存对齐可以提高传输性能。特别是对于二维和三维数据，使用cudaMallocPitch来分配内存可以确保每行的字节数对齐到适当的边界。

---

减少数据传输量：尽可能减少在host与device之间传输的数据量。这可以通过在GPU上执行更多的计算来减少数据传输的需求，或者通过优化数据结构和算法来减少不必要的数据传输。

使用锁页内存（Pinned Memory）：锁页内存是一种特殊的内存分配方式，它允许GPU直接访问host内存，而无需通过系统内存进行中转。这可以显著提高数据传输的速度。在CUDA中，可以使用cudaMallocHost或cudaHostAlloc函数来分配锁页内存。

批量传输：将多个小的数据传输合并为一个大的传输可以显著提高性能，因为这样可以减少每次传输的开销。

重叠数据传输与计算：利用CUDA的异步传输功能，可以在数据传输的同时执行CUDA内核，从而隐藏数据传输的延迟。

内存对齐：确保数据的内存对齐可以提高传输性能。特别是对于二维和三维数据，使用cudaMallocPitch来分配内存可以确保每行的字节数对齐到适当的边界。

---

#### CUDA内存类型

CUDA支持多种内存类型，每种类型具有不同的访问特性和用途：

全局内存（Global Memory）：这是最大的内存空间，所有线程都可以访问。然而，由于其访问速度相对较慢，因此通常用于存储需要在多个线程或线程块之间共享的大量数据。

共享内存（Shared Memory）：这是每个线程块内的线程可以共享的内存空间，访问速度非常快。但由于容量有限，因此通常用于存储需要在线程块内频繁访问的小量数据。

常量内存（Constant Memory）：这是只读内存，所有线程都可以访问，并且访问速度也较快。它通常用于存储常量数据或参数。

本地内存（Local Memory）：这是每个线程的私有内存空间，通常用于存储线程内的局部变量。然而，由于其访问速度较慢，因此应尽量避免使用。

寄存器（Registers）：这是每个线程的私有寄存器空间，访问速度最快。寄存器通常用于存储线程内的临时变量和计算结果。

---

#### CUDA内存分配函数

CUDA提供了一系列函数用于在GPU上分配内存：

cudaMalloc：用于在GPU上分配全局内存。其函数原型为cudaMalloc(void** devPtr, size_t size)，其中devPtr是指向分配的内存空间的指针的指针，size是要分配的内存大小（以字节为单位）。

cudaMallocPitch：用于在GPU上分配二维全局内存，并考虑内存对齐。其函数原型为cudaMallocPitch(void** devPtr, size_t* pitch, size_t widthInBytes, size_t height)，其中devPtr是指向分配的内存空间的指针的指针，pitch是返回的行宽（以字节为单位），widthInBytes是每行的字节数，height是矩阵的行数。

cudaMalloc3D：用于在GPU上分配三维全局内存。

cudaFree：用于释放之前分配的内存。其函数原型为cudaFree(void* devPtr)，其中devPtr是指向要释放的内存空间的指针。

---

#### 注意事项

1. **内存对齐**：在分配二维或三维全局内存时，需要考虑内存对齐以提高访问效率。CUDA提供了cudaMallocPitch函数来处理二维内存的对齐问题。
2. **内存释放**：在CUDA编程中，必须确保在不再需要时使用cudaFree函数释放之前分配的内存，以避免内存泄漏。
3. **错误检查**：在进行内存分配和其他CUDA操作时，应始终检查返回的错误代码以确保操作成功。

---


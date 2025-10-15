// 对base进行优化：https://zhuanlan.zhihu.com/p/98416987
// recursiveReduce： 使用CPU进行向量加和
// reduceNeighbored：相邻配对的方式进行加和（GPU）
// reduceNeighboredLess：减少分支的相邻配对方式进行加
// reduceInterleaved：交叉配对的方式进行加和

#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1)
        return data[0];
    // renew the stride
    int const stride = size / 2;
    if (size % 2 == 1)
    {
        for (int i = 0; i < stride; i++)
        {
            data[i] += data[i + stride];
        }
        data[0] += data[size - 1];
    }
    else
    {
        for (int i = 0; i < stride; i++)
        {
            data[i] += data[i + stride];
        }
    }
    // call
    return recursiveReduce(data, stride);
}
// warmup是用来激活GPU的，因为第一次调用GPU速度会慢一点
__global__ void warmup(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // 每个线程块处理 1个blockDim.x 大小的数据块
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        // 等待所有线程结束
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0) // 比如第一次循环中，每个元素对应一个线程，【1】和【2】做计算，【3】和【4】做计算，【5】和【6】做计算，【7】和【8】做计算，4个线程都用来计算了。但是第二次循环中，就是【1】和【3】做计算，【5】和【7】做计算，只有2个线程在计算，这样就有一半的线程在闲置了。
        {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local point of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx > n)
        return;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x) // 这种方法就开了8个线程但实际上用到的只有4个。第一次循环index=2*1*线程，每个元素和相邻的元素加起来，只有前4个线程进行计算。第二次循环index=2*2*线程，只有前2个线程进行计算。这种方法避免了穿插分支“不做功”。
        {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }
    // write result for this block to global men
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local point of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) // 这种方法就是将内存变成相邻的，计算的时候会快一点
    {

        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global men
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char **argv)
{
    initDevice(0);

    // initialization

    int size = 1 << 24;
    printf("	with array size %d  ", size);

    // execution configuration
    int blocksize = 256;
    if (argc > 1)
    {
        blocksize = atoi(argv[1]); // 从命令行输入设置block大小
    }
    dim3 block(blocksize, 1);
    dim3 grid(448, 1);
    printf("grid %d block %d \n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *idata_host = (int *)malloc(bytes);
    int *odata_host = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    // initialize the array
    initialData_int(idata_host, size);

    memcpy(tmp, idata_host, bytes);
    double timeStart, timeElaps;
    int gpu_sum = 0;

    // device memory
    int *idata_dev = NULL;
    int *odata_dev = NULL;
    CHECK(cudaMalloc((void **)&idata_dev, bytes));
    CHECK(cudaMalloc((void **)&odata_dev, grid.x * sizeof(int)));

    // cpu reduction 对照组
    int cpu_sum = 0;
    timeStart = cpuSecond();
    // cpu_sum = recursiveReduce(tmp, size);
    for (int i = 0; i < size; i++)
        cpu_sum += tmp[i];
    timeElaps = 1000 * (cpuSecond() - timeStart);

    printf("cpu sum:%d \n", cpu_sum);
    printf("cpu reduction elapsed %lf ms cpu_sum: %d\n", timeElaps, cpu_sum);

    // kernel 1:warmup
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    timeStart = cpuSecond();
    warmup<<<grid.x, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    timeElaps = 1000 * (cpuSecond() - timeStart);
    printf("gpu warmup                  elapsed %lf ms \n", timeElaps);

    // kernel 1 reduceNeighbored
    printf("****************** kernel 1 *****************\n");
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    timeStart = cpuSecond();
    reduceNeighbored<<<grid.x, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    timeElaps = 1000 * (cpuSecond() - timeStart);
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu sum:%d \n", gpu_sum);
    printf("gpu reduceNeighbored elapsed %lf ms     <<<grid %d block %d>>>\n",
           timeElaps, grid.x, block.x);

    // kernel 2 reduceNeighboredless
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    timeStart = cpuSecond();
    reduceNeighboredLess<<<grid.x, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    timeElaps = 1000 * (cpuSecond() - timeStart);
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];

    printf("gpu sum:%d \n", gpu_sum);
    printf("gpu reduceNeighboredless elapsed %lf ms     <<<grid %d block %d>>>\n",
           timeElaps, grid.x, block.x);

    // kernel 3 reduceInterleaved
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    timeStart = cpuSecond();
    reduceInterleaved<<<grid.x, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    timeElaps = 1000 * (cpuSecond() - timeStart);
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];

    printf("gpu sum:%d \n", gpu_sum);
    printf("gpu reduceInterleaved elapsed %lf ms     <<<grid %d block %d>>>\n",
           timeElaps, grid.x, block.x);

    // free host memory

    free(idata_host);
    free(odata_host);
    CHECK(cudaFree(idata_dev));
    CHECK(cudaFree(odata_dev));

    // reset device
    cudaDeviceReset();

    // check the results
    if (gpu_sum == cpu_sum)
    {
        printf("Test success!\n");
    }
    return EXIT_SUCCESS;
}
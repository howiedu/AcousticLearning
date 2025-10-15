// 线程束洗牌。我们知道，核函数内部的变量都在寄存器中，一个线程束可以看做是32个内核并行执行，换句话说这32个核函数中寄存器变量在硬件上其实都是邻居，这样就为相互访问提供了物理基础，线程束内线程相互访问数据不通过共享内存或者全局内存，使得通信效率高很多，线程束洗牌指令传递数据，延迟极低，且不消耗内存
// 线程束洗牌
#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"
#define BDIM 16
#define SEGM 4

// 使用正确的掩码参数
__global__ void test_shfl_broadcast(int *in, int *out, int const srcLane)
{
    int value = in[threadIdx.x];
    value = __shfl_sync(0xFFFFFFFF, value, srcLane, BDIM); // 第一个参数是掩码
    out[threadIdx.x] = value;
}

__global__ void test_shfl_up(int *in, int *out, int const delta)
{
    int value = in[threadIdx.x];
    value = __shfl_up_sync(0xFFFFFFFF, value, delta, BDIM); // 第一个参数是掩码
    out[threadIdx.x] = value;
}

__global__ void test_shfl_down(int *in, int *out, int const delta)
{
    int value = in[threadIdx.x];
    value = __shfl_down_sync(0xFFFFFFFF, value, delta, BDIM); // 第一个参数是掩码
    out[threadIdx.x] = value;
}

__global__ void test_shfl_wrap(int *in, int *out, int const offset)
{
    int value = in[threadIdx.x];
    value = __shfl_sync(0xFFFFFFFF, value, threadIdx.x + offset, BDIM); // 第一个参数是掩码
    out[threadIdx.x] = value;
}

__global__ void test_shfl_xor(int *in, int *out, int const mask)
{
    int value = in[threadIdx.x];
    value = __shfl_xor_sync(0xFFFFFFFF, value, mask, BDIM); // 第一个参数是掩码
    out[threadIdx.x] = value;
}

__global__ void test_shfl_xor_array(int *in, int *out, int const mask)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];
    for (int i = 0; i < SEGM; i++)
        value[i] = in[idx + i];

    value[0] = __shfl_xor_sync(0xFFFFFFFF, value[0], mask, BDIM);
    value[1] = __shfl_xor_sync(0xFFFFFFFF, value[1], mask, BDIM);
    value[2] = __shfl_xor_sync(0xFFFFFFFF, value[2], mask, BDIM);
    value[3] = __shfl_xor_sync(0xFFFFFFFF, value[3], mask, BDIM);

    for (int i = 0; i < SEGM; i++)
        out[idx + i] = value[i];
}

__inline__ __device__ void swap(int *value, int laneIdx, int mask, int firstIdx, int secondIdx)
{
    bool pred = ((laneIdx % (2)) == 0);
    if (pred)
    {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }
    value[secondIdx] = __shfl_xor_sync(0xFFFFFFFF, value[secondIdx], mask, BDIM);
    if (pred)
    {
        int tmp = value[firstIdx];
        value[firstIdx] = value[secondIdx];
        value[secondIdx] = tmp;
    }
}

__global__ void test_shfl_swap(int *in, int *out, int const mask, int firstIdx, int secondIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];
    for (int i = 0; i < SEGM; i++)
        value[i] = in[idx + i];
    swap(value, threadIdx.x, mask, firstIdx, secondIdx);
    for (int i = 0; i < SEGM; i++)
        out[idx + i] = value[i];
}

int main(int argc, char **argv)
{
    printf("Starting...\n");
    initDevice(0);

    // 检查设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    int dimx = BDIM;
    unsigned int data_size = BDIM;
    int nBytes = data_size * sizeof(int);
    int kernel_num = 0;
    if (argc >= 2)
        kernel_num = atoi(argv[1]);

    // 初始化数据
    int in_host[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int *out_gpu = (int *)malloc(nBytes);

    // cudaMalloc
    int *in_dev = NULL;
    int *out_dev = NULL;

    CHECK(cudaMalloc((void **)&in_dev, nBytes));
    CHECK(cudaMalloc((void **)&out_dev, nBytes));
    CHECK(cudaMemcpy(in_dev, in_host, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out_dev, 0, nBytes));

    // 测试不同的 shuffle 操作
    dim3 block(dimx);
    dim3 grid((data_size - 1) / block.x + 1);

    printf("Running kernel %d\n", kernel_num);
    switch (kernel_num)
    {
    case 0:
        test_shfl_broadcast<<<grid, block>>>(in_dev, out_dev, 2);
        printf("test_shfl_broadcast - Broadcasting from lane 2\n");
        break;
    case 1:
        test_shfl_up<<<grid, block>>>(in_dev, out_dev, 2);
        printf("test_shfl_up - Shifting up by 2\n");
        break;
    case 2:
        test_shfl_down<<<grid, block>>>(in_dev, out_dev, 2);
        printf("test_shfl_down - Shifting down by 2\n");
        break;
    case 3:
        test_shfl_wrap<<<grid, block>>>(in_dev, out_dev, 2);
        printf("test_shfl_wrap - Wrapping with offset 2\n");
        break;
    case 4:
        test_shfl_xor<<<grid, block>>>(in_dev, out_dev, 1);
        printf("test_shfl_xor - XOR with mask 1\n");
        break;
    case 5:
        test_shfl_xor_array<<<1, block.x / SEGM>>>(in_dev, out_dev, 1);
        printf("test_shfl_xor_array - XOR array with mask 1\n");
        break;
    case 6:
        test_shfl_swap<<<1, block.x / SEGM>>>(in_dev, out_dev, 1, 0, 3);
        printf("test_shfl_swap - Swapping elements\n");
        break;
    default:
        printf("Invalid kernel number\n");
        break;
    }

    // 检查内核执行错误
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess)
    {
        printf("Kernel execution error: %s\n", cudaGetErrorString(kernelError));
        return -1;
    }

    CHECK(cudaMemcpy(out_gpu, out_dev, nBytes, cudaMemcpyDeviceToHost));

    // 显示结果
    printf("Input:\t");
    for (int i = 0; i < data_size; i++)
        printf("%4d ", in_host[i]);
    printf("\nOutput:\t");
    for (int i = 0; i < data_size; i++)
        printf("%4d ", out_gpu[i]);
    printf("\n");

    // 清理
    cudaFree(in_dev);
    cudaFree(out_dev);
    free(out_gpu);
    cudaDeviceReset();

    printf("Completed successfully!\n");
    return 0;
}
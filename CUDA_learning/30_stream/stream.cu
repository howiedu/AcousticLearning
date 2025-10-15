// 流
#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"

#define N 300000
__global__ void kernel_1()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum = sum + tan(0.1) * tan(0.1);
}
__global__ void kernel_2()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum = sum + tan(0.1) * tan(0.1);
}
__global__ void kernel_3()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum = sum + tan(0.1) * tan(0.1);
}
__global__ void kernel_4()
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum = sum + tan(0.1) * tan(0.1);
}
int main()
{
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
    int dev = 0;
    cudaSetDevice(dev);

    int n_stream = 16; // 创建16个流
    cudaStream_t *stream = (cudaStream_t *)malloc(n_stream * sizeof(cudaStream_t));
    for (int i = 0; i < n_stream; i++)
    {
        cudaStreamCreate(&stream[i]); // 创建流
    }

    dim3 block(1);
    dim3 grid(1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);         // 计时开始
    for (int i = 0; i < n_stream; i++) // 每个流中顺序执行4个核函数, 一共执行16*4=64个核函数
    {
        kernel_1<<<grid, block, 0, stream[i]>>>();
        kernel_2<<<grid, block, 0, stream[i]>>>();
        kernel_3<<<grid, block, 0, stream[i]>>>();
        kernel_4<<<grid, block, 0, stream[i]>>>();
    }
    cudaEventRecord(stop, 0);
    CHECK(cudaEventSynchronize(stop)); // 注意，如果没有这句话，nvvp将会无法运行，因为所有这些都是异步操作，不会等到操作完再返回，而是启动后自动把控制权返回主机，如果没有一个阻塞指令，主机进程就会执行完毕推出，这样就跟设备失联了，nvvp也会相应的报错。
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop); // 计算耗时
    printf("elapsed time:%f ms\n", elapsed_time);

    for (int i = 0; i < n_stream; i++)
    {
        cudaStreamDestroy(stream[i]); // 销毁流
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(stream);
    CHECK(cudaDeviceReset());
    return 0;
}

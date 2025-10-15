// 全局变量使用示例
// 1. 使用__device__修饰符声明设备全局变量
// 2. 使用cudaMemcpyToSymbol()函数将主机端数据拷贝
// 如果全局变量被修改，那么要使用cudaMemcpyFromSymbol()函数将数据重新拷贝回主机
#include <cuda_runtime.h>
#include <stdio.h>
__device__ float devData;
__global__ void checkGlobalVariable()
{
    printf("Device: The value of the global variable is %f\n", devData);
    devData += 2.0;
}
int main()
{
    float value = 3.14f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float)); // 将主机端的值(3.14)拷贝到设备全局变量devData
    printf("Host: copy %f to the global variable\n", value);
    checkGlobalVariable<<<1, 1>>>();
    cudaMemcpyFromSymbol(&value, devData, sizeof(float)); // 将设备全局变量devData的值拷贝到主机
    printf("Host: the value changed by the kernel to %f \n", value);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}

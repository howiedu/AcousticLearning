// 这段代码实现了一个简单的向量加法，并且使用CPU和GPU做了一个结果对比
// 包括“分配内存”、“初始化向量”、“分配核和线程”、“运算”、“结果拷贝回主机”、“结果对比”、“释放内存”

#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"

void sumArrays(float *a, float *b, float *res, const int size)
{
  for (int i = 0; i < size; i += 4)
  {
    res[i] = a[i] + b[i];
    res[i + 1] = a[i + 1] + b[i + 1];
    res[i + 2] = a[i + 2] + b[i + 2];
    res[i + 3] = a[i + 3] + b[i + 3];
  }
}
__global__ void sumArraysGPU(float *a, float *b, float *res)
{
  // int i=threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  res[i] = a[i] + b[i];
}
int main(int argc, char **argv)
{
  int dev = 0;
  cudaSetDevice(dev);

  int nElem = 1 << 14; // 需要计算的向量的大小:2^14
  printf("Vector size:%d\n", nElem);
  int nByte = sizeof(float) * nElem;
  float *a_h = (float *)malloc(nByte);
  float *b_h = (float *)malloc(nByte);
  float *res_h = (float *)malloc(nByte);
  float *res_from_gpu_h = (float *)malloc(nByte);
  memset(res_h, 0, nByte);
  memset(res_from_gpu_h, 0, nByte);
  // =============================================
  // 分配GPU空间
  float *a_d, *b_d, *res_d;
  CHECK(cudaMalloc((float **)&a_d, nByte));
  CHECK(cudaMalloc((float **)&b_d, nByte));
  CHECK(cudaMalloc((float **)&res_d, nByte));
  // =============================================
  // 初始化数组a和b为随机数
  initialData(a_h, nElem);
  initialData(b_h, nElem);
  // =============================================
  // 复制数据到GPU
  CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));
  // =============================================
  // gridSize表示有多少个block, blockSize表示每个block有多少个thread(最大的线程数好像不能超过1024个)
  dim3 block(1024);
  dim3 grid(nElem / block.x);
  sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d);
  printf("Execution configuration<<<%d,%d>>>\n", grid.x, block.x);
  // =============================================
  // 将GPU结果拷贝回主机
  // 比较CPU和GPU结果
  CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
  sumArrays(a_h, b_h, res_h, nElem);
  checkResult(res_h, res_from_gpu_h, nElem);
  // =============================================
  // 释放内存
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(a_h);
  free(b_h);
  free(res_h);
  free(res_from_gpu_h);

  return 0;
}

// 矩阵储存方式
#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"

__global__ void printThreadIndex(float *A, const int nx, const int ny)
{

  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int idx = iy * nx + ix;

  printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d)"
         "global index %2d ival %f\n",
         threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}
int main(int argc, char **argv)
{
  initDevice(0);
  int nx = 8, ny = 6; // 定义8*6的矩阵
  int nxy = nx * ny;  // 总的元素数量
  int nBytes = nxy * sizeof(float);

  // Malloc
  float *A_host = (float *)malloc(nBytes); // 主机端分配内存
  initialData(A_host, nxy);                // 矩阵随机
  printMatrix(A_host, nx, ny);             // 打印初始矩阵内容

  // cudaMalloc
  float *A_dev = NULL;
  CHECK(cudaMalloc((void **)&A_dev, nBytes)); // GPU分配内存

  cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice);

  dim3 block(4, 2);
  dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);

  printf("grid(%d,%d) block(%d,%d)\n", grid.x, grid.y, block.x, block.y);
  printThreadIndex<<<grid, block>>>(A_dev, nx, ny);

  CHECK(cudaDeviceSynchronize());
  cudaFree(A_dev);
  free(A_host);

  cudaDeviceReset();
  return 0;
}

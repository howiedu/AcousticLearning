// 这段代码用于研究线程和核心的关系。
// 每一个线程送到GPU之后，都要有一个索引，有了这个索引才能准确的拿到数据
// 因此，数据送到每个核心中都要有编号

#include <cuda_runtime.h>
#include <stdio.h>
__global__ void checkIndex(void)
{
  printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d) gridDim(%d,%d,%d)\n",
    threadIdx.x, threadIdx.y, threadIdx.z, 
    blockIdx.x, blockIdx.y, blockIdx.z, 
    blockDim.x, blockDim.y, blockDim.z, 
    gridDim.x, gridDim.y, gridDim.z);
}
int main(int argc,char **argv)
{
  dim3 block(1,2,3);  // 每个block有几个线程
  dim3 grid(2,1,1);  // 计算得到需要几个核
  printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
  printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
  checkIndex<<<grid,block>>>();
  cudaDeviceReset();
  return 0;
}

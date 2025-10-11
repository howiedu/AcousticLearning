// 与上一个代码基本一致，显示计算时间
#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"

void sumArrays(float * a,float * b,float * res,const int size)
{
  for(int i=0;i<size;i+=4)
  {
    res[i]=a[i]+b[i];
    res[i+1]=a[i+1]+b[i+1];
    res[i+2]=a[i+2]+b[i+2];
    res[i+3]=a[i+3]+b[i+3];
  }
}
__global__ void sumArraysGPU(float*a,float*b,float*res,int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i < N)
    res[i]=a[i]+b[i];
}
int main(int argc,char **argv)
{
  // set up device
  initDevice(0);

  int nElem=1<<24;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *a_h=(float*)malloc(nByte);
  float *b_h=(float*)malloc(nByte);
  float *res_h=(float*)malloc(nByte);
  float *res_from_gpu_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);
  memset(res_from_gpu_h,0,nByte);

  float *a_d,*b_d,*res_d;
  CHECK(cudaMalloc((float**)&a_d,nByte));
  CHECK(cudaMalloc((float**)&b_d,nByte));
  CHECK(cudaMalloc((float**)&res_d,nByte));

  initialData(a_h,nElem);
  initialData(b_h,nElem);

  CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((nElem-1)/block.x+1);

  //timer
  double iStart,iElaps;
  printf("Execution Configure<<<%d,%d>>>\n",grid.x,block.x);
  // 计算GPU耗时
  // =================================================================
  // 只计时核函数执行
  iStart = cpuSecond();
  sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, nElem);
  CHECK(cudaDeviceSynchronize()); // 等待GPU完成
  iElaps = cpuSecond() - iStart;
  printf("GPU Compute Time: %f sec\n", iElaps);
  // 单独计时数据传输
  iStart = cpuSecond();
  CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
  iElaps = cpuSecond() - iStart;
  printf("Data Transfer Time: %f sec\n", iElaps);
  // =================================================================
  // 计算CPU耗时
  iStart = cpuSecond();
  sumArrays(a_h,b_h,res_h,nElem);
  iElaps = cpuSecond() - iStart;
  printf("CPU Time elapsed %f sec\n", iElaps);

  checkResult(res_h,res_from_gpu_h,nElem);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(a_h);
  free(b_h);
  free(res_h);
  free(res_from_gpu_h);

  return 0;
}

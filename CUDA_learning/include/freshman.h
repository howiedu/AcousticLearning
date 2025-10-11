#ifndef FRESHMAN_H
#define FRESHMAN_H
// ================================================================
// 封装CUDA函数调用，自动检查执行状态

// 出错时显示文件名、行号、错误代码和描述信息

// 便于CUDA程序的调试和错误定位

#define CHECK(call)                                                    \
  {                                                                    \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess)                                          \
    {                                                                  \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                         \
    }                                                                  \
  }
// ================================================================
// 为Windows系统实现了gettimeofday函数

// 提供了cpuSecond() 函数获取高精度时间戳

// 用于性能测试和代码计时
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#ifdef _WIN32
int gettimeofday(struct timeval *tp, void *tzp)
{
  time_t clock;
  struct tm tm;
  SYSTEMTIME wtm;
  GetLocalTime(&wtm);
  tm.tm_year = wtm.wYear - 1900;
  tm.tm_mon = wtm.wMonth - 1;
  tm.tm_mday = wtm.wDay;
  tm.tm_hour = wtm.wHour;
  tm.tm_min = wtm.wMinute;
  tm.tm_sec = wtm.wSecond;
  tm.tm_isdst = -1;
  clock = mktime(&tm);
  tp->tv_sec = clock;
  tp->tv_usec = wtm.wMilliseconds * 1000;
  return (0);
}
#endif
double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
// ================================================================
// initialData(): 初始化float类型数组为随机数

// initialData_int(): 初始化int类型数组为随机数

// 使用时间作为随机种子，确保每次运行数据不同
void initialData(float *ip, int size)
{
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++)
  {
    ip[i] = (float)(rand() & 0xffff) / 1000.0f;
  }
}
void initialData_int(int *ip, int size)
{
  time_t t;
  srand((unsigned)time(&t));
  for (int i = 0; i < size; i++)
  {
    ip[i] = int(rand() & 0xff);
  }
}
// ================================================================
// printMatrix(): 以矩阵格式打印二维数组

// 支持自定义行列数，便于调试和结果验证
void printMatrix(float *C, const int nx, const int ny)
{
  float *ic = C;
  printf("Matrix<%d,%d>:\n", ny, nx);
  for (int i = 0; i < ny; i++)
  {
    for (int j = 0; j < nx; j++)
    {
      printf("%6f ", ic[j]);
    }
    ic += nx;
    printf("\n");
  }
}

// ================================================================
// initDevice(): 初始化指定编号的CUDA设备

// 显示设备信息并设置为当前使用设备
void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));
}
// ================================================================
// checkResult(): 比较CPU和GPU计算结果

// 使用容差值(1.0E-8)进行浮点数比较

// 提供详细的错误信息输出
void checkResult(float *hostRef, float *gpuRef, const int N)
{
  double epsilon = 1.0E-8;
  for (int i = 0; i < N; i++)
  {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon)
    {
      printf("Results don\'t match!\n");
      printf("%f(hostRef[%d] )!= %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
      return;
    }
  }
  printf("Check result success!\n");
}
#endif // FRESHMAN_H

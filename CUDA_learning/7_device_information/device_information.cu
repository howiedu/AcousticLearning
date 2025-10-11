// 查看GPU设备信息
#include <cuda_runtime.h>
#include <stdio.h>
// 添加这个函数来计算每个SM的CUDA核心数
int getCoresPerSM(int major, int minor)
{
        // 根据计算能力版本返回每个SM的CUDA核心数
        struct SMVersion
        {
                int major, minor, cores;
        };

        SMVersion archCores[] = {
            {9, 0, 128}, // Hopper
            {8, 9, 128}, // Ada Lovelace (RTX 4090)
            {8, 6, 128}, // Ampere (RTX 3060)
            {7, 5, 64},  // Turing (RTX 20系列)
            {7, 0, 64},  // Volta
            {6, 1, 128}, // Pascal (GTX 10系列)
            {6, 0, 64},  // Pascal
            {5, 3, 128}, // Maxwell
            {5, 2, 128}, // Maxwell
            {5, 0, 128}, // Maxwell
            {3, 7, 192}, // Kepler
            {3, 5, 192}, // Kepler
            {3, 0, 192}, // Kepler
            {2, 1, 48},  // Fermi
            {2, 0, 32},  // Fermi
            {0, 0, 0}    // 结束标记
        };

        for (int i = 0; archCores[i].major != 0; i++)
        {
                if (archCores[i].major == major && archCores[i].minor == minor)
                {
                        return archCores[i].cores;
                }
        }
        printf("  Unknown GPU architecture! Using default 64 cores/SM\n");
        return 64; // 默认值
}
int main(int argc, char **argv)
{
        printf("%s Starting ...\n", argv[0]);
        int deviceCount = 0;
        // =======================================
        // 获取设备个数
        cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
        if (error_id != cudaSuccess)
        {
                printf("cudaGetDeviceCount returned %d\n ->%s\n",
                       (int)error_id, cudaGetErrorString(error_id));
                printf("Result = FAIL\n");
                exit(EXIT_FAILURE);
        }
        if (deviceCount == 0)
        {
                printf("There are no available device(s) that support CUDA\n");
        }
        else
        {
                printf("Detected %d CUDA Capable device(s)\n", deviceCount);
        }
        // =======================================
        // 获取设备名称
        int dev = 0, driverVersion = 0, runtimeVersion = 0;
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("Device %d:\"%s\"\n", dev, deviceProp.name);
        // =======================================
        // 获取驱动和运行时版本
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version         %d.%d  /  %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:   %d.%d\n",
               deviceProp.major, deviceProp.minor);
        // =======================================
        // 🆕 新增：计算并显示CUDA核心总数
        int coresPerSM = getCoresPerSM(deviceProp.major, deviceProp.minor);
        int totalCores = coresPerSM * deviceProp.multiProcessorCount;
        printf("  Number of Multiprocessors (SM):              %d\n", deviceProp.multiProcessorCount);
        printf("  CUDA Cores per SM:                           %d\n", coresPerSM);
        printf("  🎯 TOTAL CUDA CORES:                         %d\n", totalCores);
        // =======================================

        // 获取设备内存（显存）
        printf("  Total amount of global memory:                %.2f GBytes (%zu bytes)\n",
               (float)deviceProp.totalGlobalMem / pow(1024.0, 3), deviceProp.totalGlobalMem);
        // =======================================
        // 获取GPU时钟频率，内存带宽，L2缓存
        printf("  GPU Clock rate:                               %.0f MHz (%0.2f GHz)\n",
               deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
        printf("  Memory Bus width:                             %d-bits\n",
               deviceProp.memoryBusWidth);
        if (deviceProp.l2CacheSize)
        {
                printf("  L2 Cache Size:                            	%d bytes\n",
                       deviceProp.l2CacheSize);
        }
        // =======================================
        // 纹理内存最大维度
        printf("  Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
               deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Max Layered Texture Size (dim) x layers       1D=(%d) x %d,2D=(%d,%d) x %d\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);
        // =======================================
        // 常量内存大小、共享内存大小、寄存器大小、线程数大小
        printf("  Total amount of constant memory               %lu bytes\n",
               deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:      %lu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block:%d\n",
               deviceProp.regsPerBlock);
        printf("  Wrap size:                                    %d\n", deviceProp.warpSize);
        // =======================================
        // 每个处理器硬件的最大线程块数(核数），线程块的最大取值（1024）
        printf("  Maximun number of thread per multiprocesser:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximun number of thread per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        // =======================================
        // 线程的最大尺寸、块的最大尺寸、最大连续性内存
        printf("  Maximun size of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Maximun size of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximu memory pitch                           %lu bytes\n", deviceProp.memPitch);
        // =======================================
        printf("----------------------------------------------------------\n");
        printf("Number of multiprocessors:                      %d\n", deviceProp.multiProcessorCount); // 多处理器数量。核心数=mps*coresPerSM
        printf("Total amount of constant memory:                %4.2f KB\n",
               deviceProp.totalConstMem / 1024.0);
        printf("Total amount of shared memory per block:        %4.2f KB\n",
               deviceProp.sharedMemPerBlock / 1024.0);
        printf("Total number of registers available per block:  %d\n",
               deviceProp.regsPerBlock);
        printf("Warp size                                       %d\n", deviceProp.warpSize);
        printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of warps per multiprocessor:     %d\n",
               deviceProp.maxThreadsPerMultiProcessor / 32);
        return EXIT_SUCCESS;
}

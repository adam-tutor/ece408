#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
using namespace std;
#define TILE_WIDTH 16

//__constant__ float MASK_CONST[64*64];
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int WblockRow = ceil((float)(W_out)/TILE_WIDTH);
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    extern __shared__ float shared_input[];
    float* shared_input_ptr = &shared_input[0];
    int shared_block_width = S*(TILE_WIDTH - 1) + K;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define shared_3d(i2, i1, i0) shared_input[(i2) * (shared_block_width * shared_block_width) + (i1) * (shared_block_width) + i0]
    
    // Insert your GPU convolution kernel code here
    int w = TILE_WIDTH * (blockIdx.z % WblockRow) + threadIdx.x;
    int h = TILE_WIDTH * (blockIdx.z / WblockRow) + threadIdx.y;
    int w_s = TILE_WIDTH * (blockIdx.z % WblockRow);
    int h_s = TILE_WIDTH * (blockIdx.z / WblockRow);
    int b = blockIdx.x;
    int m = blockIdx.y;

    for(int c = 0; c < C; c++){
        for(int i = threadIdx.y; i < shared_block_width; i += TILE_WIDTH){
            for(int j = threadIdx.x; j < shared_block_width; j += TILE_WIDTH){
                if(((h_s * S + i) < H) && ((w_s * S + j) < W)){
                    shared_3d(c, i, j) = in_4d(b, c, (h_s * S) + i, (w_s * S) + j);
                }
                else{
                    shared_3d(c, i, j) = 0;
                }
            }
        }
    }
    __syncthreads();

    if(h < H_out && w < W_out){
        float value = 0;
        for(int c = 0; c < C; c++){
            for(int p = 0; p < K; p++){
                for(int q = 0; q < K; q++){
                    value += shared_3d(c, threadIdx.y * S + p, threadIdx.x * S + q) * mask_4d(m, c, p, q);
                }
            }
        }
        out_4d(b, m, h, w) = value;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef shared_3d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
   const int H_out = (H - K)/S + 1;
   const int W_out = (W - K)/S + 1;

   cudaMalloc((void**) device_output_ptr, B * M * H_out * W_out * sizeof(float));
   cudaMalloc((void**) device_input_ptr, B * C * H * W * sizeof(float));
   cudaMalloc((void**) device_mask_ptr, M * C * K * K * sizeof(float));

   cudaMemcpy(*device_input_ptr, host_input, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(*device_mask_ptr, host_mask, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice);
   //cudaMemcpyToSymbol(MASK_CONST, host_mask, M * C * K * K * sizeof(float), 0, cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    int W_grid = ceil((float)((W - K)/S + 1)/TILE_WIDTH);
    int H_grid = ceil((float)((H - K)/S + 1)/TILE_WIDTH);
    int Y_grid = H_grid * W_grid;
    const int shared_block_width = (TILE_WIDTH - 1)*S + K;

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, M, Y_grid);
    int kernel_Size = C * shared_block_width * shared_block_width;

    conv_forward_kernel<<<dimGrid, dimBlock, kernel_Size * sizeof(float)>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, (B * M * ((H - K)/S + 1) * ((W - K)/S + 1))*sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

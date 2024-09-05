#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
using namespace std;
#define TILE_WIDTH 16

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

    __shared__ float tile_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_mask[TILE_WIDTH][TILE_WIDTH];

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int WblockRow = ceil((float)(W_out)/TILE_WIDTH);
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define shared_3d(i2, i1, i0) input[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + (i0)]

    // Insert your GPU convolution kernel code here
    //int w = TILE_WIDTH * (blockIdx.z % WblockRow) + threadIdx.x;
    //int h = TILE_WIDTH * (blockIdx.z / WblockRow) + threadIdx.y;
    int b = blockIdx.z;
    //int m = blockIdx.x;
    int h_Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int w_Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0;


    for (int i = 0; i < ceil((float)(C * K * K)/TILE_WIDTH); i++){
        int index_x = i * TILE_WIDTH + threadIdx.x;
        int index_y = i * TILE_WIDTH + threadIdx.y;
        if ((h_Row < M) && (index_x) < (C * K * K)){ 
            tile_mask[threadIdx.y][threadIdx.x] = mask_4d(h_Row, (index_x/(K*K)), (index_x%(K*K))/K, (index_x%(K*K))%K);
        }
        else{
            tile_mask[threadIdx.y][threadIdx.x] = 0;
        }
        if (index_y < (C * K * K) && w_Col < (H_out * W_out)) {
            tile_input[threadIdx.y][threadIdx.x] = shared_3d(b, index_y, w_Col); 
        }
        else{
            tile_input[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        if (h_Row < M && w_Col < (H_out * W_out)){ 
            for (int j = 0; j < TILE_WIDTH; j++){
                value += tile_mask[threadIdx.y][j] * tile_input[j][threadIdx.x];
            }
        }
        __syncthreads(); 
    }

    if (h_Row < M && w_Col < (H_out * W_out)){
        out_4d(b, h_Row, w_Col/W_out, w_Col % W_out) = value; 
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef shared_3d
}

__global__ void unroll(const float* device_input, float* device_input_u, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
    #define in_4d(i3, i2, i1, i0) device_input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define shared_3d(i2, i1, i0) device_input_u[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + (i0)]
    //loads into shared memory 
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    if ((blockIdx.x * blockDim.x + threadIdx.x) < C * H_out * W_out) {
        //each thread in the block is going to calculate a value to store into shared
        int c = (blockIdx.x * blockDim.x + threadIdx.x) / (H_out * W_out);
        int w_u = (blockIdx.x * blockDim.x + threadIdx.x) % (H_out * W_out);
        int h_out = w_u / W_out;
        int w_out = w_u % W_out;

        for(int p = 0; p < K; p++){
            for(int q = 0; q < K; q++) {
                int h_u = (c * K * K) + (p * K) + q;
                if ((h_out * S + p) < H && (w_out * S + q) < W){
                    shared_3d(blockIdx.y, h_u, w_u) = in_4d(blockIdx.y, c, (h_out * S + p), (w_out * S + q)); //same thing as tiled shared mem
                }
                else{
                    shared_3d(blockIdx.y, h_u, w_u) = 0.0f;
                }
            }
        }
    }

    #undef in_4d
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
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int W_grid = ceil((1.0 * H_out * W_out)/TILE_WIDTH);
    int H_grid = ceil((1.0 * M)/TILE_WIDTH);
    //int Y_grid = H_grid * W_grid;

    float *device_input_u;
    cudaMalloc((void **)&device_input_u, B * C * K * K * H_out * W_out * sizeof(float));

    dim3 dimGrid_u(ceil((C * H_out * W_out) / (1.0 * 1024)), B, 1);
    dim3 dimBlock_u(1024, 1, 1);
    unroll<<<dimGrid_u, dimBlock_u>>>(device_input, device_input_u, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(W_grid, H_grid, B);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input_u, device_mask, B, M, C, H, W, K, S);
    cudaDeviceSynchronize();
    cudaFree(device_input_u);
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

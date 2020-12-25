#include "image.h"

#ifdef CUDA_ENABLED
CUDA_GLOBAL void cudaResetImageKernel(vec3* pixels, int nx, int ny)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= nx) || (j >= ny))
		return;
	int pixelIndex = j * nx + i;
	pixels[pixelIndex] = vec3(0.0f, 0.0f, 0.0f);
}
#endif	// CUDA_ENABLED

#ifdef CUDA_ENABLED
void Image::cudaResetImage()
{
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	cudaResetImageKernel << <blocks, threads >> > (pixels, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
#endif	// CUDA_ENABLED

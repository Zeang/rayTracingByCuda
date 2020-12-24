#pragma once

#include "vec3.h"
#include "util.h"

struct Image
{
#ifdef CUDA_ENABLED
	vec3* pixels;
#else
	vec3** pixels;
#endif
	uint32_t* windowPixels;
	uint8_t* fileOutputImage;

	const int nx;
	const int ny;
	const int tx;
	const int ty;

	bool showWindow;
	bool writeImage;

	CUDA_HOSTDEV Image(bool showWindow, bool writeImage, int x, int y, int tx, int ty) : showWindow(showWindow), writeImage(writeImage), nx(x), ny(y), tx(tx), ty(ty)
	{
#ifdef CUDA_ENABLED
		int pixelCount = nx * ny;
		size_t pixelsFrameBufferSize = pixelCount * sizeof(vec3);
		size_t windowPixelsFrameBufferSize = pixelCount * sizeof(uint32_t);
		size_t fileOutputImageFrameBufferSize = 3 * pixelCount * sizeof(uint8_t);

		// allocate Frame Buffers
		checkCudaErrors(cudaMallocManaged((void**)&pixels, pixelsFrameBufferSize));
		checkCudaErrors(cudaMallocManaged((void**)&windowPixels, windowPixelsFrameBufferSize));
		checkCudaErrors(cudaMallocManaged((void**)&fileOutputImage, fileOutputImageFrameBufferSize));
#else
		pixels = new vec3 * [nx];
		for (int i = 0; i < nx; ++i)
			pixels[i] = new vec3[ny];

		if (showWindow)
			windowPixels = new uint32_t[nx * ny];
		
		if (writeImage)
			fileOutputImage = new uint8_t[nx * ny * 3];
#endif	// CUDA_ENABLED
	}

#ifdef CUDA_ENABLED
	void cudaResetImage();
#endif	// CUDA_ENABLED

	CUDA_HOSTDEV void resetImage()
	{
#ifdef CUDA_ENABLED
		cudaResetImage();
#else
#pragma omp parallel for
		for (int i = 0; i < nx * ny; ++i)
		{
			pixel[i / ny][i % ny] = vec3(0, 0, 0);
		}
#endif	// CUDA_ENABLED
	}

	CUDA_HOSTDEV ~Image()
	{
#ifdef CUDA_ENABLED
		checkCudaErrors(cudaFree(pixels));
		checkCudaErrors(cudaFree(windowPixels));
		checkCudaErrors(cudaFree(fileOutputImage));
#else
		for (int i = 0; i < nx; ++i)
			delete[] pixels[i];
		delete[] pixels;

		if (showWindow)
			delete[] windowPixels;

		if (writeImage)
			delete[] fileOutputImage;
#endif	// CUDA_ENABLED
	}
};

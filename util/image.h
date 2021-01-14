#pragma once

#include "vec3.h"
#include "util.h"

#ifdef OIDN_ENABLED
#include <OpenImageDenoise/oidn.hpp>
#endif

struct Image
{
#ifdef CUDA_ENABLED
	vec3* pixels;
	vec3* pixels2;
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

#ifdef OIDN_ENABLED
	oidn::DeviceRef device;
	oidn::FilterRef filter;
#endif	// OIDN_ENABLED

	CUDA_HOSTDEV Image(bool showWindow, bool writeImage, int x, int y, int tx, int ty) : showWindow(showWindow), writeImage(writeImage), nx(x), ny(y), tx(tx), ty(ty)
	{
#ifdef CUDA_ENABLED
		int pixelCount = nx * ny;
		size_t pixelsFrameBufferSize = pixelCount * sizeof(vec3);
		size_t windowPixelsFrameBufferSize = pixelCount * sizeof(uint32_t);
		size_t fileOutputImageFrameBufferSize = 3 * pixelCount * sizeof(uint8_t);

		// allocate Frame Buffers
		checkCudaErrors(cudaMallocManaged((void**)&pixels, pixelsFrameBufferSize));
		checkCudaErrors(cudaMallocManaged((void**)&pixels2, pixelsFrameBufferSize));
		checkCudaErrors(cudaMallocManaged((void**)&windowPixels, windowPixelsFrameBufferSize));
		checkCudaErrors(cudaMallocManaged((void**)&fileOutputImage, fileOutputImageFrameBufferSize));
#else
		pixels = new vec3[nx * ny];
		pixels2 = new vec3[nx * ny];

		if (showWindow)
			windowPixels = new uint32_t[nx * ny];
		
		if (writeImage)
			fileOutputImage = new uint8_t[nx * ny * 3];
#endif	// CUDA_ENABLED

#ifdef OIDN_ENABLED
		// Create an Open Image Denoise device
		device = oidn::newDevice();
		device.commit();
		
		// Create a denoising filter
		filter = device.newFilter("RT");	// generic ray tracing filter
		filter.setImage("color", pixels2, oidn::Format::Float3, nx, ny);
		filter.setImage("output", pixels2, oidn::Format::Float3, nx, ny);
		filter.set("hdr", true);	// image is HDR
		filter.commit();
#endif
	}

#ifdef OIDN_ENABLED
	void denoise()
	{
		// Filter the image
		filter.execute();

		// Check for errors
		const char* errorMessage;
		if (device.getError(errorMessage) != oidn::Error::None)
			std::cout << "Error: " << errorMessage << std::endl;
	}
#endif	// OIDN_ENABLED

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

	void SavePfm()
	{
		FILE* f = fopen("wtf.pfm", "wb");
		fprintf(f, "PF\n%d %d\n-1\n", nx, ny);
		fwrite(pixels2, sizeof(float), nx * ny * 3, f);
		fclose(f);
	}

	CUDA_HOSTDEV ~Image()
	{
#ifdef CUDA_ENABLED
		checkCudaErrors(cudaFree(pixels));
		checkCudaErrors(cudaFree(pixels2));
		checkCudaErrors(cudaFree(windowPixels));
		checkCudaErrors(cudaFree(fileOutputImage));
#else
		delete[] pixels;
		delete[] pixels2;

		if (showWindow)
			delete[] windowPixels;

		if (writeImage)
			delete[] fileOutputImage;
#endif	// CUDA_ENABLED
	}
};

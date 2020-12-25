#include "common.h"
#include "renderer.h"
#include "scene.cuh"
#include "window.h"
#include "camera.h"

#include "../hitables/sphere.h"
#include "../hitables/hitable_list.h"
#include "../materials/material.h"

CUDA_DEV int numHitables = 102;

#ifdef CUDA_ENABLED
void initializeWorldCuda(bool showWindow, bool writeImagePPM,
	bool writeImagePNG, hitable*** list, hitable** world, Window** w,
	Image** image, camera** cam, Renderer** renderer)
{
	int choice = 3;

	switch (choice)
	{
	case 0:
		numHitables = 4;
		break;
	case 1:
		numHitables = 58;
		break;
	case 2:
		numHitables = 901;
		break;
	case 3:
		numHitables = 102;
		break;
	case 4:
		numHitables = 68;
		break;
	}

	// World
	checkCudaErrors(cudaMallocManaged(list, numHitables * sizeof(hitable*)));
	hitable** worldPtr;
	checkCudaErrors(cudaMallocManaged(&worldPtr, sizeof(hitable*)));
	switch (choice)
	{
	case 0:
		simpleScene << <1, 1 >> > (*list, worldPtr);
		break;
	case 1:
		simpleScene2 << <1, 1 >> > (*list, worldPtr);
		break;
	case 2:
		randomScene << <1, 1 >> > (*list, worldPtr);
		break;
	case 3:
		randomScene2 << <1, 1 >> > (*list, worldPtr);
		break;
	case 4:
		randomScene3 << <1, 1 >> > (*list, worldPtr);
		break;
	}
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	*world = *worldPtr;
	checkCudaErrors(cudaFree(worldPtr));

	// camera
	vec3 lookFrom(13.0f, 2.0f, 3.0f);
	vec3 lookAt(0.0f, 0.0f, 0.0f);
	checkCudaErrors(cudaMallocManaged(cam, sizeof(camera)));
	new (*cam)camera(lookFrom, lookAt, vec3(0.0f, 1.0f, 0.0f),
		20.0f, float(nx) / float(ny), distToFocus);

	// Renderer
	checkCudaErrors(cudaMallocManaged(renderer, sizeof(Renderer)));
	new(*renderer) Renderer(showWindow, writeImagePPM, writeImagePNG);

	// Image
	checkCudaErrors(cudaMallocManaged(image, sizeof(Image)));
	new(*image) Image(showWindow, writeImagePPM || writeImagePNG,
		nx, ny, tx, ty);

	// Window
	if (showWindow)
		*w = new Window(*cam, *renderer, nx, ny, thetaInit, phiInit,
			zoomScale, stepScale);
}

CUDA_GLOBAL void freeWorldCuda(hitable** list, hitable** world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		for (int i = 0; i < numHitables; ++i)
		{
			delete ((sphere*)list[i])->mat_ptr;
			delete list[i];
		}
		delete* world;
	}
}

void destroyWorldCuda(bool showWindow, hitable** list, hitable* world, Window* w, Image* image, camera* cam, Renderer* render)
{
	freeWorldCuda << <1, 1 >> > (list, &world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	// checkCudaErrors(cudaFree(list));
	// checkCudaErrors(cudaFree(world));
	checkCudaErrors(cudaFree(w));
	checkCudaErrors(cudaFree(image));
	checkCudaErrors(cudaFree(cam));
	checkCudaErrors(cudaFree(render));

	cudaDeviceReset();
}

CUDA_GLOBAL void render(camera* cam, Image* image, hitable* world,
	Renderer* render, int sampleCount)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= image->nx) || (j >= image->ny))
		return;

	int pixelIndex = j * image->nx + i;

	for (int s = 0; s < nsBatch; ++s)
	{
		RandomGenerator rng(sampleCount * nsBatch + s, i * image->nx + j);
		float u = float(i + rng.get1f()) / float(image->nx);	// left to right
		float v = float(j + rng.get1f()) / float(image->ny);	// botton to top
		ray r = cam->get_ray(rng, u, v);

		image->pixels[pixelIndex] += render->color(rng, r, world, 0);
	}
	
	vec3 col = image->pixels[pixelIndex] / (sampleCount * nsBatch);
	image->pixels2[pixelIndex] = col;
}
#endif	// CUDA_ENABLED

CUDA_GLOBAL void display(Image* image)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int pixelIndex = j * image->nx + i;

	vec3 col = image->pixels2[pixelIndex];
	// Gamma encoding of images is used to optimize the usage of bits
	// when encoding an image, or bandwidth used to transport an image,
	// by taking advantage of the non-linear manner in which humans perceive
	// light and color. (wikipedia)

	// we use gamma 2: raising the color to the power 1/gamma (1/2)
	col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

	int ir = clamp(int(255.f * col[0]), 0, 255);
	int ig = clamp(int(255.99f * col[1]), 0, 255);
	int ib = clamp(int(255.99f * col[2]), 0, 255);

	if (image->writeImage)
	{
		// PNG
		int index = (image->ny - 1 - j) * image->nx + i;
		int index3 = 3 * index;

		image->fileOutputImage[index3 + 0] = ir;
		image->fileOutputImage[index3 + 1] = ig;
		image->fileOutputImage[index3 + 2] = ib;
	}

	if (image->showWindow)
		image->windowPixels[(image->ny - j - 1) * image->nx + i] = (ir << 16) | (ig << 8) | (ib);
}

#ifdef CUDA_ENABLED
void Renderer::cudaRender(uint32_t* windowPixels, camera* cam,
	hitable* world, Image* image, int sampleCount, uint8_t* fileOutputImage)
{
	dim3 blocks((image->nx + image->tx - 1) / image->tx, (image->ny + image->ty - 1) / image->ty);
	dim3 threads(image->tx, image->ty);

	// Kernel call for the computation of pixel colors.
	render << <blocks, threads >> > (cam, image, world, this, sampleCount);

	// Denoise here.
#ifdef OIDN_ENABLED
	checkCudaErrors(cudaDeviceSynchronize());
	image->denoise();
	checkCudaErrors(cudaDeviceSynchronize());
#endif	// OIDN_ENABLED
	// Kernel call to fill the output buffers.
	display << <blocks, threads >> > (image);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
#endif	// CUDA_ENABLED

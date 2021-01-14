#pragma once
#define CUDA_ENABLED
#define OIDN_ENABLED
#define CUDA_DEBUG

#ifdef CUDA_ENABLED
	#include <cuda_runtime_api.h>
	#include <cuda.h>
	#include <curand_kernel.h>
#endif

#ifdef CUDA_DEBUG
#include <chrono>
#endif

#include <string>
#include <limits.h>

#ifdef CUDA_ENABLED
#define CUDA_HOST __host__
#else
#define CUDA_HOST
#endif

#ifdef CUDA_ENABLED
#define CUDA_GLOBAL __global__
#else
#define CUDA_GLOBAL
#endif

#ifdef CUDA_ENABLED
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

#ifdef CUDA_ENABLED
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

const int nx = 1280;
const int ny = 720;
const int ns = 64;
const int nsDenoise = 64;
static int imageNr = 0;
const int sampleNrToWrite = 1;
const int sampleNrToWriteDenoise = sampleNrToWrite;
const std::string folderName = "output";
const std::string fileName = "raytracer";

#ifdef CUDA_ENABLED
const int nsBatch = 4;
#else
const int nsBatch = 1;
#endif

const int tx = 16;
const int ty = 16;
const int benchmarkCount = 100;
const float thetaInit = 1.34888f;
const float phiInit = 1.32596f;
const float zoomScale = 0.5f;
const float stepScale = 0.5f;

const float distToFocus = 10.0f;
const float aperture = 0.1f;

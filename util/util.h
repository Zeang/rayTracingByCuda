#pragma once

#include "common.h"
#include "vec3.h"
#include <ios>
#include <iomanip>
#include <sstream>

template <typename T>
CUDA_HOSTDEV T clamp(const T& n, const T& lower, const T& upper)
{
	T min = n < upper ? n : upper;
	return lower > min ? lower : min;
}

#ifdef CUDA_ENABLED
#ifndef checkCudaErrors
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__)
#endif
#endif

#ifdef CUDA_ENABLED
CUDA_HOST void check_cuda(cudaError_t result, const char* const func, const char* const file, const int line);
#endif

inline std::string formatNumber(int n)
{
	std::ostringstream out;
	out << std::internal << std::setfill('0') << std::setw(4) << n;
	return out.str();
}

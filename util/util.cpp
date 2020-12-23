#include "util.h"

#ifdef CUDA_ENABLED
CUDA_HOST void check_cuda(cudaError_t result, const char* const func, const char* const file, const int line)
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
#endif
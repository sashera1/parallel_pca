#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <cuda_runtime.h>

#define gpuErrorCheck(result) {gpuAssert((result), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif
#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <cuda_runtime.h>

#define checkKernel(result) {gpuAssert((result), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    //include else statement if suspected that gpuAssert isnt working
    //else{
    //    printf("No error detected with gpuAssert: %s line %d\n", file, line);
    //}
}
#endif
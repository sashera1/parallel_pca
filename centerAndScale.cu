#include <cstdlib> //for malloc
#include <iostream> //for cout
#include "util.h" 

#include "centerAndScale.h"

__global__ void centerAndScaleKernel(float* inputMatrix, float* outputMatrix, int rowCount, int colCount){
    //naive implementation:
    //one thread per column
    //block of 32 threads (1 warp)
    //likley max rows = ~768 (assuming 96kb)
    //absolute max rows = 792 (assuming 99kb)
    extern __shared__ float column[]; //stored in row-major format; ie for block 0, 0-31 are col 0-31 of row 0, 32-63 are col 0-31 of row 1, etc
    float mean = 0;
    unsigned int threadIdGlobal = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdGlobal < colCount){
        for (unsigned int row = 0; row < rowCount; ++row){
            float val = inputMatrix[row*colCount + threadIdGlobal];
            mean+=val;
            column[row*blockDim.x + threadIdx.x] = val;
        }
    
        //at this point, shared mem is populated with raw values

        //calculate mean
        mean = mean / (float)rowCount;
        float standardDevInv = 0;

        //go thru and for each val
        //accumulate difference square between val mean
        //and subtract mean from val in sm
        for (unsigned int row = 0; row < rowCount; ++row){
            float val = column[row*blockDim.x + threadIdx.x];
            float dif = (val-mean);
            column[row*blockDim.x + threadIdx.x] = dif;
            standardDevInv += dif * dif;
        }

        standardDevInv = rsqrtf(standardDevInv / (float)rowCount); 
        //consider: "When scaling, always add a tiny "epsilon" value inside your square root" 1e-5f
        
        //read shared mem values, multiply by standardDevInv, writeback to output matrix
        for (unsigned int row=0; row < rowCount; ++row){
            float val = column[row*blockDim.x + threadIdx.x];
            outputMatrix[row*colCount + threadIdGlobal] = val * standardDevInv;
        }
    }
}

float* centerAndScaleWrapper(float* inputMatrixHost, int rowCount, int colCount, bool debugMode){
    
    if (debugMode){
        std::cout << "Entered centerAndScale file\n";
    }

    size_t matrixSize = rowCount*colCount*sizeof(float)
    
    unsigned int threadsPerBlock = 32;
    unsigned int blockCount = (colCount + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int sharedMemoryPerBlock = threadsPerBlock * rowCount * sizeof(float);

    if (debugMode){
        std::count << "Requesting " << sharedMemoryPerBlock / 1024.0f << "KB shared memory per block\n";
    }

    float* inputMatrixDevice;
    cudaMalloc((void**)&inputMatrixDevice, matrixSize);
    cudaMemcpy(inputMatrixDevice,inputMatrixHost,matrixSize,cudaMemcpyHostToDevice);

    float* scaledMatrixDevice;
    cudaMalloc((void**)&scaledMatrixDevice, matrixSize);

    centerAndScaleKernel<<<blockCount,threadsPerBlock,sharedMemoryPerBlock>>>(
        inputMatrixDevice,scaledMatrixDevice,rowCount,colCount);

    if (debugMode){
        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());
    }
    else {
        gpuErrorCheck(cudaGetLastError());
    }

    cudaFree(inputMatrixDevice);

    //do not free or return scaledMatrixDevice: use as input for next kernel

    //pointer to scaled matrix in device
    return scaledMatrixDevice;
}


#include <cstdlib> //for malloc
#include <iostream> //for cout

#include "centerAndScale.h"

__global__ void centerAndScaleKernel(float* inputMatrix, float* outputMatrix, int rowCount, int colCount){
    //naive implementation:
    //one thread per column
    //block of 32 threads (1 warp)
    //max columns = ~1500
    extern __shared__ float column[]; //stored in row-major format; ie for block 0, 0-31 are col 0-31 of row 0, 32-63 are col 0-31 of row 1, etc
    float mean = 0;
    unsigned int threadIdGlobal = BlockIdx.x * BlockDim.x + threadIdx.x;
    if (threadIdGlobal < colCount){
        for (unsigned int row = 0; row < rowCount; ++row){
            float val = inputMatrix[row*rowCount + threadIdGlobal];
            mean+=val;
            column[row*rowCount + threadIdx.x] = val;
        }
    
    //at this point, shared mem is populated with raw values

    //calculate mean

    //go thru and for each val
    //accumulate difference square between val mean
    //and subtract mean from val in sm
    


    //once done, divide accumulation of squared differences by rowCount-1 to get standard d
    //then go thru and dvidie each val by standard dev
    }




}

float* centerAndScaleWrapper(float* inputMatrixHost, int rowCount, int colCount, bool debugMode){
    
    if (debugMode){
        std::cout << "Entered centerAndScale file\n";
    }
    
    float *scaledMatrixHost = (float *) malloc(rowCount*colCount*sizeof(float));

    unsigned int threadsPerBlock = 32;
    unsigned int blockCount = (colCount + threadsPerBlock - 1) / threadsPerBlock;

    unsigned int sharedMemoryPerBlock = threadsPerBlock * rowCount * sizeof(float);

    float* inputMatrixDevice;
    cudaMalloc((void**)&inputMatrixDevice, rowCount*colCount*sizeof(float));
    cudaMemcpy(inputMatrixDevice,inputMatrixHost,rowCount*colCount*sizeof(float),cudaMemcpyHostToDevice);

    float* scaledMatrixDevice;
    cudaMalloc((void**)&scaledMatrixDevice, rowCount*colCount*sizeof(float));

    centerAndScaleKernel<<blockCount,threadsPerBlock,sharedMemoryPerBlock>>(inputMatrixDevice,scaledMatrixDevice,rowCount,colCount);

    cudaMemcpy(inputMatrixHost,inputMatrixDevice,rowCount*colCount*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(inputMatrixDevice);
    cudaFree(scaledMatrixDevice);

    return scaledMatrixHost;
}


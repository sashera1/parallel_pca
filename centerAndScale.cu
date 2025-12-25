#include <cstdlib> //for malloc
#include <iostream> //for cout

#include "centerAndScale.h"

__global__ void centerAndScaleKernel(int* inputMatrix, int* outputMatrix, int rowCount, int colCount){
    //naive implementation:
    //one thread per column
    //block of 32 threads (1 warp)
    //max columns = ~3000
    extern __shared__ int column[]; //stored in row-major format; ie for block 0, 0-31 are col 0-31 of row 0, 32-63 are col 0-31 of row 1, etc
    int sumOfVals = 0;
    unsigned int threadIdGlobal = BlockIdx.x * BlockDim.x + threadIdx.x;
    if (threadIdGlobal < colCount){
        for (unsigned int row = 0; row < rowCount; ++row){
            int val = inputMatrix[row*rowCount + threadIdGlobal];
            sumOfVals+=val;
            column[row*rowCount + threadIdx.x] = val;
        }
    
    //at this point, shared mem is populated with raw values

    //go thru and for each val
    //accumulate difference square between val mean
    //and subtract mean from val in sm
    int sumOfDifSquared = 0;
        for (unsigned int row = 0; row < rowCount; ++row){

        }


    //once done, divide accumulation of squared differences by rowCount-1 to get standard d
    //then go thru and dvidie each val by standard dev
    }




}

int* centerAndScaleWrapper(int* inputMatrixHost, int rowCount, int colCount, bool debugMode){
    
    if (debugMode){
        std::cout << "Entered centerAndScale file\n";
    }
    
    int *scaledMatrixHost = (int *) malloc(rowCount*colCount*sizeof(int));

    unsigned int threadsPerBlock = 32;
    unsigned int blockCount = (colCount + threadsPerBlock - 1) / threadsPerBlock;

    unsigned int sharedMemoryPerBlock = threadsPerBlock * rowCount * sizeof(int);

    int* inputMatrixDevice;
    cudaMalloc((void**)&inputMatrixDevice, rowCount*colCount*sizeof(int));
    cudaMemcpy(inputMatrixDevice,inputMatrixHost,rowCount*colCount*sizeof(int),cudaMemcpyHostToDevice);

    int* scaledMatrixDevice;
    cudaMalloc((void**)&scaledMatrixDevice, rowCount*colCount*sizeof(int));

    centerAndScaleKernel<<blockCount,threadsPerBlock,sharedMemoryPerBlock>>(inputMatrixDevice,scaledMatrixDevice,rowCount,colCount);

    cudaMemcpy(inputMatrixHost,inputMatrixDevice,rowCount*colCount*sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(inputMatrixDevice);
    cudaFree(scaledMatrixDevice);

    return scaledMatrixHost;
}


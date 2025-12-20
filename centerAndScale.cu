#include <cstdlib> //for malloc
#include <iostream> //for cout

#include "centerAndScale.h"

__global__ void centerAndScaleKernel(int* inputMatrix, int* outputMatrix, int rowCount, int colCount){


}

int* centerAndScaleWrapper(int* inputMatrixHost, int rowCount, int colCount, bool debugMode){
    
    if (debugMode){
        std::cout << "Entered centerAndScale file\n";
    }
    
    int *scaledMatrixHost = (int *) malloc(rowCount*colCount*sizeof(int));

    int threadsNeeded = colCount; //one thread per column: may need more than this




    return scaledMatrixHost;
}


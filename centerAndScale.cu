#include <cstdlib> //for malloc
#include <iostream> //for cout

#include "centerAndScale.h"

int* centerAndScaleWrapper(int* inputMatrixHost, int rowCount, int colCount, bool debugMode){
    
    if (debugMode){
        std::cout << "Entered centerAndScale file\n";
    }
    
    int *scaledMatrixHost = (int *) malloc(rowCount*colCount*sizeof(int));



    return scaledMatrixHost;
}


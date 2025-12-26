#include <stdio.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream> //for cout, cerr, endl
#include <fstream> //for ifstream
#include <cstdlib> //for malloc, exit

#include "centerAndScale.h" //first kernel wrapper: center and scale the matrix


float* loadMatrix(std::string matrixFile, unsigned int &rowCount, unsigned int &colCount){
    std::ifstream infile(matrixFile);

    if (!infile.is_open()){
        std::cerr << "Error opening " << matrixFile << std::endl;
        exit(1);
    }

    if (!(infile >> rowCount >> colCount)) {
        std::cerr << "Issue reading matrix dimenstions\n";
        exit(1);
    }

    std::cout << "Reading " << matrixFile << ", of dimensions " << rowCount << " * " << colCount << std::endl;

    float *matrix = (float *) malloc(rowCount*colCount*sizeof(float));

    for (unsigned int r = 0; r < rowCount; r++){
        for (unsigned int c = 0; c < colCount; c++){
            infile >> matrix[r*colCount + c];
        }
    }

    return matrix;

}


int main(int argc, char**argv){
    cudaDeviceSynchronize(); //pick up any errors from past runs
    setbuf(stdout, NULL); //advisable for better debugging output

    if (argc < 3){
        std::cout << "usage: " << argv[0] << " <input file> <dim for pca> [-d]\n";
        return 1;
    }

    bool debugMode = false;
    std::string matrixFile;
    unsigned int targetDimension = 0 ;

    for (unsigned int argIter = 1; argIter<argc; ++argIter){
        std::string flag = argv[argIter];

        if (flag=="-d" || flag=="-debug"){
            debugMode = true;
        }
        else if (matrixFile.empty()){
            matrixFile = flag;
        }
        else{
            targetDimension = std::stoi(flag);
        }
    }

    if (debugMode){
        std::cout << "Debug mode on\n";
    }
    else
    {
        std::cout << "Debug mode off\n";
    }

    std::cout << "Performing PCA on " << matrixFile << ", finding " << targetDimension << " principle components\n";
    

    unsigned int rowCount, colCount;

    float* inputMatrixHost = loadMatrix(matrixFile, rowCount, colCount);

    if (debugMode){
        std::cout << "Verifying correct input matrix allocation, printing elements for first row:\n";
        for (unsigned int e = 0; e < colCount; ++e){
            std::cout << inputMatrixHost[e] << ", ";
        }
        std::cout << std::endl;
    }

    
    
    //NEXT STEPS

    //kernel 1
    //center the data - do mean AND variance / standard deviation
    float* scaledMatrixDevice = centerAndScaleWrapper(inputMatrixHost, rowCount, colCount, debugMode);

    if(debugMode){
        std::cout << "Successfully exited center and scale\n"
    }

    //kernel 2
    //covariance matrix

    //kernel 3
    //find eigenvalues and eigenvectors

    //kernel 4
    //project the data


    //at the end, free allocated memory
    free(inputMatrixHost);

    //may make sense to put this within a kernel, as it is created within a kernel
    //dont know when it stops being needed though, so for now put at end
    cudaFree(scaledMatrixDevice);

    return 0;
}
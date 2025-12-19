#include <stdio.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream> //for cout, cerr, endl
#include <fstream> //for ifstream
#include <cstdlib> //for mallc, exit


int* loadMatrix(std::string matrixFile, unsigned int &rowCount, unsigned int &colCount){
    std::ifstream infile(matrixFile);

    if (!infile.is_open()){
        std::cerr << "Error opening " << matrixFile << std::endl;
        exit(1);
    }

    if (!(infile >> rowCount >> colCount)) {
        std::cerr << "Issue reading matrix dimenstions" << std::endl;
        exit(1);
    }

    std::cout << "Reading " << matrixFile << ", of dimensions " << rowCount << " * " << colCount << std::endl;

    int *matrix = (int *) malloc(rowCount*colCount*sizeof(int));

    for (unsigned int r = 0; r < rowCount; r++){
        for (unsigned int c = 0; c < colCount; c++){
            infile >> matrix[r*colCount + c];
        }
    }

    return matrix;

};


int main(int argc, char**argv){
    cudaDeviceSynchronize(); //pick up any errors from past runs
    setbuf(stdout, NULL); //advisable for better debugging output

    if (argc < 3){
        std::cout << "usage: " << argv[0] << " <input file> <dim for pca> [-d]" << std::endl;
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
        std::cout << "Debug mode on" << std::endl;
    }
    else
    {
        std::cout << "Debug mode off" << std::endl;
    }

    std::cout << "Performing PCA on " << matrixFile << ", finding " << targetDimension << " principle components" << std::endl;
    

    unsigned int rowCount, colCount;

    int* inputMatrixHost = loadMatrix(matrixFile, rowCount, colCount);

    if (debugMode){
        std::cout << "Verifying correct input matrix allocation, printing elements for first row:" << std::endl;
        for (unsigned int e = 0; e < colCount; ++e){
            std::cout << inputMatrixHost[e] << ", ";
        }
        std::cout << std::endl;
    }


    //at the end, free allocated memory
    free(inputMatrixHost);
    return 0;
}
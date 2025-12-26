from pathlib import Path 
import sys
import numpy as np
from datetime import datetime

def generateMatrixWriteFile(fileName:str, rowCount: int, colCount: int):

    #generate matrix
    matrix = np.random.standard_normal((rowCount,colCount)).astype(np.float32)

    colMeans = np.random.normal(loc=0.0, scale = 50.0, size=colCount).astype(np.float32)
    colStandardDeviations = np.random.gamma(shape=3.0, scale=7.0, size=colCount).astype(np.float32)

    matrix = (matrix * colStandardDeviations) + colMeans #performs element-wise, not matrix, mul. colMeans is "stretched/duplicated" vertically

    #create columns correlating
    correlations = max(1, colCount // 20)

    for _ in range(correlations):
        src = np.random.choice(colCount)
        for _ in range(np.random.randint(low=2, high=5)):
            target = np.random.choice(colCount)

            if target==src:
                continue

            multiplier = np.random.normal(loc=0.0, scale=1.5)
            offset = np.random.normal(loc=0.0, scale=25)

            matrix[:, target] = (matrix[:, src] * multiplier) + offset #target col as rescaled and shifted copy of src col

            targetStandardDev = np.std(matrix[:, target]) 
            noise_level = targetStandardDev * np.random.uniform(0.1,0.5) 
            noise = np.random.normal(loc=0.0, scale=noise_level, size=rowCount).astype(np.float32)
            matrix[:, target] += noise

    #begin file writing process
    dataFolderDirectory = Path(__file__).parent

    if fileName=="": #if need to generate file name
        fileName = f"{rowCount}_by_{colCount}_matrix"
    filePath = dataFolderDirectory / (fileName+".txt")

    if filePath.exists(): #add timestamp when file with same name already exists so we dont overwrite
        dateAndTime = datetime.now()
        timestamp = dateAndTime.strftime("%m-%d-%y_%I.%M%p")
        fileName += timestamp
        filePath = dataFolderDirectory / (fileName+".txt")

    with open(filePath, 'w') as f:
        f.write(f"{rowCount}\n{colCount}\n")
        np.savetxt(f, matrix, fmt='%.5f',delimiter=' ')


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage (choose matrix name):")
        print("python generateData.py <generatedFileName> <rowCount> <columnCount>")
        print("Usage (generate matrix name automatically):")
        print("python -a <row_count> <column_count")
        sys.exit(1)
    
    if (sys.argv[1]=="-a" or sys.argv[1]=="-A"):
        fileName=""
    else:
        fileName = sys.argv[1]

    rowCount = int(sys.argv[2])
    colCount = int(sys.argv[3])

    generateMatrixWriteFile(fileName,rowCount, colCount)
    

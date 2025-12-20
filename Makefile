NVCC = nvcc
TARGET = pca
SRC = main.cu centerAndScale.cu

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
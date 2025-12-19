NVCC = nvcc
TARGET = pca
SRC = main.cu

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
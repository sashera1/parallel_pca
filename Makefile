NVCC = nvcc
TARGET = pca
SRC = main.cu centerAndScale.cu
HEADERS = util.h

$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
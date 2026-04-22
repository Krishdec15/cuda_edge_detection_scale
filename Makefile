# Define the compiler and paths
CUDA_PATH ?= /usr/local/cuda
NPP_SAMPLES_PATH ?= /usr/share/doc/nvidia-cuda-samples

NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++
CXXFLAGS = -std=c++17 -O3 -g \
           -I$(CUDA_PATH)/include \
           -Iinclude \
           -I$(NPP_SAMPLES_PATH)/examples/Common/UtilNPP \
           -I$(NPP_SAMPLES_PATH)/examples/Common \
           `pkg-config --cflags opencv4`

LDFLAGS = -L$(CUDA_PATH)/lib64 \
          -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc \
          -lfreeimage \
          `pkg-config --libs opencv4`

SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data

SRC_MAIN = $(SRC_DIR)/main.cpp
TARGET_EXEC = $(BIN_DIR)/run_filter

all: $(TARGET_EXEC)

$(TARGET_EXEC): $(SRC_MAIN) $(wildcard $(SRC_DIR)/*.hpp)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(SRC_MAIN) -o $(TARGET_EXEC) $(LDFLAGS)

run_image: $(TARGET_EXEC)
	./$(TARGET_EXEC) --input $(DATA_DIR)/sample.png --output $(DATA_DIR)/sample_edges.png

run_video: $(TARGET_EXEC)
	./$(TARGET_EXEC) --input $(DATA_DIR)/sample.mp4 --output $(DATA_DIR)/sample_edges.mp4

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "Available commands:"
	@echo "  make           - Build the executable"
	@echo "  make run_image - Run on sample image"
	@echo "  make run_video - Run on sample video"
	@echo "  make clean     - Remove compiled binaries"

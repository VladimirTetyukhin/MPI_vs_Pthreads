# Refactored build for the hybrid MPI/OpenMP/OpenACC matrix benchmark.
#
# Examples:
#   make
#   make USE_CBLAS=1 CBLAS_LIBS="-lopenblas"
#   make USE_CUBLAS=1 CUDA_HOME=/usr/local/cuda
#
# The original program required an MPI compiler and OpenACC-capable compiler.

CC ?= mpicc
TARGET ?= hybrid_refactored
BUILD_DIR ?= build
SRC_DIR := src
INC_DIR := include

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

CFLAGS ?= -O3 -std=c11 -Wall -Wextra -fopenmp -fopenacc
CPPFLAGS += -I$(INC_DIR)
LDLIBS += -lm

ifeq ($(USE_CBLAS),1)
CPPFLAGS += -DUSE_CBLAS
LDLIBS += $(CBLAS_LIBS)
endif

ifeq ($(USE_CUBLAS),1)
CPPFLAGS += -DUSE_CUBLAS
CUDA_HOME ?= /usr/local/cuda
CPPFLAGS += -I$(CUDA_HOME)/include
LDFLAGS += -L$(CUDA_HOME)/lib64
LDLIBS += -lcublas -lcudart
endif

.PHONY: all clean print-sources

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/hybrid_common.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

print-sources:
	@printf '%s\n' $(SRCS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

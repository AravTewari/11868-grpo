NVCC = nvcc
CFLAGS = -shared -Xcompiler -fPIC -O2
TARGET = grpo_kernels.so
SRC = grpo_kernels.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

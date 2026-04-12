# External parameters
opt ?= 3# Optimization level
impl ?= seq# Implementation (seq, omp, cuda)
cuda_arch ?= native# CUDA target (native, v100s)
size_w ?= 256# Width
size_h ?= $(size_w)# Height (blank means equal to width)
kernel_size ?= 26# Convolution kernel size
precision ?= 64# Floating point precision, 64, 32 or 16 (16 is really slow on CPU because no native instructions)
gif ?=# Set to non-blank to enable gif output (still need to pass gif=filename.gif in argv)

# Compiler
CC := nvcc

# Compiler flags
LDLIBS := -lm
CFLAGS := \
	-Wno-deprecated-gpu-targets -Xcompiler --openmp,-Wall \
	-O$(opt) -DFEAT_IMPL_$(shell echo $(impl) | tr a-z A-Z) \
	-DFEAT_SIZE_W=$(size_w) -DFEAT_SIZE_H=$(size_h) \
	-DFEAT_KERNEL_SIZE=$(kernel_size) -DFEAT_PRECISION=$(precision)
PRINT_PREFIX := \
	opt=$(opt) impl=$(impl) \
	size_w=$(size_w) size_h=$(size_h) \
	kernel_size=$(kernel_size) precision=$(precision)

ifeq ($(cuda_arch),native)
	CFLAGS += -arch=native
endif
ifeq ($(cuda_arch),v100s)
	CFLAGS += -gencode=arch=compute_70,code=sm_70
endif

ifneq ($(gif),)
	CFLAGS += -DFEAT_GIF
	PRINT_PREFIX += gif=y
endif

CFLAGS += -DPRINT_PREFIX='"$(PRINT_PREFIX)"'

# File locations
SRC := src_c
DST_DIR := build
DST := $(DST_DIR)/$(shell printf '%s\0' $(CFLAGS) $(LDFLAGS) | md5sum | awk '{ print $$1 }')

# Source files
SRCS := $(SRC)/main.c $(SRC)/orbium.c
ifeq ($(impl),seq)
	SRCS += $(SRC)/impl_cpu.c
endif
ifeq ($(impl),omp)
	SRCS += $(SRC)/impl_cpu.c
endif
ifeq ($(impl),cuda)
	SRCS += $(SRC)/impl_lenia.cu
endif
ifneq ($(gif),)
	SRCS += $(SRC)/gifenc.c
endif

# Object files
OBJS_1 := $(patsubst $(SRC)/%.c,$(DST)/%.o,$(SRCS))
OBJS := $(patsubst $(SRC)/%.cu,$(DST)/%.o,$(OBJS_1))

# Dynamically generated dependency includes
DEPS := $(OBJS:.o=.d)

# Executable name
TARGET := $(DST)/lenia.out

.PHONY: all printexe3 clean

all: $(TARGET)

printexe3: $(TARGET)
	@echo $^ >&3

$(TARGET): $(OBJS) | $(DST)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(DST)/%.o: $(SRC)/%.cu | $(DST)
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(DST)/%.o: $(SRC)/%.c | $(DST)
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(DST):
	@mkdir -p $@
	@printf "Signature: 8a477f597d28d172789f06886806bc55\n" > $(DST_DIR)/CACHEDIR.TAG

-include $(DEPS)

clean:
	rm -rf $(DST_DIR)

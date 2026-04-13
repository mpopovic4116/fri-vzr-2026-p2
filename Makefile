# External parameters
opt ?= 3# Optimization level
impl ?= seq# Implementation (seq, omp, cuda)
cuda_arch ?= native# CUDA target (native, v100s)
size_w ?= 256# Width
size_h ?= $(size_w)# Height (blank means equal to width)
kernel_size ?= 26# Convolution kernel size
precision ?= 64# Floating point precision, 64, 32 or 16 (16 is really slow on CPU because no native instructions)
simd ?= y# Explicitly vectorize some loops, even in impl=seq
toroid_impl ?= halo# Alternate ways of implementing wrapping, CPU only for now (naive, sections, halo)
kernel ?= default# kernel implementation (default, shared, fused)
toroid ?= bitwise# toroidal wrap with: (mod, bitwise)
unroll ?=# Set to non-blank to use loop unrolling
threads_x ?= 32# number of threads for x dim.
threads_y ?= $(threads_x)# number of threads for y dim (blank means equal to threads_x).
gif ?=# Set to non-blank to enable gif output (still need to pass gif=filename.gif in argv)

# Compiler
CC := gcc
NVCC := nvcc

# Compiler flags
LDLIBS := -lm
CCFLAGS := --openmp -Wall -std=gnu11
NVCCFLAGS := -Wno-deprecated-gpu-targets -Xcompiler --openmp,-Wall
CFLAGS := \
	-O$(opt) -DFEAT_IMPL_$(shell echo $(impl) | tr a-z A-Z) \
	-DFEAT_SIZE_W=$(size_w) -DFEAT_SIZE_H=$(size_h) \
	-DFEAT_KERNEL_SIZE=$(kernel_size) -DFEAT_PRECISION=$(precision) \
	-DFEAT_THREAD_X=$(threads_x) -DFEAT_THREAD_Y=$(threads_y)
PRINT_PREFIX := \
	opt=$(opt) impl=$(impl) \
	size_w=$(size_w) size_h=$(size_h) \
	kernel_size=$(kernel_size) precision=$(precision)

IS_CPU :=
ifeq ($(impl),seq)
	IS_CPU := y
endif
ifeq ($(impl),omp)
	IS_CPU := y
endif

ifeq ($(IS_CPU),y)
ifneq ($(simd),)
	CFLAGS += -DFEAT_SIMD
	PRINT_PREFIX += simd=y
endif

ifeq ($(toroid_impl),naive)
	CFLAGS += -DFEAT_TOROID_IMPL_NAIVE
	PRINT_PREFIX += toroid_impl=naive
endif
ifeq ($(toroid_impl),sections)
	CFLAGS += -DFEAT_TOROID_IMPL_SECTIONS
	PRINT_PREFIX += toroid_impl=sections
endif
ifeq ($(toroid_impl),halo)
	CFLAGS += -DFEAT_TOROID_IMPL_HALO
	PRINT_PREFIX += toroid_impl=halo
endif
endif

ifeq ($(cuda_arch),native)
	NVCCFLAGS += -arch=native
endif
ifeq ($(cuda_arch),v100s)
	NVCCFLAGS += -gencode=arch=compute_70,code=sm_70
endif

ifeq ($(impl),cuda)
PRINT_PREFIX += threads=$(threads_x)x$(threads_y)

ifeq ($(kernel),default)
	CFLAGS += -DFEAT_DEFAULT_IMPL
	PRINT_PREFIX += kernel=default
endif
ifeq ($(kernel),shared)
	CFLAGS += -DFEAT_SHARED_IMPL
	PRINT_PREFIX += kernel=shared
endif
ifeq ($(kernel),fused)
	CFLAGS += -DFEAT_FUSED_IMPL
	PRINT_PREFIX += kernel=fused
endif

ifneq ($(unroll),)
    CFLAGS += -DFEAT_UNROLL
    PRINT_PREFIX += unroll=y
endif

ifeq ($(toroid),bitwise)
	CFLAGS += -DFEAT_BITWISE_MASK
	PRINT_PREFIX += toroid=bitwise
else
	PRINT_PREFIX += toroid=mod
endif
endif

ifneq ($(gif),)
	CFLAGS += -DFEAT_GIF
	PRINT_PREFIX += gif=y
endif

CFLAGS += -DPRINT_PREFIX='"$(PRINT_PREFIX)"'

# File locations
SRC := src_c
DST_DIR := build
DST := $(DST_DIR)/$(shell printf '%s\0' $(CCFLAGS) $(NVCCFLAGS) $(CFLAGS) $(LDFLAGS) | md5sum | awk '{ print $$1 }')

# Source files
SRCS := $(SRC)/main.c $(SRC)/orbium.c
ifeq ($(impl),seq)
	SRCS += $(SRC)/impl_cpu.c
endif
ifeq ($(impl),omp)
	SRCS += $(SRC)/impl_cpu.c
endif
ifeq ($(impl),cuda)
	SRCS += $(SRC)/impl_cuda.cu
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
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(DST)/%.o: $(SRC)/%.cu | $(DST)
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) -MD -MP -c -o $@ $<

$(DST)/%.o: $(SRC)/%.c | $(DST)
	$(CC) $(CCFLAGS) $(CFLAGS) -MD -MP -c -o $@ $<

$(DST):
	@mkdir -p $@
	@printf "Signature: 8a477f597d28d172789f06886806bc55\n" > $(DST_DIR)/CACHEDIR.TAG

-include $(DEPS)

clean_single:
	rm -rf $(DST)

clean:
	rm -rf $(DST_DIR)

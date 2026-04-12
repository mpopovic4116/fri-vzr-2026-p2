#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "float_alias.h"
#include "helper_cuda.h"
#include "impl.h"
#include "orbium.h"

#define ROWS FEAT_SIZE_H
#define COLS FEAT_SIZE_W

#define TILE_W 32
#define TILE_H 32

struct lenia_impl_state
{
    int pixels;
    int grid_size;
    int kernel_size_memory;
    dim3 threadsPerBlock;
    dim3 numBlocks;
    fhost *w_host;
    fhost *world_host;
    fcuda *world_device;
    fhost *tmp_host;
    fcuda *tmp_device;
    uint8_t *world_bytes_host;
    uint8_t *world_bytes_device;
    int local_memory_size;
};

static fhost gauss(fhost x, fhost mu, fhost sigma)
{
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

static void generate_kernel(fhost *K, const unsigned int size)
{
    // Construct ring convolution filter
    fhost mu = 0.5;
    fhost sigma = 0.15;
    int r = size / 2;
    fhost sum = 0;
    if (K != NULL) {
        for (int y = -r; y < r; y++) {
            for (int x = -r; x < r; x++) {
                fhost distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                if (distance > 1) {
                    K[(y + r) * size + x + r] = 0; // Cut at d=1
                }
                sum += K[(y + r) * size + x + r];
            }
        }
        // Normalize
        for (unsigned int y = 0; y < size; y++) {
            for (unsigned int x = 0; x < size; x++) {
                K[y * size + x] /= sum;
            }
        }
    }
}

struct orbium_coo
{
    int row;
    int col;
    int angle;
};

struct lenia_impl_state *lenia_impl_init()
{

    // Allocate memory
    struct lenia_impl_state *state = (struct lenia_impl_state *) malloc(sizeof(struct lenia_impl_state));
    state->pixels = ROWS * COLS;
    state->grid_size = ROWS * COLS * sizeof(fcuda);
    state->kernel_size_memory = FEAT_KERNEL_SIZE * FEAT_KERNEL_SIZE * sizeof(fcuda);
    state->threadsPerBlock = dim3(TILE_W, TILE_H);
    state->local_memory_size = (TILE_W + FEAT_KERNEL_SIZE) * (TILE_W + FEAT_KERNEL_SIZE) * sizeof(fcuda);
    state->numBlocks = dim3((COLS + TILE_W - 1) / TILE_W, (ROWS + TILE_H - 1) / TILE_H);
    state->w_host = (fhost *) calloc(FEAT_KERNEL_SIZE * FEAT_KERNEL_SIZE, sizeof(fhost));
    state->world_host = (fhost *) calloc(ROWS * COLS, sizeof(fhost));
    state->tmp_host = (fhost *) calloc(ROWS * COLS, sizeof(fhost));
    state->world_bytes_host = (uint8_t *) calloc(ROWS * COLS, sizeof(uint8_t));
    checkCudaErrors(cudaMalloc((void **) &state->world_device, state->grid_size));
    checkCudaErrors(cudaMalloc((void **) &state->tmp_device, state->grid_size));
    checkCudaErrors(cudaMalloc((void **) &state->world_bytes_device, ROWS * COLS));

    // Generate convolution kernel
    generate_kernel(state->w_host, FEAT_KERNEL_SIZE);

    // Place orbiums
    struct orbium_coo orbiums[] = {{0, COLS / 3, 0}, {ROWS / 3, 0, 180}};
    for (unsigned int o = 0; o < sizeof(orbiums) / sizeof(*orbiums); o++) {
        place_orbium(state->world_host, ROWS, COLS, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    return state;
}

static __constant__ fcuda w_device_constant[FEAT_KERNEL_SIZE * FEAT_KERNEL_SIZE];

void lenia_impl_upload(struct lenia_impl_state *state)
{
    // copy mem host -> device
    checkCudaErrors(cudaMemcpy(state->world_device, state->world_host, state->grid_size, cudaMemcpyHostToDevice));
    // cudaMemcpy(w_device, w_host, kernel_size_memory, cudaMemcpyHostToDevice);
    checkCudaErrors(cudaMemcpyToSymbol(w_device_constant, state->w_host, state->kernel_size_memory, 0, cudaMemcpyHostToDevice)); // move kernel to constant memory
}

// #define w(r, c) (w[(r) * w_cols + (c)])
// #define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// gpu version of funct convolve2d
static __global__ void conv_kernel(fcuda *result, fcuda *input, int rows, int cols, int k_size)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // curr row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // curr col

    unsigned int w_cols = k_size;
    unsigned int w_rows = k_size;

    if (i < rows && j < cols) {
        fcuda sum = 0;

        for (int ki = 0; ki < w_rows; ki++) {
            int kri = w_rows - ki - 1;
            for (int kj = 0; kj < w_cols; kj++) {
                int kcj = w_cols - kj - 1;
                // kri and kcj because we fist flip the kernel
                sum += w_device_constant[(ki) *w_cols + (kj)] * input[(((i - w_rows / 2 + rows + kri)) % rows) * cols + (((j - w_cols / 2 + cols + kcj)) % cols)];
            }
        }
        result[i * cols + j] = sum;
    }
}
// improved version of conv_kernel, uses local(shared) memory
static __global__ void conv_kernel_shared(fcuda *result, fcuda *input, int rows, int cols, int k_size)
{
    extern __shared__ fcuda tile[];

    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;

    int r = k_size / 2;
    int shared_w = TILE_W + k_size;

    // load tile and halo into local mem
    for (int j = threadIdx.y; j < shared_w; j += TILE_H) {
        // blockIdx.x * TILE_W <- find ur tile
        // - r <- halo on one side, int shared_w = TILE_W + k_size; <- k_size is 2r, so here we get the other side of the halo
        // + i / + j <- increment in dir
        // + cols / + rows <- so we dont mod a neg num
#ifdef FEAT_BITWISE_MASK
        // bitwise toroidal wrap (should work as long as cols and rows is a power of 2)
        int global_y = (blockIdx.y * TILE_H - r + j + rows) & (rows - 1);
#else
        // toroidal wrap with %
        int global_y = (blockIdx.y * TILE_H - r + j + rows) % rows;
#endif

        for (int i = threadIdx.x; i < shared_w; i += TILE_W) {
#ifdef FEAT_BITWISE_MASK
            // bitwise toroidal wrap (should work as long as cols and rows is a power of 2)
            int global_x = (blockIdx.x * TILE_W - r + i + cols) & (cols - 1);
#else
            // toroidal wrap with %
            int global_x = (blockIdx.x * TILE_W - r + i + cols) % cols;
#endif

            tile[j * shared_w + i] = input[global_y * cols + global_x];
        }
    }

    __syncthreads();

    // the normal conv part
    if (y < rows && x < cols) {
        fcuda sum = 0;
#ifdef DFEAT_UNROLL
#pragma unroll 16
#endif
        for (int ki = 0; ki < k_size; ki++) {
            int kri = k_size - ki - 1;
#ifdef DFEAT_UNROLL
#pragma unroll 16
#endif
            for (int kj = 0; kj < k_size; kj++) {
                int kcj = k_size - kj - 1;
                // just fetch from local mem
                sum += w_device_constant[kri * k_size + kcj] * tile[(threadIdx.y + ki) * shared_w + (threadIdx.x + kj)];
            }
        }
        result[y * cols + x] = sum;
    }
}

static __global__ void byte_packing_kernel(uint8_t *dst, fcuda *src, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        fcuda val = src[idx];
        if (val < 0.0) val = 0.0;
        if (val > 1.0) val = 1.0;
        dst[idx] = (uint8_t) (val * 255.0f);
    }
}

// gpu version of funct gauss
static __device__ fcuda gauss_device(fcuda x, fcuda mu, fcuda sigma)
{
    return exp(-0.5 * ((x - mu) / sigma) * ((x - mu) / sigma));
}

// gpu world updating
static __global__ void update_kernel(fcuda *world, fcuda *tmp, int rows, int cols, fcuda dt)
{
    fcuda mu = 0.15;
    fcuda sigma = 0.015;

    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * cols + (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx < rows * cols) {
        fcuda u = tmp[idx];
        fcuda growth = -1.0 + 2.0 * gauss_device(u, mu, sigma);
        fcuda val = world[idx] + dt * growth;
        world[idx] = fcuda_fmin(1.0, fcuda_fmax(0.0, val));
    }
}

void lenia_impl_step(struct lenia_impl_state *state, fhost dt)
{
#ifdef FEAT_SHARED_MEM
    conv_kernel_shared<<<state->numBlocks, state->threadsPerBlock, state->local_memory_size>>>(
        state->tmp_device, state->world_device, ROWS, COLS, FEAT_KERNEL_SIZE);
#else
    conv_kernel<<<state->numBlocks, state->threadsPerBlock>>>(
        state->tmp_device, state->world_device, ROWS, COLS, FEAT_KERNEL_SIZE);
#endif

    update_kernel<<<state->numBlocks, state->threadsPerBlock>>>(state->world_device, state->tmp_device, ROWS, COLS, dt);
}

void lenia_impl_dump(struct lenia_impl_state *state, uint8_t *out_frame)
{
    byte_packing_kernel<<<state->numBlocks, state->threadsPerBlock>>>(state->world_bytes_device, state->world_device, ROWS, COLS);
    checkCudaErrors(cudaMemcpy(out_frame, state->world_bytes_device, state->pixels, cudaMemcpyDeviceToHost));
}

void lenia_impl_download(struct lenia_impl_state *state)
{
    byte_packing_kernel<<<state->numBlocks, state->threadsPerBlock>>>(state->world_bytes_device, state->world_device, ROWS, COLS);
    checkCudaErrors(cudaMemcpy(state->world_bytes_host, state->world_bytes_device, state->pixels, cudaMemcpyDeviceToHost));
}

void lenia_impl_free(struct lenia_impl_state *state)
{
    free(state->w_host);
    free(state->world_host);
    checkCudaErrors(cudaFree(state->world_device));
    free(state->tmp_host);
    checkCudaErrors(cudaFree(state->tmp_device));
    free(state->world_bytes_host);
    checkCudaErrors(cudaFree(state->world_bytes_device));
    free(state);
}

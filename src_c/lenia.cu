#include "gifenc.h"
#include "lenia.h"
#include "orbium.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Include CUDA headers
// #include <cuda_runtime.h>
// #include <cuda.h>

// Uncomment to generate gif animation
#define GENERATE_GIF

// uncomment to use cpu instead
// #define CPU

// For prettier indexing syntax
#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// Function to calculate Gaussian
inline double gauss(double x, double mu, double sigma)
{
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

// Function for growth criteria
double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma); // Baseline -1, peak +1
}

// Function to generate convolution kernel
double *generate_kernel(double *K, const unsigned int size)
{
    // Construct ring convolution filter
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0;
    if (K != NULL) {
        for (int y = -r; y < r; y++) {
            for (int x = -r; x < r; x++) {
                double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
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
    return K;
}

// Function to perform convolution on input using kernel w
// Note that the kernel is flipped for convolution as per definition, and we use modular indexing for toroidal world
inline double *convolve2d(double *result, const double *input, const double *w, const unsigned int rows, const unsigned int cols, const unsigned int w_rows, const unsigned int w_cols)
{
    if (result != NULL && input != NULL && w != NULL) {
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                double sum = 0;
                for (int ki = w_rows - 1, kri = 0; ki >= 0; ki--, kri++) {
                    for (int kj = w_cols - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                        sum += w(ki, kj) * input((i - w_rows / 2 + rows + kri), (j - w_cols / 2 + cols + kcj));
                    }
                }
                result[i * cols + j] = sum;
            }
        }
    }
    return result;
}
// gpu version of funct convolve2d
__global__ void conv_kernel(double *result, double *input, double *w, int rows, int cols, int k_size)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // curr row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // curr col

    unsigned int w_cols = k_size;
    unsigned int w_rows = k_size;

    if (i < rows && j < cols) {
        double sum = 0;
        int r = k_size / 2;

        for (int ki = 0; ki < w_rows; ki++) {
            int kri = w_rows - ki - 1;
            for (int kj = 0; kj < w_cols; kj++) {
                int kcj = w_cols - kj - 1;
                // kri and kcj because we fist flip the kernel
                sum += w[(ki) *w_cols + (kj)] * input[(((i - w_rows / 2 + rows + kri)) % rows) * cols + (((j - w_cols / 2 + cols + kcj)) % cols)];
            }
        }
        result[i * cols + j] = sum;
    }
}

// gpu version of funct gauss
__device__ double gauss_device(double x, double mu, double sigma)
{
    return exp(-0.5 * ((x - mu) / sigma) * ((x - mu) / sigma));
}

// gpu world updating
__global__ void update_kernel(double *world, double *tmp, int rows, int cols, double dt)
{
    double mu = 0.15;
    double sigma = 0.015;

    int idx = (blockIdx.y * blockDim.y + threadIdx.y) * cols + (blockIdx.x * blockDim.x + threadIdx.x);

    if (idx < rows * cols) {
        double u = tmp[idx];
        double growth = -1.0 + 2.0 * gauss_device(u, mu, sigma);
        double val = world[idx] + dt * growth;
        world[idx] = fmin(1.0, fmax(0.0, val));
    }
}

// Function to evolve Lenia
double *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums)
{

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif(
#ifdef CPU
        "lenia_cpu.gif", /* file name */
#else
        "lenia.gif", /* file name */
#endif
        cols, rows,      /* canvas size */
        inferno_pallete, /*pallete*/
        8,               /* palette depth == log2(# of colors) */
        -1,              /* no transparency */
        0                /* infinite loop */
    );
#endif

    int grid_size = rows * cols * sizeof(double);
    int kernel_size_memory = kernel_size * kernel_size * sizeof(double);

    // host memory alloc
    double *w_host = (double *) calloc(kernel_size * kernel_size, sizeof(double));
    double *world_host = (double *) calloc(rows * cols, sizeof(double));
    double *tmp_host = (double *) calloc(rows * cols, sizeof(double));
#ifndef CPU
    // device memory alloc
    double *world_device, *tmp_device, *w_device;
    cudaMalloc((void **) &world_device, grid_size);
    cudaMalloc((void **) &tmp_device, grid_size);
    cudaMalloc((void **) &w_device, kernel_size_memory);
#endif
    // Generate convolution kernel
    w_host = generate_kernel(w_host, kernel_size);

    // Place orbiums
    for (unsigned int o = 0; o < num_orbiums; o++) {
        world_host = place_orbium(world_host, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }
#ifdef CPU
    // Lenia Simulation
    for (unsigned int step = 0; step < steps; step++) {
        // Convolution
        tmp_host = convolve2d(tmp_host, world_host, w_host, rows, cols, kernel_size, kernel_size);

        // Evolution
        for (unsigned int i = 0; i < rows; i++) {
            for (unsigned int j = 0; j < cols; j++) {
                world_host[i * rows + j] += dt * growth_lenia(tmp_host[i * rows + j]);
                world_host[i * rows + j] = fmin(1, fmax(0, world_host[i * rows + j])); // Clip between 0 and 1

#ifdef GENERATE_GIF
                gif->frame[i * rows + j] = world_host[i * rows + j] * 255;
#endif
            }
        }
#ifdef GENERATE_GIF
        ge_add_frame(gif, 5);
#endif
    }
#else
    // copy mem host -> device
    cudaMemcpy(world_device, world_host, grid_size, cudaMemcpyHostToDevice);
    cudaMemcpy(w_device, w_host, kernel_size_memory, cudaMemcpyHostToDevice);

    // lets use 16x16 threads
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(cols / threadsPerBlock.x, rows / threadsPerBlock.y);
    // loop where we call the kernels
    for (unsigned int step = 0; step < steps; step++) {
        conv_kernel<<<numBlocks, threadsPerBlock>>>(tmp_device, world_device, w_device, rows, cols, kernel_size);
        update_kernel<<<numBlocks, threadsPerBlock>>>(world_device, tmp_device, rows, cols, dt);

#ifdef GENERATE_GIF
        cudaMemcpy(world_host, world_device, grid_size, cudaMemcpyDeviceToHost);
        for (unsigned int i = 0; i < rows * cols; i++) {
            gif->frame[i] = (uint8_t) (fmin(1.0, fmax(0.0, world_host[i])) * 255);
        }
        ge_add_frame(gif, 5);
#endif
    }
#endif

#ifdef GENERATE_GIF
    ge_close_gif(gif);
#endif
#ifndef CPU
    cudaFree(world_device);
    cudaFree(tmp_device);
    cudaFree(w_device);
#endif
    free(w_host);
    free(tmp_host);

    return world_host;
}

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "float_alias.h"
#include "impl.h"
#include "orbium.h"

#define ROWS FEAT_SIZE_H
#define COLS FEAT_SIZE_W

struct lenia_impl_state
{
    fhost *w;
    fhost *world;
    fhost *tmp;
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
    state->w = (fhost *) calloc(FEAT_KERNEL_SIZE * FEAT_KERNEL_SIZE, sizeof(fhost));
    state->world = (fhost *) calloc(ROWS * COLS, sizeof(fhost));
    state->tmp = (fhost *) calloc(ROWS * COLS, sizeof(fhost));

    // Generate convolution kernel
    generate_kernel(state->w, FEAT_KERNEL_SIZE);

    // Place orbiums
    struct orbium_coo orbiums[] = {{0, COLS / 3, 0}, {ROWS / 3, 0, 180}};
    for (unsigned int o = 0; o < sizeof(orbiums) / sizeof(*orbiums); o++) {
        place_orbium(state->world, ROWS, COLS, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    return state;
}

void lenia_impl_upload(struct lenia_impl_state *state) {} // No-op for cpu

#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

// Function to perform convolution on input using kernel w
// Note that the kernel is flipped for convolution as per definition, and we use modular indexing for toroidal world
static void convolve2d(fhost *result, const fhost *input, const fhost *w, const unsigned int rows, const unsigned int cols, const unsigned int w_rows, const unsigned int w_cols)
{
#ifdef FEAT_IMPL_OMP
#pragma omp for
#endif
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
            fhost sum = 0;
            for (int ki = w_rows - 1, kri = 0; ki >= 0; ki--, kri++) {
                for (int kj = w_cols - 1, kcj = 0; kj >= 0; kj--, kcj++) {
                    sum += w(ki, kj) * input((i - w_rows / 2 + rows + kri), (j - w_cols / 2 + cols + kcj));
                }
            }
            result[i * cols + j] = sum;
        }
    }
}

static fhost growth_lenia(fhost u)
{
    fhost mu = 0.15;
    fhost sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma); // Baseline -1, peak +1
}

void lenia_impl_step(struct lenia_impl_state *state, fhost dt)
{
#ifdef FEAT_IMPL_OMP
#pragma omp parallel
#endif
    {
        // Convolution
        convolve2d(state->tmp, state->world, state->w, ROWS, COLS, FEAT_KERNEL_SIZE, FEAT_KERNEL_SIZE);

        // Evolution
#ifdef FEAT_IMPL_OMP
#pragma omp for
#endif
        for (unsigned int i = 0; i < ROWS; i++) {
            for (unsigned int j = 0; j < COLS; j++) {
                state->world[i * ROWS + j] += dt * growth_lenia(state->tmp[i * ROWS + j]);
                state->world[i * ROWS + j] = fmin(1, fmax(0, state->world[i * ROWS + j])); // Clip between 0 and 1
            }
        }
    }
}

void lenia_impl_dump(const struct lenia_impl_state *state, uint8_t *out_frame)
{
    for (unsigned int i = 0; i < ROWS; i++) {
        for (unsigned int j = 0; j < COLS; j++) {
            out_frame[i * ROWS + j] = state->world[i * ROWS + j] * 255;
        }
    }
}

void lenia_impl_download(struct lenia_impl_state *state) {} // No-op for cpu

void lenia_impl_free(struct lenia_impl_state *state)
{
    free(state->w);
    free(state->world);
    free(state->tmp);
    free(state);
}

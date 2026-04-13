#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "float_alias.h"
#include "impl.h"
#include "orbium.h"

#define ROWS FEAT_SIZE_H
#define COLS FEAT_SIZE_W
#define KRNL FEAT_KERNEL_SIZE

#ifdef FEAT_TOROID_IMPL_HALO
#define PADDING (KRNL - 1)
#define PROWS (ROWS + PADDING * 2)
#define PCOLS (COLS + PADDING * 2)
#else
#define PADDING 0
#define PROWS ROWS
#define PCOLS COLS
#endif

struct lenia_impl_state
{
    fhost *w;
    fhost *world;
    fhost *tmp;
};

static fhost gauss(fhost x, fhost mu, fhost sigma)
{
    fhost z = (x - mu) / sigma;
    return exp(-0.5 * z * z);
}

static void generate_kernel(fhost *K, const unsigned int size)
{
    // Construct ring convolution filter
    fhost mu = 0.5;
    fhost sigma = 0.15;
    int r = size / 2;
    fhost sum = 0;
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
    // Rotate 180 degrees for consistency with reference implementation (our step does correlation instead of convolution)
    for (unsigned int y = 0; y < size; y++) {
        unsigned int y_other = size - 1 - y;
        for (unsigned int x = 0; x < size - y; x++) {
            unsigned int x_other = size - 1 - x;
            fhost other = K[y_other * size + x_other];
            K[y_other * size + x_other] = K[y * size + x];
            K[y * size + x] = other;
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
    state->w = (fhost *) calloc(KRNL * KRNL, sizeof(fhost));
    state->world = (fhost *) calloc(PROWS * PCOLS, sizeof(fhost));
    state->tmp = (fhost *) calloc(PROWS * PCOLS, sizeof(fhost));

    // Generate convolution kernel
    generate_kernel(state->w, KRNL);

    // Place orbiums
    struct orbium_coo orbiums[] = {{0, COLS / 3, 0}, {ROWS / 3, 0, 180}};
    for (unsigned int o = 0; o < sizeof(orbiums) / sizeof(*orbiums); o++) {
        place_orbium(state->world, PROWS, PCOLS, PADDING + orbiums[o].row, PADDING + orbiums[o].col, orbiums[o].angle);
    }

    return state;
}

void lenia_impl_upload(struct lenia_impl_state *state) {} // No-op for cpu

static fhost growth_lenia(fhost u)
{
    fhost mu = 0.15;
    fhost sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma); // Baseline -1, peak +1
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

static void kernel_universal(const fhost *w, const fhost *input, fhost *output, int x, int y, fhost dt)
{
    fhost sum = 0;
    for (int ky = 0; ky < KRNL; ky++) {
        int iy = (y - KRNL / 2 + ROWS + ky) % ROWS;
        for (int kx = 0; kx < KRNL; kx++) {
            int ix = (x - KRNL / 2 + COLS + kx) % COLS;
            sum += w[ky * KRNL + kx] * input[iy * COLS + ix];
        }
    }
    fhost val = input[y * PCOLS + x];
    val += dt * growth_lenia(sum);
    val = fhost_fmin(1, fhost_fmax(0, val));
    output[y * PCOLS + x] = val;
}

static void kernel_inner(const fhost *w, const fhost *input, fhost *output, int x, int y, fhost dt)
{
    fhost sum = 0;
    for (int ky = 0; ky < KRNL; ky++) {
        int iy = y - KRNL / 2 + ky;
        for (int kx = 0; kx < KRNL; kx++) {
            int ix = x - KRNL / 2 + kx;
            sum += w[ky * KRNL + kx] * input[iy * PCOLS + ix];
        }
    }
    fhost val = input[y * PCOLS + x];
    val += dt * growth_lenia(sum);
    val = fhost_fmin(1, fhost_fmax(0, val));
    output[y * PCOLS + x] = val;
}

#pragma GCC diagnostic pop

void lenia_impl_step(struct lenia_impl_state *state, fhost dt)
{
    fhost *w = state->w;
    fhost *input = state->world;
    fhost *output = state->tmp;

#ifdef FEAT_IMPL_OMP
#pragma omp parallel
#endif
    {
#if defined(FEAT_TOROID_IMPL_NAIVE)
#ifdef FEAT_IMPL_OMP
#pragma omp for collapse(2)
#endif
        for (unsigned int y = 0; y < ROWS; y++) {
            for (unsigned int x = 0; x < COLS; x++) {
                kernel_universal(w, input, output, x, y, dt);
            }
        }
#elif defined(FEAT_TOROID_IMPL_SECTIONS)
#ifdef FEAT_IMPL_OMP
#pragma omp for collapse(2)
#endif
        for (unsigned int y = 0; y < KRNL - 1; y++) { // Top + bottom (including corners)
            for (unsigned int x = 0; x < COLS; x++) {
                kernel_universal(w, input, output, x, y, dt);
                kernel_universal(w, input, output, x, ROWS - (KRNL - 1) + y, dt);
            }
        }

#ifdef FEAT_IMPL_OMP
#pragma omp for collapse(2)
#endif
        for (unsigned int y = KRNL - 1; y < ROWS - (KRNL - 1); y++) { // Left + right
            for (unsigned int x = 0; x < KRNL - 1; x++) {
                kernel_universal(w, input, output, x, y, dt);
                kernel_universal(w, input, output, COLS - (KRNL - 1) + x, y, dt);
            }
        }

#ifdef FEAT_IMPL_OMP
#pragma omp for collapse(2)
#endif
        for (unsigned int y = KRNL - 1; y < ROWS - (KRNL - 1); y++) { // Inner
            for (unsigned int x = KRNL - 1; x < COLS - (KRNL - 1); x++) {
                kernel_inner(w, input, output, x, y, dt);
            }
        }
#elif defined(FEAT_TOROID_IMPL_HALO)
#ifdef FEAT_IMPL_OMP
#pragma omp for collapse(2)
#endif
        for (int y = 0; y < PADDING; y++) { // Top + bottom (including corners)
            int dst_top = y * PCOLS;
            int dst_bot = (PADDING + ROWS + y) * PCOLS;
            int src_top = (PADDING + ((y - PADDING + ROWS) % ROWS)) * PCOLS;
            int src_bot = (PADDING + (y % ROWS)) * PCOLS;

            for (int x = 0; x < PCOLS; x++) {
                int src_x = PADDING + ((x - PADDING + COLS) % COLS);
                input[dst_top + x] = input[src_top + src_x];
                input[dst_bot + x] = input[src_bot + src_x];
            }
        }

#ifdef FEAT_IMPL_OMP
#pragma omp for collapse(2)
#endif
        for (int y = 0; y < ROWS; y++) { // Left + right
            int row = (PADDING + y) * PCOLS;
            for (int x = 0; x < PADDING; x++) {
                int src_left = PADDING + ((x - PADDING + COLS) % COLS);
                int src_right = PADDING + (x % COLS);

                input[row + x] = input[row + src_left];
                input[row + (PADDING + COLS + x)] = input[row + src_right];
            }
        }

#ifdef FEAT_IMPL_OMP
#pragma omp for collapse(2)
#endif
        for (unsigned int y = PADDING; y < PROWS - PADDING; y++) {
            for (unsigned int x = PADDING; x < PCOLS - PADDING; x++) {
                kernel_inner(w, input, output, x, y, dt);
            }
        }
#else
#error No toroid wrapping method defined
#endif
    }

    state->world = output;
    state->tmp = input;
}

void lenia_impl_dump(struct lenia_impl_state *state, uint8_t *out_frame)
{
    for (unsigned int y = 0; y < ROWS; y++) {
        for (unsigned int x = 0; x < COLS; x++) {
            out_frame[y * COLS + x] = state->world[(PADDING + y) * PCOLS + (PADDING + x)] * 255;
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

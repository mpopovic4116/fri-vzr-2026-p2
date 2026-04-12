#include <errno.h>
#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "impl.h"

#ifdef FEAT_GIF
#include "gifenc.h"
#include "orbium.h"
#endif

static bool startswith(const char *str, const char *prefix)
{
    size_t i = 0;
    for (;;) {
        if (prefix[i] == 0)
            return true;
        if (str[i] != prefix[i])
            return false;
        i++;
    }
}

static const char *mayberemoveprefix(const char *str, const char *prefix)
{
    if (startswith(str, prefix)) {
        return str + strlen(prefix);
    } else {
        return NULL;
    }
}

bool parse_int(const char *str, int *out)
{
    char *end;
    long value;

    if (str == NULL || out == NULL)
        return false;

    errno = 0;
    value = strtol(str, &end, 10);

    if (end == str || *end != '\0')
        return false;
    if (errno == ERANGE || value < INT_MIN || value > INT_MAX)
        return false;

    *out = (int) value;
    return true;
}

bool parse_double(const char *str, double *out)
{
    char *end;
    double value;

    if (str == NULL || out == NULL)
        return false;

    errno = 0;
    value = strtod(str, &end);

    if (end == str || *end != '\0')
        return false;
    if (errno == ERANGE || isnan(value))
        return false;

    *out = value;
    return true;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    int steps = 100;
    double dt = 0.1;
#ifdef FEAT_GIF
    const char *gif_fname = NULL;
#endif

    for (size_t i = 1; i < argc; i++) {
        char *arg = argv[i];
        const char *v = NULL;
        if ((v = mayberemoveprefix(arg, "steps="))) {
            if (!parse_int(v, &steps) || steps < 1) {
                fprintf(stderr, "Failed to parse argument: steps\n");
                return 1;
            }
        } else if ((v = mayberemoveprefix(arg, "dt="))) {
            if (!parse_double(v, &dt) || dt <= 0) {
                fprintf(stderr, "Failed to parse argument: dt\n");
                return 1;
            }
        }
#ifdef FEAT_GIF
        else if ((v = mayberemoveprefix(arg, "gif="))) {
            if (strlen(v) == 0) {
                fprintf(stderr, "Failed to parse argument: gif\n");
                return 1;
            }
            gif_fname = v;
        }
#endif
    }

#ifdef FEAT_GIF
    ge_GIF *gif = NULL;
    if (gif_fname != NULL) {
        gif = ge_new_gif(
            gif_fname,                // File name
            FEAT_SIZE_W, FEAT_SIZE_H, // Canvas size
            inferno_pallete,          // Palette
            8,                        // Palette depth == log2(# of colors)
            -1,                       // No transparency
            0                         // Infinite loop
        );
    }
#endif

    double t_start = omp_get_wtime();
    struct lenia_impl_state *state = lenia_impl_init();
    double t_init = omp_get_wtime();
    lenia_impl_upload(state);
    double t_upload = omp_get_wtime();
    for (int step = 0; step < steps; step++) {
        lenia_impl_step(state, dt);
#ifdef FEAT_GIF
        if (gif != NULL) {
            lenia_impl_dump(state, gif->frame);
            ge_add_frame(gif, 5);
        }
#endif
    }
    double t_steps = omp_get_wtime();
    lenia_impl_download(state);
    double t_download = omp_get_wtime();
    lenia_impl_free(state);
#ifdef FEAT_GIF
    if (gif != NULL) ge_close_gif(gif);
#endif
    double t_free = omp_get_wtime();

    printf("%s steps=%d dt=%0.3lf "
           "t_init=%0.15lf t_upload=%0.15lf t_steps=%0.15lf t_download=%0.15lf t_free=%0.15lf t_total=%0.15lf\n",
           PRINT_PREFIX,
           steps,
           dt,
           t_init - t_start,
           t_upload - t_init,
           t_steps - t_upload,
           t_download - t_steps,
           t_free - t_download,
           t_free - t_start);

    return 0;
}

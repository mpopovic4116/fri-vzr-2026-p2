#ifndef IMPL_H
#define IMPL_H

#include <stdint.h>

#include "float_alias.h"

#ifdef __cplusplus
extern "C"
{
#endif

    struct lenia_impl_state;
    struct lenia_impl_state *lenia_impl_init();
    void lenia_impl_upload(struct lenia_impl_state *state);
    void lenia_impl_step(struct lenia_impl_state *state, fhost dt);
    void lenia_impl_dump(struct lenia_impl_state *state, uint8_t *out_frame);
    void lenia_impl_download(struct lenia_impl_state *state);
    void lenia_impl_free(struct lenia_impl_state *state);

#ifdef __cplusplus
}
#endif

#endif

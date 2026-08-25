#define qbl_step_wide_v0 qbl_reference_step_wide_v0
#include "../../tools/qbl_step_wide_v0_kernel.c"

uint32_t qbl_wide_used_pair_limbs(const qbl_wide_custody_v0 *state) {
    uint32_t used = QBL_WIDE_MAX_PAIR_LIMBS;
    if (state == NULL) return 0u;
    while (used != 0u && state->u[used - 1u] == 0u && state->v[used - 1u] == 0u) --used;
    return used;
}

uint64_t qbl_wide_fingerprint64(const qbl_wide_custody_v0 *state) {
    const unsigned char *bytes = (const unsigned char *)(const void *)state;
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    size_t index;
    if (state == NULL) return 0u;
    for (index = 0u; index < sizeof(*state); ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

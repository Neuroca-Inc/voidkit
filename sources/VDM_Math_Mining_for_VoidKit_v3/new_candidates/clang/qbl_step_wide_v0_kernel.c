#include "qbl_wide_v0.h"

#define U64_MAX_VALUE UINT64_C(0xffffffffffffffff)

__extension__ typedef unsigned __int128 qbl_u128;

static uint32_t used_limbs(const uint64_t *x, uint32_t count) {
    while (count != 0u && x[count - 1u] == 0u) --count;
    return count;
}

static int compare_limbs(const uint64_t *a, const uint64_t *b, uint32_t count) {
    while (count != 0u) {
        uint32_t index = count - 1u;
        if (a[index] < b[index]) return -1;
        if (a[index] > b[index]) return 1;
        --count;
    }
    return 0;
}

static uint32_t add_pair(const uint64_t *a, const uint64_t *b, uint64_t *out) {
    qbl_u128 carry = 0;
    uint32_t index;
    for (index = 0u; index < QBL_WIDE_MAX_PAIR_LIMBS; ++index) {
        qbl_u128 sum = (qbl_u128)a[index] + b[index] + carry;
        out[index] = (uint64_t)sum;
        carry = sum >> 64;
    }
    return (uint32_t)carry;
}

static void mul_pair(const uint64_t *a, const uint64_t *b, uint64_t *out) {
    uint32_t i;
    uint32_t j;
    for (i = 0u; i < QBL_WIDE_MAX_PRODUCT_LIMBS; ++i) out[i] = 0u;
    for (i = 0u; i < QBL_WIDE_MAX_PAIR_LIMBS; ++i) {
        qbl_u128 carry = 0;
        for (j = 0u; j < QBL_WIDE_MAX_PAIR_LIMBS; ++j) {
            uint32_t k = i + j;
            qbl_u128 accum = (qbl_u128)a[i] * b[j];
            accum += out[k];
            accum += carry;
            out[k] = (uint64_t)accum;
            carry = accum >> 64;
        }
        if (i + QBL_WIDE_MAX_PAIR_LIMBS < QBL_WIDE_MAX_PRODUCT_LIMBS) {
            uint32_t k = i + QBL_WIDE_MAX_PAIR_LIMBS;
            while (carry != 0u && k < QBL_WIDE_MAX_PRODUCT_LIMBS) {
                qbl_u128 accum = (qbl_u128)out[k] + carry;
                out[k] = (uint64_t)accum;
                carry = accum >> 64;
                ++k;
            }
        }
    }
}

static int compare_product_to_pow2(const uint64_t *product, uint64_t exponent) {
    uint32_t limb;
    uint32_t bit;
    uint32_t index;
    uint64_t target;
    if (exponent >= 512u) return -1;
    limb = (uint32_t)(exponent >> 6);
    bit = (uint32_t)(exponent & 63u);
    for (index = QBL_WIDE_MAX_PRODUCT_LIMBS; index > limb + 1u; --index) {
        if (product[index - 1u] != 0u) return 1;
    }
    target = UINT64_C(1) << bit;
    if (product[limb] < target) return -1;
    if (product[limb] > target) return 1;
    for (index = 0u; index < limb; ++index) {
        if (product[index] != 0u) return 1;
    }
    return 0;
}

static uint64_t phase_positions(uint64_t domain) {
    return UINT64_C(6) << domain;
}

static uint64_t global_position(uint64_t phase_count, uint64_t local_position) {
    return phase_count - UINT64_C(5) + local_position;
}

static uint64_t capacity_exponent(uint64_t global) {
    if (global == 1u) return 1u;
    if (global == 2u) return 2u;
    return global << 1;
}

static void copy_u64s(uint64_t *dst, const uint64_t *src, uint32_t count) {
    uint32_t index;
    for (index = 0u; index < count; ++index) dst[index] = src[index];
}

static void zero_transition(qbl_wide_transition_v0 *transition) {
    uint64_t *words = (uint64_t *)(void *)transition;
    uint32_t count = (uint32_t)(sizeof(*transition) / sizeof(uint64_t));
    uint32_t index;
    for (index = 0u; index < count; ++index) words[index] = 0u;
}

static void zero_demand(qbl_wide_limb_demand_v0 *demand) {
    demand->required_pair_limbs = 0u;
    demand->provisioned_pair_limbs = 0u;
    demand->primitive = 0u;
    demand->reason = 0u;
}

uint32_t qbl_step_wide_v0(qbl_wide_custody_v0 *state,
                          uint32_t provisioned_pair_limbs,
                          qbl_wide_transition_v0 *transition,
                          qbl_wide_limb_demand_v0 *demand) {
    qbl_wide_custody_v0 next;
    qbl_wide_transition_v0 receipt;
    uint64_t candidate[QBL_WIDE_MAX_PAIR_LIMBS];
    uint64_t current_product[QBL_WIDE_MAX_PRODUCT_LIMBS];
    uint64_t candidate_product[QBL_WIDE_MAX_PRODUCT_LIMBS];
    uint32_t current_used;
    uint32_t candidate_used = 0u;
    uint32_t add_overflow;
    uint64_t before_n;
    uint64_t before_j;
    uint64_t before_e;
    uint64_t after_n;
    uint64_t after_j;
    uint64_t after_e;
    int final_position;
    int current_cmp;
    int candidate_cmp = 1;
    uint32_t primitive;
    uint32_t reason;

    if (state == 0 || transition == 0 || demand == 0) return QBL_WIDE_STATUS_NULL;
    if (provisioned_pair_limbs == 0u || provisioned_pair_limbs > QBL_WIDE_MAX_PAIR_LIMBS)
        return QBL_WIDE_STATUS_INVALID_STATE;

    current_used = used_limbs(state->v, QBL_WIDE_MAX_PAIR_LIMBS);
    if (used_limbs(state->u, QBL_WIDE_MAX_PAIR_LIMBS) > current_used)
        current_used = used_limbs(state->u, QBL_WIDE_MAX_PAIR_LIMBS);
    if (current_used == 0u || current_used > provisioned_pair_limbs)
        return QBL_WIDE_STATUS_INVALID_STATE;
    if (compare_limbs(state->u, state->v, QBL_WIDE_MAX_PAIR_LIMBS) > 0)
        return QBL_WIDE_STATUS_INVALID_PAIR;
    if (state->domain > 59u) return QBL_WIDE_STATUS_DOMAIN_RANGE;

    before_n = phase_positions(state->domain);
    if (state->local_position >= before_n) return QBL_WIDE_STATUS_INVALID_STATE;
    before_j = global_position(before_n, state->local_position);
    before_e = capacity_exponent(before_j);
    final_position = state->local_position == before_n - 1u;

    mul_pair(state->u, state->v, current_product);
    current_cmp = compare_product_to_pow2(current_product, before_e);

    add_overflow = add_pair(state->u, state->v, candidate);
    if (add_overflow == 0u) {
        candidate_used = used_limbs(candidate, QBL_WIDE_MAX_PAIR_LIMBS);
        mul_pair(state->v, candidate, candidate_product);
        candidate_cmp = compare_product_to_pow2(candidate_product, before_e);
    }

    if (final_position) {
        if (current_cmp < 0) {
            primitive = QBL_WIDE_PRIMITIVE_B;
            reason = QBL_WIDE_REASON_B_FINAL_CROSSING;
        } else {
            primitive = QBL_WIDE_PRIMITIVE_L;
            reason = QBL_WIDE_REASON_L_DOMAIN_SATURATED;
        }
    } else if (add_overflow == 0u && candidate_cmp <= 0) {
        primitive = QBL_WIDE_PRIMITIVE_B;
        reason = QBL_WIDE_REASON_B_NEXT_WITHIN_CAPACITY;
    } else if (add_overflow != 0u && before_e > 256u) {
        return QBL_WIDE_STATUS_WIDE_EXHAUSTED;
    } else {
        primitive = QBL_WIDE_PRIMITIVE_Q;
        reason = QBL_WIDE_REASON_Q_B_BLOCKED_POSITION_AVAILABLE;
    }

    if (primitive == QBL_WIDE_PRIMITIVE_B) {
        if (add_overflow != 0u) return QBL_WIDE_STATUS_WIDE_EXHAUSTED;
        if (candidate_used > provisioned_pair_limbs) {
            demand->required_pair_limbs = candidate_used;
            demand->provisioned_pair_limbs = provisioned_pair_limbs;
            demand->primitive = primitive;
            demand->reason = reason;
            return QBL_WIDE_STATUS_NEEDS_MORE_LIMBS;
        }
    }

    next = *state;
    zero_transition(&receipt);
    receipt.primitive = primitive;
    receipt.reason = reason;
    receipt.state_changed = 1u;
    receipt.before_pair_limbs = current_used;
    receipt.before_product_limbs = used_limbs(current_product, QBL_WIDE_MAX_PRODUCT_LIMBS);
    receipt.before_phase_positions = before_n;
    receipt.before_global_position = before_j;
    receipt.before_capacity_exponent = before_e;
    copy_u64s(receipt.before_product, current_product, QBL_WIDE_MAX_PRODUCT_LIMBS);

    if (primitive == QBL_WIDE_PRIMITIVE_B) {
        copy_u64s(next.u, state->v, QBL_WIDE_MAX_PAIR_LIMBS);
        copy_u64s(next.v, candidate, QBL_WIDE_MAX_PAIR_LIMBS);
    } else if (primitive == QBL_WIDE_PRIMITIVE_Q) {
        if (next.quarter_turns == U64_MAX_VALUE) return QBL_WIDE_STATUS_OVERFLOW;
        next.local_position += 1u;
        next.quarter_turns += 1u;
    } else {
        if (next.domain == 59u) return QBL_WIDE_STATUS_DOMAIN_RANGE;
        next.domain += 1u;
        next.local_position = 0u;
    }

    after_n = phase_positions(next.domain);
    after_j = global_position(after_n, next.local_position);
    after_e = capacity_exponent(after_j);
    mul_pair(next.u, next.v, receipt.after_product);
    receipt.after_pair_limbs = used_limbs(next.v, QBL_WIDE_MAX_PAIR_LIMBS);
    if (used_limbs(next.u, QBL_WIDE_MAX_PAIR_LIMBS) > receipt.after_pair_limbs)
        receipt.after_pair_limbs = used_limbs(next.u, QBL_WIDE_MAX_PAIR_LIMBS);
    receipt.after_product_limbs = used_limbs(receipt.after_product, QBL_WIDE_MAX_PRODUCT_LIMBS);
    receipt.after_phase_positions = after_n;
    receipt.after_global_position = after_j;
    receipt.after_capacity_exponent = after_e;

    *state = next;
    *transition = receipt;
    zero_demand(demand);
    return QBL_WIDE_STATUS_OK;
}

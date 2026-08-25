#include "qbl_abi.h"

#include <stdint.h>
#include <string.h>

uint32_t qbl_reference_b_u64(qbl_pair_u64 *pair) {
    if (pair == NULL) {
        return QBL_STATUS_NULL;
    }
    if (pair->u == 0 || pair->u > pair->v) {
        return QBL_STATUS_INVALID_PAIR;
    }
    if (UINT64_MAX - pair->u < pair->v) {
        return QBL_STATUS_OVERFLOW;
    }

    const uint64_t next_u = pair->v;
    const uint64_t next_v = pair->u + pair->v;
    pair->u = next_u;
    pair->v = next_v;
    return QBL_STATUS_OK;
}

static int product_le_capacity(__uint128_t product, uint64_t exponent) {
    if (exponent >= 128) {
        return 1;
    }
    return product <= (((__uint128_t)1) << exponent);
}

static int product_lt_capacity(__uint128_t product, uint64_t exponent) {
    if (exponent >= 128) {
        return 1;
    }
    return product < (((__uint128_t)1) << exponent);
}

uint32_t qbl_reference_inspect_u64(const qbl_inspect_state_u64 *state,
                                   qbl_inspection_u64 *inspection) {
    if (state == NULL || inspection == NULL) {
        return QBL_STATUS_NULL;
    }
    if (state->u == 0 || state->u > state->v) {
        return QBL_STATUS_INVALID_PAIR;
    }
    if (state->domain > QBL_MAX_U64_DOMAIN) {
        return QBL_STATUS_DOMAIN_RANGE;
    }

    const uint64_t phase_positions = UINT64_C(6) << state->domain;
    if (state->local_position >= phase_positions) {
        return QBL_STATUS_INVALID_STATE;
    }

    const uint64_t global_position =
        (phase_positions - UINT64_C(5)) + state->local_position;
    const uint64_t capacity_exponent =
        global_position == 1 ? 1
        : global_position == 2 ? 2
                               : 2 * global_position;
    const int terminal_position =
        state->local_position == phase_positions - UINT64_C(1);
    const __uint128_t current_product =
        (__uint128_t)state->u * (__uint128_t)state->v;

    qbl_inspection_u64 candidate;
    memset(&candidate, 0, sizeof(candidate));
    candidate.terminal_position = (uint32_t)terminal_position;
    candidate.phase_positions = phase_positions;
    candidate.global_position = global_position;
    candidate.capacity_exponent = capacity_exponent;
    candidate.current_product_lo = (uint64_t)current_product;
    candidate.current_product_hi = (uint64_t)(current_product >> 64);

    if (terminal_position) {
        if (!product_lt_capacity(current_product, capacity_exponent)) {
            candidate.primitive = QBL_PRIMITIVE_L;
            candidate.reason = QBL_REASON_L_DOMAIN_SATURATED;
            *inspection = candidate;
            return QBL_STATUS_OK;
        }

        if (UINT64_MAX - state->u < state->v) {
            return QBL_STATUS_NEEDS_WIDE_PAIR;
        }
        candidate.next_u = state->v;
        candidate.next_v = state->u + state->v;
        const __uint128_t candidate_product =
            (__uint128_t)candidate.next_u * (__uint128_t)candidate.next_v;
        candidate.candidate_product_lo = (uint64_t)candidate_product;
        candidate.candidate_product_hi = (uint64_t)(candidate_product >> 64);
        candidate.can_b = 1;
        candidate.primitive = QBL_PRIMITIVE_B;
        candidate.reason = QBL_REASON_B_FINAL_CROSSING;
        *inspection = candidate;
        return QBL_STATUS_OK;
    }

    candidate.can_q = 1;
    if (!product_lt_capacity(current_product, capacity_exponent)) {
        candidate.primitive = QBL_PRIMITIVE_Q;
        candidate.reason = QBL_REASON_Q_B_BLOCKED_POSITION_AVAILABLE;
        *inspection = candidate;
        return QBL_STATUS_OK;
    }
    if (UINT64_MAX - state->u < state->v) {
        return QBL_STATUS_NEEDS_WIDE_PAIR;
    }
    candidate.next_u = state->v;
    candidate.next_v = state->u + state->v;
    const __uint128_t candidate_product =
        (__uint128_t)candidate.next_u * (__uint128_t)candidate.next_v;
    candidate.candidate_product_lo = (uint64_t)candidate_product;
    candidate.candidate_product_hi = (uint64_t)(candidate_product >> 64);

    if (product_le_capacity(candidate_product, capacity_exponent)) {
        candidate.can_b = 1;
        candidate.primitive = QBL_PRIMITIVE_B;
        candidate.reason = QBL_REASON_B_NEXT_WITHIN_CAPACITY;
    } else {
        candidate.primitive = QBL_PRIMITIVE_Q;
        candidate.reason = QBL_REASON_Q_B_BLOCKED_POSITION_AVAILABLE;
    }

    *inspection = candidate;
    return QBL_STATUS_OK;
}

static uint64_t capacity_exponent_for_position(uint64_t global_position) {
    if (global_position == 1) {
        return 1;
    }
    if (global_position == 2) {
        return 2;
    }
    return 2 * global_position;
}

uint32_t qbl_reference_step_u64(qbl_custody_state_u64 *state,
                                qbl_transition_u64 *transition) {
    if (state == NULL || transition == NULL) {
        return QBL_STATUS_NULL;
    }
    if ((void *)state == (void *)transition) {
        return QBL_STATUS_INVALID_STATE;
    }
    if (state->u == 0 || state->u > state->v) {
        return QBL_STATUS_INVALID_PAIR;
    }
    if (state->domain > QBL_MAX_U64_DOMAIN) {
        return QBL_STATUS_DOMAIN_RANGE;
    }

    const uint64_t before_phase_positions = UINT64_C(6) << state->domain;
    if (state->local_position >= before_phase_positions) {
        return QBL_STATUS_INVALID_STATE;
    }

    const uint64_t before_global_position =
        before_phase_positions - UINT64_C(5) + state->local_position;
    const uint64_t before_capacity_exponent =
        capacity_exponent_for_position(before_global_position);
    const __uint128_t before_product =
        (__uint128_t)state->u * (__uint128_t)state->v;
    const int terminal_position =
        state->local_position == before_phase_positions - UINT64_C(1);

    qbl_custody_state_u64 next = *state;
    qbl_transition_u64 receipt;
    memset(&receipt, 0, sizeof(receipt));
    receipt.before_phase_positions = before_phase_positions;
    receipt.before_global_position = before_global_position;
    receipt.before_capacity_exponent = before_capacity_exponent;
    receipt.before_product_lo = (uint64_t)before_product;
    receipt.before_product_hi = (uint64_t)(before_product >> 64);

    if (terminal_position) {
        if (product_lt_capacity(before_product, before_capacity_exponent)) {
            if (UINT64_MAX - state->u < state->v) {
                return QBL_STATUS_NEEDS_WIDE_PAIR;
            }
            next.u = state->v;
            next.v = state->u + state->v;
            receipt.primitive = QBL_PRIMITIVE_B;
            receipt.reason = QBL_REASON_B_FINAL_CROSSING;
        } else {
            if (state->domain == QBL_MAX_U64_DOMAIN) {
                return QBL_STATUS_DOMAIN_RANGE;
            }
            next.domain = state->domain + UINT64_C(1);
            next.local_position = 0;
            receipt.primitive = QBL_PRIMITIVE_L;
            receipt.reason = QBL_REASON_L_DOMAIN_SATURATED;
        }
    } else {
        if (product_lt_capacity(before_product, before_capacity_exponent)) {
            if (UINT64_MAX - state->u < state->v) {
                return QBL_STATUS_NEEDS_WIDE_PAIR;
            }
            const uint64_t candidate_u = state->v;
            const uint64_t candidate_v = state->u + state->v;
            const __uint128_t candidate_product =
                (__uint128_t)candidate_u * (__uint128_t)candidate_v;
            if (product_le_capacity(candidate_product,
                                    before_capacity_exponent)) {
                next.u = candidate_u;
                next.v = candidate_v;
                receipt.primitive = QBL_PRIMITIVE_B;
                receipt.reason = QBL_REASON_B_NEXT_WITHIN_CAPACITY;
            } else {
                if (state->quarter_turns == UINT64_MAX) {
                    return QBL_STATUS_OVERFLOW;
                }
                next.local_position = state->local_position + UINT64_C(1);
                next.quarter_turns = state->quarter_turns + UINT64_C(1);
                receipt.primitive = QBL_PRIMITIVE_Q;
                receipt.reason =
                    QBL_REASON_Q_B_BLOCKED_POSITION_AVAILABLE;
            }
        } else {
            if (state->quarter_turns == UINT64_MAX) {
                return QBL_STATUS_OVERFLOW;
            }
            next.local_position = state->local_position + UINT64_C(1);
            next.quarter_turns = state->quarter_turns + UINT64_C(1);
            receipt.primitive = QBL_PRIMITIVE_Q;
            receipt.reason = QBL_REASON_Q_B_BLOCKED_POSITION_AVAILABLE;
        }
    }

    const uint64_t after_phase_positions = UINT64_C(6) << next.domain;
    const uint64_t after_global_position =
        after_phase_positions - UINT64_C(5) + next.local_position;
    const uint64_t after_capacity_exponent =
        capacity_exponent_for_position(after_global_position);
    const __uint128_t after_product =
        (__uint128_t)next.u * (__uint128_t)next.v;

    receipt.state_changed = 1;
    receipt.after_phase_positions = after_phase_positions;
    receipt.after_global_position = after_global_position;
    receipt.after_capacity_exponent = after_capacity_exponent;
    receipt.after_product_lo = (uint64_t)after_product;
    receipt.after_product_hi = (uint64_t)(after_product >> 64);

    *transition = receipt;
    *state = next;
    return QBL_STATUS_OK;
}

uint32_t qbl_word_get_u2(const uint8_t *bytes, uint64_t length,
                         uint64_t index, uint32_t *primitive) {
    if (bytes == NULL || primitive == NULL) {
        return QBL_STATUS_NULL;
    }
    if (index >= length) {
        return QBL_STATUS_INVALID_WORD;
    }
    const uint64_t byte_index = index >> 2;
    const uint32_t shift = (uint32_t)((index & UINT64_C(3)) << 1);
    const uint32_t code = (bytes[byte_index] >> shift) & UINT32_C(3);
    if (code < QBL_PRIMITIVE_B || code > QBL_PRIMITIVE_L) {
        return QBL_STATUS_INVALID_WORD;
    }
    *primitive = code;
    return QBL_STATUS_OK;
}

static int address_ranges_overlap(uintptr_t a, size_t a_size,
                                  uintptr_t b, size_t b_size) {
    if (a <= b) {
        return b - a < a_size;
    }
    return a - b < b_size;
}

uint32_t qbl_reference_step_record_u64(qbl_retained_state_u64 *state,
                                       qbl_transition_u64 *transition) {
    if (state == NULL || transition == NULL) {
        return QBL_STATUS_NULL;
    }
    if (address_ranges_overlap((uintptr_t)state, sizeof(*state),
                               (uintptr_t)transition, sizeof(*transition))) {
        return QBL_STATUS_INVALID_STATE;
    }
    if (state->word_bytes == NULL) {
        return QBL_STATUS_INVALID_WORD;
    }
    if (state->word_length > state->word_capacity) {
        return QBL_STATUS_INVALID_WORD;
    }
    if (state->word_length == state->word_capacity) {
        return QBL_STATUS_WORD_FULL;
    }

    const uint64_t byte_index = state->word_length >> 2;
    if ((uintptr_t)state->word_bytes > UINTPTR_MAX - byte_index) {
        return QBL_STATUS_INVALID_WORD;
    }
    uint8_t *const target =
        (uint8_t *)((uintptr_t)state->word_bytes + byte_index);
    if (address_ranges_overlap((uintptr_t)target, 1,
                               (uintptr_t)state, sizeof(*state)) ||
        address_ranges_overlap((uintptr_t)target, 1,
                               (uintptr_t)transition, sizeof(*transition))) {
        return QBL_STATUS_INVALID_WORD;
    }
    const uint32_t shift =
        (uint32_t)((state->word_length & UINT64_C(3)) << 1);
    const uint8_t mask = (uint8_t)(UINT8_C(3) << shift);
    const uint8_t old_byte = *target;
    if ((old_byte & mask) != 0) {
        return QBL_STATUS_INVALID_WORD;
    }

    qbl_custody_state_u64 next = {
        state->u,
        state->v,
        state->domain,
        state->local_position,
        state->quarter_turns,
    };
    qbl_transition_u64 receipt;
    memset(&receipt, 0, sizeof(receipt));
    const uint32_t status = qbl_reference_step_u64(&next, &receipt);
    if (status != QBL_STATUS_OK) {
        return status;
    }
    if (receipt.primitive < QBL_PRIMITIVE_B ||
        receipt.primitive > QBL_PRIMITIVE_L) {
        return QBL_STATUS_INVALID_STATE;
    }

    const uint8_t new_byte =
        (uint8_t)(old_byte | (uint8_t)(receipt.primitive << shift));

    *target = new_byte;
    state->u = next.u;
    state->v = next.v;
    state->domain = next.domain;
    state->local_position = next.local_position;
    state->quarter_turns = next.quarter_turns;
    state->word_length += UINT64_C(1);
    *transition = receipt;
    return QBL_STATUS_OK;
}

static int range_overlap_cp5(uintptr_t a, size_t a_size,
                             uintptr_t b, size_t b_size) {
    if (a <= b) {
        return b - a < a_size;
    }
    return a - b < b_size;
}

static int axis_zero(const qbl_orthad_axis_u128 *axis) {
    const unsigned char *bytes = (const unsigned char *)axis;
    for (size_t i = 0; i < sizeof(*axis); ++i) {
        if (bytes[i] != 0) {
            return 0;
        }
    }
    return 1;
}

static int axis_product_matches(const qbl_orthad_axis_u128 *axis,
                                uint64_t lo, uint64_t hi) {
    return axis->current_product_lo == lo &&
           axis->current_product_hi == hi;
}

uint32_t qbl_reference_step_orthad_local_u64(
    qbl_orthad_local_state_u64 *state, qbl_transition_u64 *transition) {
    if (state == NULL || transition == NULL) {
        return QBL_STATUS_NULL;
    }
    if (range_overlap_cp5((uintptr_t)state, sizeof(*state),
                          (uintptr_t)transition, sizeof(*transition))) {
        return QBL_STATUS_INVALID_STATE;
    }
    if (state->axes == NULL || state->axis_count == 0 ||
        state->axis_count > state->axis_capacity) {
        return QBL_STATUS_INVALID_ORTHAD;
    }
    if (state->domain == UINT64_MAX || state->axis_count != state->domain + 1) {
        return QBL_STATUS_INVALID_ORTHAD;
    }
    if (state->axis_count > SIZE_MAX / sizeof(qbl_orthad_axis_u128)) {
        return QBL_STATUS_INVALID_ORTHAD;
    }

    const uintptr_t axes_addr = (uintptr_t)state->axes;
    const size_t used_bytes =
        (size_t)state->axis_count * sizeof(qbl_orthad_axis_u128);
    if (axes_addr > UINTPTR_MAX - used_bytes) {
        return QBL_STATUS_INVALID_ORTHAD;
    }
    if (range_overlap_cp5(axes_addr, used_bytes,
                          (uintptr_t)state, sizeof(*state)) ||
        range_overlap_cp5(axes_addr, used_bytes,
                          (uintptr_t)transition, sizeof(*transition))) {
        return QBL_STATUS_INVALID_ORTHAD;
    }

    qbl_orthad_axis_u128 *const active =
        &state->axes[state->axis_count - 1];
    if (active->flags != QBL_ORTHAD_AXIS_ACTIVE ||
        active->phase_quadrant > 3 ||
        (active->origin_product_lo == 0 &&
         active->origin_product_hi == 0) ||
        (active->current_product_lo == 0 &&
         active->current_product_hi == 0) ||
        active->phase_quadrant != (uint32_t)(state->local_position & 3)) {
        return QBL_STATUS_INVALID_ORTHAD;
    }

    qbl_custody_state_u64 next = {
        state->u,
        state->v,
        state->domain,
        state->local_position,
        state->quarter_turns,
    };
    qbl_transition_u64 receipt;
    memset(&receipt, 0, sizeof(receipt));
    const uint32_t status = qbl_reference_step_u64(&next, &receipt);
    if (status != QBL_STATUS_OK) {
        return status;
    }
    if (!axis_product_matches(active, receipt.before_product_lo,
                              receipt.before_product_hi)) {
        return QBL_STATUS_INVALID_ORTHAD;
    }

    qbl_orthad_axis_u128 next_active = *active;
    qbl_orthad_axis_u128 new_axis;
    memset(&new_axis, 0, sizeof(new_axis));

    if (receipt.primitive == QBL_PRIMITIVE_B) {
        next_active.current_product_lo = receipt.after_product_lo;
        next_active.current_product_hi = receipt.after_product_hi;
    } else if (receipt.primitive == QBL_PRIMITIVE_Q) {
        next_active.phase_quadrant =
            (next_active.phase_quadrant + 1U) & 3U;
    } else if (receipt.primitive == QBL_PRIMITIVE_L) {
        if (state->axis_count == state->axis_capacity) {
            return QBL_STATUS_ORTHAD_FULL;
        }
        if (state->axis_count >= SIZE_MAX / sizeof(qbl_orthad_axis_u128)) {
            return QBL_STATUS_INVALID_ORTHAD;
        }
        qbl_orthad_axis_u128 *const target = &state->axes[state->axis_count];
        if (range_overlap_cp5((uintptr_t)target, sizeof(*target),
                              (uintptr_t)state, sizeof(*state)) ||
            range_overlap_cp5((uintptr_t)target, sizeof(*target),
                              (uintptr_t)transition, sizeof(*transition)) ||
            !axis_zero(target)) {
            return QBL_STATUS_INVALID_ORTHAD;
        }
        next_active.flags = QBL_ORTHAD_AXIS_LATCHED;
        new_axis.origin_product_lo = receipt.after_product_lo;
        new_axis.origin_product_hi = receipt.after_product_hi;
        new_axis.current_product_lo = receipt.after_product_lo;
        new_axis.current_product_hi = receipt.after_product_hi;
        new_axis.phase_quadrant = 0;
        new_axis.flags = QBL_ORTHAD_AXIS_ACTIVE;
    } else {
        return QBL_STATUS_INVALID_STATE;
    }

    *active = next_active;
    if (receipt.primitive == QBL_PRIMITIVE_L) {
        state->axes[state->axis_count] = new_axis;
        state->axis_count += 1;
    }
    state->u = next.u;
    state->v = next.v;
    state->domain = next.domain;
    state->local_position = next.local_position;
    state->quarter_turns = next.quarter_turns;
    *transition = receipt;
    return QBL_STATUS_OK;
}

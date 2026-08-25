/*
 * Complete lifted object query surface.
 *
 * The retained state already contains the entire history. Each axis carries
 * the pair product at which it opened and the pair product it currently
 * holds. B advances that product by the balanced rule and nothing is ever
 * overwritten, so every remaining component of the complete lifted object is
 * recovered by walking the accepted recurrence between two products that CP20
 * already stores.
 *
 * Nothing here is retained. No index, no stored count, no cached layer table,
 * no word, no hash. These are reads.
 */

#include "qbl_abi.h"

#include <stddef.h>

/* Walk the balanced rule until the carried product equals the target. */
static uint32_t advance_to_product(uint64_t *u, uint64_t *v, uint64_t target_lo,
                                   uint64_t target_hi, uint64_t *steps_out) {
    uint64_t steps = 0;
    for (;;) {
        const unsigned __int128 product =
            (unsigned __int128)(*u) * (unsigned __int128)(*v);
        const uint64_t lo = (uint64_t)product;
        const uint64_t hi = (uint64_t)(product >> 64);
        if (lo == target_lo && hi == target_hi) {
            if (steps_out != NULL) {
                *steps_out = steps;
            }
            return QBL_STATUS_OK;
        }
        if (hi > target_hi || (hi == target_hi && lo > target_lo)) {
            return QBL_STATUS_INVALID_ORTHAD;
        }
        const uint64_t next = *u + *v;
        if (next < *u) {
            return QBL_STATUS_OVERFLOW;
        }
        *u = *v;
        *v = next;
        steps += 1;
        if (steps > (uint64_t)QBL_MAX_U64_DOMAIN * 64u) {
            return QBL_STATUS_INVALID_ORTHAD;
        }
    }
}

uint32_t qbl_orthad_layer_refinements(const qbl_orthad_local_state_u64 *state,
                                      uint64_t layer, uint64_t *out) {
    if (state == NULL || out == NULL || state->axes == NULL) {
        return QBL_STATUS_NULL;
    }
    if (layer >= state->axis_count) {
        return QBL_STATUS_INVALID_ORTHAD;
    }
    const qbl_orthad_axis_u128 *const axis = &state->axes[layer];
    uint64_t u = 1;
    uint64_t v = 1;
    const uint32_t status = advance_to_product(
        &u, &v, axis->origin_product_lo, axis->origin_product_hi, NULL);
    if (status != QBL_STATUS_OK) {
        return status;
    }
    return advance_to_product(&u, &v, axis->current_product_lo,
                              axis->current_product_hi, out);
}

/* Retained origin, plus the identity determination, plus one insertion per B. */
uint32_t qbl_orthad_layer_point_count(const qbl_orthad_local_state_u64 *state,
                                      uint64_t layer, uint64_t *out) {
    uint64_t refinements = 0;
    const uint32_t status =
        qbl_orthad_layer_refinements(state, layer, &refinements);
    if (status != QBL_STATUS_OK) {
        return status;
    }
    if (refinements > UINT64_MAX - 2u) {
        return QBL_STATUS_OVERFLOW;
    }
    *out = refinements + 2u;
    return QBL_STATUS_OK;
}

/* Omega- reads the retained order reversed. The reference does not flip. */
uint32_t qbl_orthad_minus_chart_index(const qbl_orthad_local_state_u64 *state,
                                      uint64_t layer, uint64_t plus_index,
                                      uint64_t *out) {
    uint64_t points = 0;
    const uint32_t status = qbl_orthad_layer_point_count(state, layer, &points);
    if (status != QBL_STATUS_OK) {
        return status;
    }
    if (plus_index >= points) {
        return QBL_STATUS_INVALID_ORTHAD;
    }
    *out = points - 1u - plus_index;
    return QBL_STATUS_OK;
}

/* Both directed transfers are the same reversal read in the opposite chart. */
uint32_t qbl_orthad_transfer_index(const qbl_orthad_local_state_u64 *state,
                                   uint64_t layer, uint64_t from_index,
                                   uint64_t *out) {
    return qbl_orthad_minus_chart_index(state, layer, from_index, out);
}

/*
 * Primary relation membership. Within one layer the relation is the strict
 * retained order. Across layers every directed placement is retained, so
 * membership holds in both directions.
 */
uint32_t qbl_orthad_primary_contains(const qbl_orthad_local_state_u64 *state,
                                     uint64_t left_layer, uint64_t left_point,
                                     uint64_t right_layer, uint64_t right_point,
                                     uint32_t *out) {
    uint64_t left_points = 0;
    uint64_t right_points = 0;
    uint32_t status =
        qbl_orthad_layer_point_count(state, left_layer, &left_points);
    if (status != QBL_STATUS_OK) {
        return status;
    }
    status = qbl_orthad_layer_point_count(state, right_layer, &right_points);
    if (status != QBL_STATUS_OK) {
        return status;
    }
    if (left_point >= left_points || right_point >= right_points) {
        return QBL_STATUS_INVALID_ORTHAD;
    }
    *out = (left_layer == right_layer) ? (uint32_t)(left_point < right_point) : 1u;
    return QBL_STATUS_OK;
}

/* Exact cardinalities over every retained layer. Counted, never enumerated. */
uint32_t qbl_orthad_relation_totals(const qbl_orthad_local_state_u64 *state,
                                    uint64_t *total_points_out,
                                    unsigned __int128 *within_out,
                                    unsigned __int128 *cross_out) {
    if (state == NULL || total_points_out == NULL || within_out == NULL ||
        cross_out == NULL) {
        return QBL_STATUS_NULL;
    }
    uint64_t total = 0;
    unsigned __int128 sum_squares = 0;
    for (uint64_t layer = 0; layer < state->axis_count; ++layer) {
        uint64_t points = 0;
        const uint32_t status =
            qbl_orthad_layer_point_count(state, layer, &points);
        if (status != QBL_STATUS_OK) {
            return status;
        }
        if (total > UINT64_MAX - points) {
            return QBL_STATUS_OVERFLOW;
        }
        total += points;
        sum_squares += (unsigned __int128)points * (unsigned __int128)points;
    }
    *total_points_out = total;
    *within_out = (sum_squares - total) / 2u;
    *cross_out = (unsigned __int128)total * total - sum_squares;
    return QBL_STATUS_OK;
}

#include "phase_wide_v0.h"

#include <limits.h>
#include <string.h>

static void hash_feed(uint64_t *hash, const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t *)data;
    size_t i;
    for (i = 0u; i < size; ++i) {
        *hash ^= bytes[i];
        *hash *= UINT64_C(0x100000001b3);
    }
}

static int product_equal(const uint64_t *a, const uint64_t *b) {
    return memcmp(a, b, QBL_WIDE_MAX_PRODUCT_LIMBS * sizeof(uint64_t)) == 0;
}

static int product_zero(const uint64_t *a) {
    uint32_t i;
    for (i = 0u; i < QBL_WIDE_MAX_PRODUCT_LIMBS; ++i) if (a[i] != 0u) return 0;
    return 1;
}

static pww_result_v0 result_base(const pww_cell_v0 *cell) {
    pww_result_v0 result;
    memset(&result, 0, sizeof(result));
    if (cell != NULL) {
        result.local_commit_id = cell->world.accepted_transition_id;
        result.phase_fingerprint = pww_phase_fingerprint(&cell->phase);
        result.world_fingerprint = wc_world_fingerprint(&cell->world);
    }
    return result;
}

uint64_t pww_phase_fingerprint(const pww_phase_v0 *phase) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    uint32_t count;
    if (phase == NULL) return 0u;
    hash_feed(&hash, &phase->custody, sizeof(phase->custody));
    hash_feed(&hash, &phase->axis_count, sizeof(phase->axis_count));
    count = phase->axis_count;
    if (count > PWW_AXIS_STORAGE) count = PWW_AXIS_STORAGE;
    hash_feed(&hash, phase->axes, count * sizeof(phase->axes[0]));
    return hash;
}

uint32_t pww_cell_init(pww_cell_v0 *cell, uint64_t zone_key, uint32_t pair_limb_limit) {
    if (cell == NULL || pair_limb_limit == 0u || pair_limb_limit > QBL_WIDE_MAX_PAIR_LIMBS)
        return PWW_STATUS_INVALID_ARGUMENT;
    memset(cell, 0, sizeof(*cell));
    cell->phase.custody.u[0] = 1u;
    cell->phase.custody.v[0] = 1u;
    cell->phase.pair_limb_limit = pair_limb_limit;
    cell->phase.axis_count = 1u;
    cell->phase.axes[0].origin_product[0] = 1u;
    cell->phase.axes[0].current_product[0] = 1u;
    cell->phase.axes[0].flags = PWW_AXIS_ACTIVE;
    wc_world_init(&cell->world, zone_key);
    return PWW_STATUS_OK;
}

uint32_t pww_cell_provision_pair_limbs(pww_cell_v0 *cell, uint32_t new_limit) {
    if (cell == NULL || new_limit < cell->phase.pair_limb_limit ||
        new_limit == 0u || new_limit > QBL_WIDE_MAX_PAIR_LIMBS)
        return PWW_STATUS_INVALID_ARGUMENT;
    cell->phase.pair_limb_limit = new_limit;
    return PWW_STATUS_OK;
}

pww_result_v0 pww_cell_transact(pww_cell_v0 *cell,
                                const wc_cause_v0 *causes, size_t cause_count,
                                const wc_intent_v0 *intents, size_t intent_count) {
    pww_result_v0 result;
    pww_cell_v0 staged;
    qbl_wide_transition_v0 transition;
    qbl_wide_limb_demand_v0 demand;
    pww_axis_v0 *active;
    wc_transition_result_v0 world_result;
    uint32_t status;

    if (cell == NULL || (cause_count != 0u && causes == NULL) ||
        (intent_count != 0u && intents == NULL)) {
        result = result_base(cell);
        result.status = PWW_STATUS_INVALID_ARGUMENT;
        return result;
    }
    if (cause_count == 0u && intent_count == 0u) {
        result = result_base(cell);
        result.status = PWW_STATUS_NO_WORK;
        return result;
    }
    if (cause_count == 0u) {
        result = result_base(cell);
        result.status = PWW_STATUS_WORLD_FAULT;
        result.world_status = WC_STATUS_FAULT;
        result.world_fault = WC_FAULT_CAUSE_MISSING;
        return result;
    }

    staged = *cell;
    memset(&transition, 0, sizeof(transition));
    memset(&demand, 0, sizeof(demand));
    status = qbl_step_wide_v0(&staged.phase.custody,
                              staged.phase.pair_limb_limit,
                              &transition, &demand);
    if (status == QBL_WIDE_STATUS_NEEDS_MORE_LIMBS) {
        result = result_base(cell);
        result.status = PWW_STATUS_PROVISION_REQUIRED;
        result.qbl_status = status;
        result.required_pair_limbs = demand.required_pair_limbs;
        return result;
    }
    if (status != QBL_WIDE_STATUS_OK) {
        result = result_base(cell);
        result.status = PWW_STATUS_PHASE_FAULT;
        result.qbl_status = status;
        return result;
    }
    if (staged.phase.axis_count == 0u || staged.phase.axis_count > PWW_AXIS_STORAGE) {
        result = result_base(cell);
        result.status = PWW_STATUS_PHASE_FAULT;
        result.qbl_status = QBL_WIDE_STATUS_INVALID_STATE;
        return result;
    }
    active = &staged.phase.axes[staged.phase.axis_count - 1u];
    if (active->flags != PWW_AXIS_ACTIVE ||
        !product_equal(active->current_product, transition.before_product)) {
        result = result_base(cell);
        result.status = PWW_STATUS_PHASE_FAULT;
        result.qbl_status = QBL_WIDE_STATUS_INVALID_STATE;
        return result;
    }
    if (transition.primitive == QBL_WIDE_PRIMITIVE_B) {
        memcpy(active->current_product, transition.after_product,
               sizeof(active->current_product));
    } else if (transition.primitive == QBL_WIDE_PRIMITIVE_Q) {
        active->phase_quadrant = (active->phase_quadrant + 1u) & 3u;
    } else if (transition.primitive == QBL_WIDE_PRIMITIVE_L) {
        pww_axis_v0 *new_axis;
        if (staged.phase.axis_count == PWW_AXIS_STORAGE) {
            result = result_base(cell);
            result.status = PWW_STATUS_ORTHAD_FULL;
            return result;
        }
        active->flags = PWW_AXIS_LATCHED;
        new_axis = &staged.phase.axes[staged.phase.axis_count];
        if (!product_zero(new_axis->origin_product) ||
            !product_zero(new_axis->current_product) ||
            new_axis->flags != 0u || new_axis->phase_quadrant != 0u) {
            result = result_base(cell);
            result.status = PWW_STATUS_PHASE_FAULT;
            result.qbl_status = QBL_WIDE_STATUS_INVALID_STATE;
            return result;
        }
        memcpy(new_axis->origin_product, transition.after_product,
               sizeof(new_axis->origin_product));
        memcpy(new_axis->current_product, transition.after_product,
               sizeof(new_axis->current_product));
        new_axis->flags = PWW_AXIS_ACTIVE;
        staged.phase.axis_count += 1u;
    } else {
        result = result_base(cell);
        result.status = PWW_STATUS_PHASE_FAULT;
        result.qbl_status = QBL_WIDE_STATUS_INVALID_STATE;
        return result;
    }

    world_result = wc_world_transact(&staged.world, causes, cause_count, intents, intent_count);
    if (world_result.status != WC_STATUS_OK) {
        result = result_base(cell);
        result.status = PWW_STATUS_WORLD_FAULT;
        result.world_status = world_result.status;
        result.world_fault = world_result.fault;
        return result;
    }

    *cell = staged;
    result = result_base(cell);
    result.status = PWW_STATUS_OK;
    result.qbl_status = QBL_WIDE_STATUS_OK;
    result.world_status = WC_STATUS_OK;
    result.primitive = transition.primitive;
    result.accepted_intents = world_result.accepted_intents;
    result.rejected_requests = world_result.rejected_requests;
    return result;
}

uint32_t pww_region_init(pww_region_v0 *region, const uint64_t *zone_keys,
                         size_t cell_count, uint32_t pair_limb_limit) {
    size_t i;
    size_t j;
    if (region == NULL || zone_keys == NULL || cell_count == 0u ||
        cell_count > PWW_REGION_MAX_CELLS) return PWW_FRONTIER_INVALID;
    for (i = 0u; i < cell_count; ++i) {
        for (j = i + 1u; j < cell_count; ++j) {
            if (zone_keys[i] == zone_keys[j]) return PWW_FRONTIER_DUPLICATE;
        }
    }
    memset(region, 0, sizeof(*region));
    region->cell_count = (uint32_t)cell_count;
    for (i = 0u; i < cell_count; ++i) {
        if (pww_cell_init(&region->cells[i], zone_keys[i], pair_limb_limit) != PWW_STATUS_OK)
            return PWW_FRONTIER_INVALID;
    }
    return PWW_FRONTIER_OK;
}

static size_t select_next(const pww_region_v0 *region,
                          const pww_frontier_work_v0 *work,
                          size_t count, const uint8_t *used) {
    size_t selected = SIZE_MAX;
    size_t i;
    for (i = 0u; i < count; ++i) {
        uint64_t zone;
        if (used[i]) continue;
        if (selected == SIZE_MAX) { selected = i; continue; }
        zone = region->cells[work[i].slot].world.zone_key;
        if (zone < region->cells[work[selected].slot].world.zone_key) selected = i;
    }
    return selected;
}

pww_frontier_receipt_v0 pww_region_transact(pww_region_v0 *region,
                                             const pww_frontier_work_v0 *work,
                                             size_t work_count) {
    pww_frontier_receipt_v0 receipt;
    pww_region_v0 staged;
    uint8_t seen[PWW_REGION_MAX_CELLS] = {0};
    uint8_t used[PWW_REGION_MAX_CELLS] = {0};
    size_t i;
    size_t rank;
    int any = 0;
    memset(&receipt, 0, sizeof(receipt));
    if (region == NULL || work_count > region->cell_count ||
        (work_count != 0u && work == NULL)) { receipt.status = PWW_FRONTIER_INVALID; return receipt; }
    for (i = 0u; i < work_count; ++i) {
        if (work[i].slot >= region->cell_count || seen[work[i].slot] ||
            (work[i].cause_count != 0u && work[i].causes == NULL) ||
            (work[i].intent_count != 0u && work[i].intents == NULL)) {
            receipt.status = seen[work[i].slot] ? PWW_FRONTIER_DUPLICATE : PWW_FRONTIER_INVALID;
            receipt.fault_slot = work[i].slot;
            return receipt;
        }
        seen[work[i].slot] = 1u;
        if (work[i].cause_count != 0u || work[i].intent_count != 0u) any = 1;
        if (work[i].expected_local_commit_id != region->cells[work[i].slot].world.accepted_transition_id) {
            receipt.status = PWW_FRONTIER_STALE;
            receipt.fault_slot = work[i].slot;
            return receipt;
        }
    }
    receipt.coordination_commit_id = region->coordination_commit_id;
    receipt.participant_count = (uint32_t)work_count;
    if (!any) { receipt.status = PWW_FRONTIER_NO_WORK; return receipt; }
    if (region->coordination_commit_id == UINT64_MAX) { receipt.status = PWW_FRONTIER_INVALID; return receipt; }
    staged = *region;
    for (rank = 0u; rank < work_count; ++rank) {
        size_t selected = select_next(region, work, work_count, used);
        uint32_t slot = work[selected].slot;
        pww_result_v0 result;
        used[selected] = 1u;
        result = pww_cell_transact(&staged.cells[slot], work[selected].causes,
                                   work[selected].cause_count, work[selected].intents,
                                   work[selected].intent_count);
        if (result.status == PWW_STATUS_PROVISION_REQUIRED) {
            receipt.status = PWW_FRONTIER_PROVISION_REQUIRED;
            receipt.fault_slot = slot;
            receipt.participant_status = result.status;
            receipt.required_pair_limbs = result.required_pair_limbs;
            return receipt;
        }
        if (result.status != PWW_STATUS_OK) {
            receipt.status = PWW_FRONTIER_PARTICIPANT_FAULT;
            receipt.fault_slot = slot;
            receipt.participant_status = result.status;
            return receipt;
        }
        receipt.primitives[rank] = result.primitive;
    }
    staged.coordination_commit_id += 1u;
    *region = staged;
    receipt.status = PWW_FRONTIER_OK;
    receipt.coordination_commit_id = region->coordination_commit_id;
    return receipt;
}

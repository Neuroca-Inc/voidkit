#include "phase_world_cell_v0.h"

#include <string.h>

static void hash_feed(uint64_t *hash, const void *data, size_t length) {
    const uint8_t *bytes = (const uint8_t *)data;
    size_t index;
    for (index = 0u; index < length; ++index) {
        *hash ^= bytes[index];
        *hash *= UINT64_C(0x100000001b3);
    }
}

static pwc_transition_result_v0 result_base(const pwc_cell_v0 *cell) {
    pwc_transition_result_v0 result;
    memset(&result, 0, sizeof(result));
    if (cell != NULL) {
        result.local_commit_id = cell->world.accepted_transition_id;
        result.phase_fingerprint = pwc_phase_fingerprint(&cell->phase);
        result.world_fingerprint = wc_world_fingerprint(&cell->world);
    }
    return result;
}

uint32_t pwc_cell_init(pwc_cell_v0 *cell, uint64_t zone_key, uint64_t axis_capacity) {
    if (cell == NULL || axis_capacity == 0u || axis_capacity > PWC_PHASE_AXIS_STORAGE) {
        return PWC_STATUS_INVALID_ARGUMENT;
    }
    memset(cell, 0, sizeof(*cell));
    cell->phase.u = 1u;
    cell->phase.v = 1u;
    cell->phase.axis_capacity = axis_capacity;
    cell->phase.axis_count = 1u;
    cell->phase.axes[0].origin_product_lo = 1u;
    cell->phase.axes[0].current_product_lo = 1u;
    cell->phase.axes[0].phase_quadrant = 0u;
    cell->phase.axes[0].flags = QBL_ORTHAD_AXIS_ACTIVE;
    wc_world_init(&cell->world, zone_key);
    return PWC_STATUS_OK;
}

uint64_t pwc_phase_fingerprint(const pwc_phase_local_v0 *phase) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    uint64_t count;
    if (phase == NULL) return 0u;
    hash_feed(&hash, &phase->u, sizeof(phase->u));
    hash_feed(&hash, &phase->v, sizeof(phase->v));
    hash_feed(&hash, &phase->domain, sizeof(phase->domain));
    hash_feed(&hash, &phase->local_position, sizeof(phase->local_position));
    hash_feed(&hash, &phase->quarter_turns, sizeof(phase->quarter_turns));
    hash_feed(&hash, &phase->axis_count, sizeof(phase->axis_count));
    count = phase->axis_count;
    if (count > PWC_PHASE_AXIS_STORAGE) count = PWC_PHASE_AXIS_STORAGE;
    hash_feed(&hash, phase->axes, (size_t)count * sizeof(phase->axes[0]));
    return hash;
}

uint64_t pwc_cell_fingerprint(const pwc_cell_v0 *cell) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    uint64_t phase_hash;
    uint64_t world_hash;
    if (cell == NULL) return 0u;
    phase_hash = pwc_phase_fingerprint(&cell->phase);
    world_hash = wc_world_fingerprint(&cell->world);
    hash_feed(&hash, &phase_hash, sizeof(phase_hash));
    hash_feed(&hash, &world_hash, sizeof(world_hash));
    return hash;
}

pwc_transition_result_v0 pwc_cell_transact(
    pwc_cell_v0 *cell,
    const wc_cause_v0 *causes,
    size_t cause_count,
    const wc_intent_v0 *intents,
    size_t intent_count) {
    pwc_transition_result_v0 result;
    pwc_phase_local_v0 staged_phase;
    qbl_orthad_local_state_u64 ffi_phase;
    qbl_transition_u64 qbl_receipt;
    wc_transition_result_v0 world_result;
    uint32_t qbl_status;

    if (cell == NULL || (cause_count != 0u && causes == NULL) ||
        (intent_count != 0u && intents == NULL)) {
        result = result_base(cell);
        result.status = PWC_STATUS_INVALID_ARGUMENT;
        return result;
    }

    if (cause_count == 0u && intent_count == 0u) {
        result = result_base(cell);
        result.status = PWC_STATUS_NO_WORK;
        result.world_status = WC_STATUS_NO_WORK;
        result.qbl_status = QBL_STATUS_OK;
        result.primitive = QBL_PRIMITIVE_NONE;
        return result;
    }

    staged_phase = cell->phase;
    memset(&qbl_receipt, 0, sizeof(qbl_receipt));
    ffi_phase.u = staged_phase.u;
    ffi_phase.v = staged_phase.v;
    ffi_phase.domain = staged_phase.domain;
    ffi_phase.local_position = staged_phase.local_position;
    ffi_phase.quarter_turns = staged_phase.quarter_turns;
    ffi_phase.axes = staged_phase.axes;
    ffi_phase.axis_capacity = staged_phase.axis_capacity;
    ffi_phase.axis_count = staged_phase.axis_count;

    qbl_status = qbl_step_orthad_local_u64(&ffi_phase, &qbl_receipt);
    if (qbl_status != QBL_STATUS_OK) {
        result = result_base(cell);
        result.status = PWC_STATUS_PHASE_FAULT;
        result.qbl_status = qbl_status;
        result.world_status = WC_STATUS_NO_WORK;
        result.primitive = QBL_PRIMITIVE_NONE;
        return result;
    }

    staged_phase.u = ffi_phase.u;
    staged_phase.v = ffi_phase.v;
    staged_phase.domain = ffi_phase.domain;
    staged_phase.local_position = ffi_phase.local_position;
    staged_phase.quarter_turns = ffi_phase.quarter_turns;
    staged_phase.axis_count = ffi_phase.axis_count;

    world_result = wc_world_transact(&cell->world, causes, cause_count, intents, intent_count);
    if (world_result.status != WC_STATUS_OK) {
        result = result_base(cell);
        result.status = PWC_STATUS_WORLD_FAULT;
        result.qbl_status = QBL_STATUS_OK;
        result.world_status = world_result.status;
        result.world_fault = world_result.fault;
        result.primitive = QBL_PRIMITIVE_NONE;
        return result;
    }

    cell->phase = staged_phase;

    result = result_base(cell);
    result.status = PWC_STATUS_OK;
    result.qbl_status = QBL_STATUS_OK;
    result.world_status = WC_STATUS_OK;
    result.world_fault = WC_FAULT_NONE;
    result.primitive = qbl_receipt.primitive;
    result.accepted_intents = world_result.accepted_intents;
    result.rejected_requests = world_result.rejected_requests;
    return result;
}

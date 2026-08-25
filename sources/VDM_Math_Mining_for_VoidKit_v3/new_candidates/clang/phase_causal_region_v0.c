#include "phase_causal_region_v0.h"

#include <limits.h>
#include <string.h>

static int work_arguments_valid(const pcr_frontier_work_v0 *work) {
    if (work == NULL) return 0;
    if (work->cause_count != 0u && work->causes == NULL) return 0;
    if (work->intent_count != 0u && work->intents == NULL) return 0;
    return 1;
}

static int work_is_empty(const pcr_frontier_work_v0 *work) {
    return work->cause_count == 0u && work->intent_count == 0u;
}

static size_t select_next_by_zone(
    const pcr_region_v0 *region,
    const pcr_frontier_work_v0 *work,
    size_t work_count,
    const uint8_t *consumed) {
    size_t selected = SIZE_MAX;
    size_t index;
    for (index = 0u; index < work_count; ++index) {
        uint64_t zone;
        uint64_t selected_zone;
        if (consumed[index] != 0u) continue;
        zone = region->cells[work[index].slot].world.zone_key;
        if (selected == SIZE_MAX) {
            selected = index;
            continue;
        }
        selected_zone = region->cells[work[selected].slot].world.zone_key;
        if (zone < selected_zone) selected = index;
    }
    return selected;
}

static void fill_snapshot_participant(
    pcr_participant_receipt_v0 *out,
    uint32_t slot,
    const pwc_cell_v0 *cell) {
    memset(out, 0, sizeof(*out));
    out->slot = slot;
    out->zone_key = cell->world.zone_key;
    out->before_local_commit_id = cell->world.accepted_transition_id;
    out->after_local_commit_id = cell->world.accepted_transition_id;
    out->phase_fingerprint = pwc_phase_fingerprint(&cell->phase);
    out->world_fingerprint = wc_world_fingerprint(&cell->world);
}

static void fill_committed_participant(
    pcr_participant_receipt_v0 *out,
    uint32_t slot,
    const pwc_cell_v0 *cell,
    uint64_t before,
    const pwc_transition_result_v0 *result) {
    memset(out, 0, sizeof(*out));
    out->slot = slot;
    out->primitive = result->primitive;
    out->zone_key = cell->world.zone_key;
    out->before_local_commit_id = before;
    out->after_local_commit_id = result->local_commit_id;
    out->accepted_intents = result->accepted_intents;
    out->rejected_requests = result->rejected_requests;
    out->phase_fingerprint = result->phase_fingerprint;
    out->world_fingerprint = result->world_fingerprint;
}

uint32_t pcr_region_init(
    pcr_region_v0 *region,
    const uint64_t *zone_keys,
    size_t cell_count,
    uint64_t axis_capacity) {
    size_t first;
    size_t second;
    if (region == NULL || zone_keys == NULL) return PCR_STATUS_INVALID_ARGUMENT;
    if (cell_count == 0u || cell_count > PCR_MAX_CELLS) return PCR_STATUS_INVALID_ARGUMENT;
    for (first = 0u; first < cell_count; ++first) {
        for (second = first + 1u; second < cell_count; ++second) {
            if (zone_keys[first] == zone_keys[second]) return PCR_STATUS_DUPLICATE_ZONE;
        }
    }
    memset(region, 0, sizeof(*region));
    region->cell_count = (uint32_t)cell_count;
    for (first = 0u; first < cell_count; ++first) {
        if (pwc_cell_init(&region->cells[first], zone_keys[first], axis_capacity) != PWC_STATUS_OK) {
            memset(region, 0, sizeof(*region));
            return PCR_STATUS_INVALID_ARGUMENT;
        }
    }
    return PCR_STATUS_OK;
}

pcr_frontier_receipt_v0 pcr_transact_frontier(
    pcr_region_v0 *region,
    const pcr_frontier_work_v0 *work,
    size_t work_count) {
    pcr_frontier_receipt_v0 receipt;
    pwc_cell_v0 staged[PCR_MAX_CELLS];
    pwc_transition_result_v0 results[PCR_MAX_CELLS];
    uint64_t before_ids[PCR_MAX_CELLS];
    uint8_t seen_slots[PCR_MAX_CELLS];
    uint8_t consumed[PCR_MAX_CELLS];
    int any_empty = 0;
    int any_nonempty = 0;
    size_t index;
    size_t rank;

    memset(&receipt, 0, sizeof(receipt));
    if (region == NULL) {
        receipt.status = PCR_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (work_count > region->cell_count || work_count > PCR_MAX_CELLS) {
        receipt.status = PCR_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (work_count != 0u && work == NULL) {
        receipt.status = PCR_STATUS_INVALID_ARGUMENT;
        return receipt;
    }

    memset(seen_slots, 0, sizeof(seen_slots));
    for (index = 0u; index < work_count; ++index) {
        uint32_t slot;
        if (!work_arguments_valid(&work[index])) {
            receipt.status = PCR_STATUS_INVALID_ARGUMENT;
            return receipt;
        }
        slot = work[index].slot;
        if (slot >= region->cell_count) {
            receipt.status = PCR_STATUS_INVALID_SLOT;
            receipt.fault_slot = slot;
            return receipt;
        }
        if (seen_slots[slot] != 0u) {
            receipt.status = PCR_STATUS_DUPLICATE_PARTICIPANT;
            receipt.fault_slot = slot;
            return receipt;
        }
        seen_slots[slot] = 1u;
        if (work_is_empty(&work[index])) any_empty = 1;
        else any_nonempty = 1;
    }

    receipt.participant_count = (uint32_t)work_count;
    receipt.coordination_commit_id = region->coordination_commit_id;

    if (!any_nonempty) {
        memset(consumed, 0, sizeof(consumed));
        for (rank = 0u; rank < work_count; ++rank) {
            size_t selected = select_next_by_zone(region, work, work_count, consumed);
            uint32_t slot = work[selected].slot;
            consumed[selected] = 1u;
            fill_snapshot_participant(&receipt.participants[rank], slot, &region->cells[slot]);
        }
        receipt.status = PCR_STATUS_NO_WORK;
        return receipt;
    }

    if (any_empty) {
        for (index = 0u; index < work_count; ++index) {
            if (work_is_empty(&work[index])) {
                receipt.status = PCR_STATUS_PARTICIPANT_WORK_MISSING;
                receipt.fault_slot = work[index].slot;
                return receipt;
            }
        }
    }

    for (index = 0u; index < work_count; ++index) {
        uint32_t slot = work[index].slot;
        uint64_t actual = region->cells[slot].world.accepted_transition_id;
        if (work[index].expected_local_commit_id != actual) {
            receipt.status = PCR_STATUS_STALE_VERSION;
            receipt.fault_slot = slot;
            return receipt;
        }
    }

    if (region->coordination_commit_id == UINT64_MAX) {
        receipt.status = PCR_STATUS_COORDINATION_EXHAUSTED;
        return receipt;
    }

    memcpy(staged, region->cells, sizeof(staged));
    memset(results, 0, sizeof(results));
    memset(before_ids, 0, sizeof(before_ids));
    memset(consumed, 0, sizeof(consumed));

    for (rank = 0u; rank < work_count; ++rank) {
        size_t selected = select_next_by_zone(region, work, work_count, consumed);
        uint32_t slot = work[selected].slot;
        pwc_transition_result_v0 result;
        consumed[selected] = 1u;
        before_ids[rank] = staged[slot].world.accepted_transition_id;
        result = pwc_cell_transact(
            &staged[slot],
            work[selected].causes,
            work[selected].cause_count,
            work[selected].intents,
            work[selected].intent_count);
        results[rank] = result;
        if (result.status != PWC_STATUS_OK) {
            receipt.status = PCR_STATUS_PARTICIPANT_FAULT;
            receipt.fault_slot = slot;
            receipt.participant_status = result.status;
            receipt.participant_qbl_status = result.qbl_status;
            receipt.participant_world_status = result.world_status;
            receipt.participant_world_fault = result.world_fault;
            return receipt;
        }
        fill_committed_participant(
            &receipt.participants[rank],
            slot,
            &staged[slot],
            before_ids[rank],
            &results[rank]);
    }

    memcpy(region->cells, staged, sizeof(region->cells));
    region->coordination_commit_id += 1u;
    receipt.status = PCR_STATUS_OK;
    receipt.coordination_commit_id = region->coordination_commit_id;
    return receipt;
}

#include "phase_causal_pair_v0.h"

#include <string.h>

typedef struct ordered_participant_v0 {
    pwc_cell_v0 *cell;
    const pcc_local_work_v0 *work;
} ordered_participant_v0;

static int work_is_empty(const pcc_local_work_v0 *work) {
    return work->cause_count == 0u && work->intent_count == 0u;
}

static int work_arguments_valid(const pcc_local_work_v0 *work) {
    if (work == NULL) return 0;
    if (work->cause_count != 0u && work->causes == NULL) return 0;
    if (work->intent_count != 0u && work->intents == NULL) return 0;
    return 1;
}

static pcc_participant_receipt_v0 snapshot_participant(const pwc_cell_v0 *cell) {
    pcc_participant_receipt_v0 receipt;
    memset(&receipt, 0, sizeof(receipt));
    if (cell != NULL) {
        receipt.zone_key = cell->world.zone_key;
        receipt.before_local_commit_id = cell->world.accepted_transition_id;
        receipt.after_local_commit_id = cell->world.accepted_transition_id;
        receipt.phase_fingerprint = pwc_phase_fingerprint(&cell->phase);
        receipt.world_fingerprint = wc_world_fingerprint(&cell->world);
    }
    return receipt;
}

static pcc_pair_receipt_v0 base_receipt(
    const pcc_coordinator_v0 *coordinator,
    const pwc_cell_v0 *left,
    const pwc_cell_v0 *right) {
    pcc_pair_receipt_v0 receipt;
    pcc_participant_receipt_v0 a;
    pcc_participant_receipt_v0 b;
    memset(&receipt, 0, sizeof(receipt));
    if (coordinator != NULL) receipt.coordination_commit_id = coordinator->coordination_commit_id;
    a = snapshot_participant(left);
    b = snapshot_participant(right);
    if (a.zone_key <= b.zone_key) {
        receipt.first = a;
        receipt.second = b;
    } else {
        receipt.first = b;
        receipt.second = a;
    }
    return receipt;
}

static void fill_committed_participant(
    pcc_participant_receipt_v0 *receipt,
    const pwc_cell_v0 *cell,
    uint64_t before,
    const pwc_transition_result_v0 *result) {
    receipt->zone_key = cell->world.zone_key;
    receipt->before_local_commit_id = before;
    receipt->after_local_commit_id = cell->world.accepted_transition_id;
    receipt->primitive = result->primitive;
    receipt->accepted_intents = result->accepted_intents;
    receipt->rejected_requests = result->rejected_requests;
    receipt->phase_fingerprint = pwc_phase_fingerprint(&cell->phase);
    receipt->world_fingerprint = wc_world_fingerprint(&cell->world);
}

void pcc_coordinator_init(pcc_coordinator_v0 *coordinator) {
    if (coordinator != NULL) memset(coordinator, 0, sizeof(*coordinator));
}

pcc_pair_receipt_v0 pcc_transact_pair(
    pcc_coordinator_v0 *coordinator,
    pwc_cell_v0 *left,
    const pcc_local_work_v0 *left_work,
    pwc_cell_v0 *right,
    const pcc_local_work_v0 *right_work) {
    pcc_pair_receipt_v0 receipt;
    ordered_participant_v0 first;
    ordered_participant_v0 second;
    pwc_cell_v0 staged_first;
    pwc_cell_v0 staged_second;
    pwc_transition_result_v0 first_result;
    pwc_transition_result_v0 second_result;
    uint64_t first_before;
    uint64_t second_before;

    receipt = base_receipt(coordinator, left, right);
    if (coordinator == NULL || left == NULL || right == NULL || left == right ||
        !work_arguments_valid(left_work) || !work_arguments_valid(right_work)) {
        receipt.status = PCC_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (left->world.zone_key == right->world.zone_key) {
        receipt.status = PCC_STATUS_INVALID_ARGUMENT;
        return receipt;
    }

    if (work_is_empty(left_work) && work_is_empty(right_work)) {
        receipt.status = PCC_STATUS_NO_WORK;
        return receipt;
    }
    if (work_is_empty(left_work) || work_is_empty(right_work)) {
        receipt.status = PCC_STATUS_PARTICIPANT_WORK_MISSING;
        return receipt;
    }

    if (left->world.zone_key < right->world.zone_key) {
        first.cell = left;
        first.work = left_work;
        second.cell = right;
        second.work = right_work;
    } else {
        first.cell = right;
        first.work = right_work;
        second.cell = left;
        second.work = left_work;
    }

    if (first.work->expected_local_commit_id != first.cell->world.accepted_transition_id) {
        receipt.status = PCC_STATUS_STALE_VERSION;
        receipt.fault_participant = PCC_PARTICIPANT_FIRST;
        return receipt;
    }
    if (second.work->expected_local_commit_id != second.cell->world.accepted_transition_id) {
        receipt.status = PCC_STATUS_STALE_VERSION;
        receipt.fault_participant = PCC_PARTICIPANT_SECOND;
        return receipt;
    }
    if (coordinator->coordination_commit_id == UINT64_MAX) {
        receipt.status = PCC_STATUS_COORDINATION_EXHAUSTED;
        return receipt;
    }

    first_before = first.cell->world.accepted_transition_id;
    second_before = second.cell->world.accepted_transition_id;
    staged_first = *first.cell;
    staged_second = *second.cell;

    first_result = pwc_cell_transact(
        &staged_first,
        first.work->causes,
        first.work->cause_count,
        first.work->intents,
        first.work->intent_count);
    if (first_result.status != PWC_STATUS_OK) {
        receipt.status = PCC_STATUS_PARTICIPANT_FAULT;
        receipt.fault_participant = PCC_PARTICIPANT_FIRST;
        receipt.participant_status = first_result.status;
        receipt.participant_qbl_status = first_result.qbl_status;
        receipt.participant_world_status = first_result.world_status;
        receipt.participant_world_fault = first_result.world_fault;
        return receipt;
    }

    second_result = pwc_cell_transact(
        &staged_second,
        second.work->causes,
        second.work->cause_count,
        second.work->intents,
        second.work->intent_count);
    if (second_result.status != PWC_STATUS_OK) {
        receipt.status = PCC_STATUS_PARTICIPANT_FAULT;
        receipt.fault_participant = PCC_PARTICIPANT_SECOND;
        receipt.participant_status = second_result.status;
        receipt.participant_qbl_status = second_result.qbl_status;
        receipt.participant_world_status = second_result.world_status;
        receipt.participant_world_fault = second_result.world_fault;
        return receipt;
    }

    *first.cell = staged_first;
    *second.cell = staged_second;
    coordinator->coordination_commit_id += 1u;

    memset(&receipt, 0, sizeof(receipt));
    receipt.status = PCC_STATUS_OK;
    receipt.coordination_commit_id = coordinator->coordination_commit_id;
    fill_committed_participant(&receipt.first, first.cell, first_before, &first_result);
    fill_committed_participant(&receipt.second, second.cell, second_before, &second_result);
    return receipt;
}

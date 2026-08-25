#include "phase_return_journey_v0.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

static void hash_feed(uint64_t *hash, const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t *)data;
    size_t index;
    for (index = 0u; index < size; ++index) {
        *hash ^= bytes[index];
        *hash *= UINT64_C(1099511628211);
    }
}

static prj_receipt_v0 receipt_base(const prj_return_journey_v0 *journey) {
    prj_receipt_v0 receipt;
    memset(&receipt, 0, sizeof(receipt));
    if (journey == NULL) return receipt;
    receipt.return_commit_id = journey->return_commit_id;
    receipt.active_local_commit_id = journey->active.world.accepted_transition_id;
    receipt.origin_local_commit_id = journey->origin.world.accepted_transition_id;
    receipt.traveler_key = journey->traveler_key;
    receipt.active_actor_key = journey->active_actor_key;
    receipt.return_actor_key = journey->return_actor_key;
    receipt.origin_snapshot_bytes = journey->origin_snapshot_bytes;
    receipt.active_phase_fingerprint = pww_phase_fingerprint(&journey->active.phase);
    receipt.active_world_fingerprint = wc_world_fingerprint(&journey->active.world);
    if (journey->origin_resident != 0u) {
        receipt.origin_phase_fingerprint = pww_phase_fingerprint(&journey->origin.phase);
        receipt.origin_world_fingerprint = wc_world_fingerprint(&journey->origin.world);
    } else {
        receipt.origin_phase_fingerprint = journey->origin_snapshot_receipt.phase_fingerprint;
        receipt.origin_world_fingerprint = journey->origin_snapshot_receipt.world_fingerprint;
    }
    return receipt;
}

static uint32_t map_active_failure(const pww_result_v0 *result,
                                   prj_receipt_v0 *receipt) {
    receipt->fault_participant = PRJ_PARTICIPANT_ACTIVE;
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PRJ_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PRJ_STATUS_ACTIVE_WORLD_FAILURE;
    return PRJ_STATUS_ACTIVE_PHASE_FAILURE;
}

static uint32_t map_origin_failure(const pww_result_v0 *result,
                                   prj_receipt_v0 *receipt) {
    receipt->fault_participant = PRJ_PARTICIPANT_ORIGIN;
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PRJ_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PRJ_STATUS_ORIGIN_WORLD_FAILURE;
    return PRJ_STATUS_ORIGIN_PHASE_FAILURE;
}

uint32_t prj_init_from_completed_traversal(
    prj_return_journey_v0 *journey,
    const pt_streamed_traversal_v0 *completed) {
    int active_slot;
    if (journey == NULL || completed == NULL || completed->traversed == 0u ||
        completed->traversal_commit_id == 0u || completed->traveler_key == 0u ||
        completed->source.world.object_count != 0u ||
        completed->destination.world.object_count != 1u ||
        completed->destination_actor_key == 0u)
        return PRJ_STATUS_OUTBOUND_INCOMPLETE;
    active_slot = wc_world_resolve(&completed->destination.world,
                                   completed->destination_actor_key);
    if (active_slot < 0 ||
        completed->destination.world.kind[active_slot] != WC_KIND_ACTOR)
        return PRJ_STATUS_ACTIVE_ACTOR_MISSING;
    memset(journey, 0, sizeof(*journey));
    journey->origin = completed->source;
    journey->active = completed->destination;
    journey->region_key = completed->region_key;
    journey->traveler_key = completed->traveler_key;
    journey->origin_zone_key = completed->source.world.zone_key;
    journey->active_zone_key = completed->destination.world.zone_key;
    journey->outbound_traversal_commit_id = completed->traversal_commit_id;
    journey->active_actor_key = completed->destination_actor_key;
    journey->origin_resident = 1u;
    return PRJ_STATUS_OK;
}

prj_receipt_v0 prj_evict_origin(prj_return_journey_v0 *journey) {
    prj_return_journey_v0 staged;
    prj_receipt_v0 receipt;
    pcs_receipt_v1 snapshot_receipt;
    uint32_t status;

    receipt = receipt_base(journey);
    if (journey == NULL) {
        receipt.status = PRJ_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (journey->origin_resident == 0u || journey->origin_evicted != 0u) {
        receipt.status = PRJ_STATUS_ALREADY_EVICTED;
        return receipt;
    }
    if (journey->returned != 0u) {
        receipt.status = PRJ_STATUS_ALREADY_RETURNED;
        return receipt;
    }
    if (journey->origin.world.object_count != 0u) {
        receipt.status = PRJ_STATUS_ORIGIN_NOT_EMPTY;
        return receipt;
    }
    staged = *journey;
    status = pcs_snapshot_encode_v1(&staged.origin,
                                    staged.origin_snapshot,
                                    sizeof(staged.origin_snapshot),
                                    &snapshot_receipt);
    if (status != PCS_STATUS_OK) {
        receipt.status = PRJ_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    staged.origin_snapshot_receipt = snapshot_receipt;
    staged.origin_snapshot_bytes = snapshot_receipt.snapshot_bytes;
    memset(&staged.origin, 0, sizeof(staged.origin));
    staged.origin_resident = 0u;
    staged.origin_evicted = 1u;
    *journey = staged;
    receipt = receipt_base(journey);
    receipt.status = PRJ_STATUS_OK;
    return receipt;
}

prj_receipt_v0 prj_restore_origin(prj_return_journey_v0 *journey) {
    prj_return_journey_v0 staged;
    prj_receipt_v0 receipt;
    pww_cell_v0 candidate;
    pcs_receipt_v1 decoded;
    uint32_t status;

    receipt = receipt_base(journey);
    if (journey == NULL) {
        receipt.status = PRJ_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (journey->origin_resident != 0u || journey->origin_evicted == 0u ||
        journey->origin_snapshot_bytes == 0u) {
        receipt.status = PRJ_STATUS_NOT_EVICTED;
        return receipt;
    }
    staged = *journey;
    memset(&candidate, 0, sizeof(candidate));
    status = pcs_snapshot_decode_v1(staged.origin_snapshot,
                                    (size_t)staged.origin_snapshot_bytes,
                                    &candidate,
                                    &decoded);
    if (status != PCS_STATUS_OK) {
        receipt.status = PRJ_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    if (decoded.source_zone_key != staged.origin_zone_key ||
        decoded.source_local_commit_id != staged.origin_snapshot_receipt.source_local_commit_id ||
        decoded.phase_fingerprint != staged.origin_snapshot_receipt.phase_fingerprint ||
        decoded.world_fingerprint != staged.origin_snapshot_receipt.world_fingerprint) {
        receipt.status = PRJ_STATUS_SNAPSHOT_MISMATCH;
        return receipt;
    }
    staged.origin = candidate;
    memset(staged.origin_snapshot, 0, sizeof(staged.origin_snapshot));
    memset(&staged.origin_snapshot_receipt, 0, sizeof(staged.origin_snapshot_receipt));
    staged.origin_snapshot_bytes = 0u;
    staged.origin_resident = 1u;
    staged.origin_evicted = 0u;
    *journey = staged;
    receipt = receipt_base(journey);
    receipt.status = PRJ_STATUS_OK;
    return receipt;
}

prj_receipt_v0 prj_advance_active(
    prj_return_journey_v0 *journey,
    uint32_t source_sequence,
    uint64_t payload0,
    uint64_t payload1) {
    prj_return_journey_v0 staged;
    prj_receipt_v0 receipt;
    wc_cause_v0 cause;
    pww_result_v0 result;

    receipt = receipt_base(journey);
    if (journey == NULL || source_sequence == 0u) {
        receipt.status = PRJ_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (journey->returned != 0u) {
        receipt.status = PRJ_STATUS_ALREADY_RETURNED;
        return receipt;
    }
    staged = *journey;
    cause = wc_external_input(source_sequence, payload0, payload1);
    result = pww_cell_transact(&staged.active, &cause, 1u, NULL, 0u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_active_failure(&result, &receipt);
        return receipt;
    }
    receipt.active_primitive = result.primitive;
    *journey = staged;
    receipt = receipt_base(journey);
    receipt.status = PRJ_STATUS_OK;
    receipt.active_primitive = result.primitive;
    return receipt;
}

uint32_t prj_provision_active_pair_limbs(
    prj_return_journey_v0 *journey,
    uint32_t new_limit) {
    if (journey == NULL) return PRJ_STATUS_INVALID_ARGUMENT;
    if (pww_cell_provision_pair_limbs(&journey->active, new_limit) != PWW_STATUS_OK)
        return PRJ_STATUS_INVALID_ARGUMENT;
    return PRJ_STATUS_OK;
}

prj_receipt_v0 prj_return_to_origin(
    prj_return_journey_v0 *journey,
    uint64_t return_cause_id,
    uint32_t active_sequence,
    uint32_t origin_sequence) {
    prj_return_journey_v0 staged;
    prj_receipt_v0 receipt;
    wc_cause_v0 active_cause;
    wc_cause_v0 origin_cause;
    wc_intent_v0 active_intent;
    wc_intent_v0 origin_intent;
    pww_result_v0 active_result;
    pww_result_v0 origin_result;
    uint64_t next_commit_id;
    uint64_t expected_origin_key;
    int active_slot;
    int origin_slot;
    uint16_t health;

    receipt = receipt_base(journey);
    if (journey == NULL || return_cause_id == 0u ||
        active_sequence == 0u || origin_sequence == 0u) {
        receipt.status = PRJ_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (journey->returned != 0u) {
        if (journey->consumed_return_cause_id == return_cause_id) {
            receipt.status = PRJ_STATUS_DUPLICATE_IGNORED;
            return receipt;
        }
        receipt.status = PRJ_STATUS_ALREADY_RETURNED;
        return receipt;
    }
    if (journey->origin_resident == 0u) {
        receipt.status = PRJ_STATUS_ORIGIN_NOT_RESIDENT;
        return receipt;
    }
    active_slot = wc_world_resolve(&journey->active.world,
                                   journey->active_actor_key);
    if (active_slot < 0 || journey->active.world.kind[active_slot] != WC_KIND_ACTOR) {
        receipt.status = PRJ_STATUS_ACTIVE_ACTOR_MISSING;
        return receipt;
    }
    health = journey->active.world.actor_health[active_slot];
    if (journey->return_commit_id == UINT64_MAX) {
        receipt.status = PRJ_STATUS_COMMIT_EXHAUSTED;
        return receipt;
    }
    next_commit_id = journey->return_commit_id + 1u;
    staged = *journey;

    active_cause = wc_external_input(active_sequence,
                                     staged.traveler_key,
                                     staged.origin_zone_key);
    active_intent = wc_despawn(active_sequence, staged.active_actor_key);
    active_result = pww_cell_transact(&staged.active,
                                      &active_cause, 1u,
                                      &active_intent, 1u);
    if (active_result.status != PWW_STATUS_OK) {
        receipt.status = map_active_failure(&active_result, &receipt);
        return receipt;
    }

    expected_origin_key = staged.origin.world.next_object_key;
    origin_cause = wc_external_input(origin_sequence,
                                     staged.traveler_key,
                                     return_cause_id);
    origin_intent = wc_spawn_actor(origin_sequence, health, WC_NON_SPATIAL_SITE);
    origin_result = pww_cell_transact(&staged.origin,
                                      &origin_cause, 1u,
                                      &origin_intent, 1u);
    if (origin_result.status != PWW_STATUS_OK) {
        receipt.status = map_origin_failure(&origin_result, &receipt);
        return receipt;
    }
    origin_slot = wc_world_resolve(&staged.origin.world, expected_origin_key);
    if (origin_slot < 0 ||
        staged.origin.world.kind[origin_slot] != WC_KIND_ACTOR ||
        staged.origin.world.actor_health[origin_slot] != health) {
        receipt.status = PRJ_STATUS_ORIGIN_SPAWN_REJECTED;
        receipt.fault_participant = PRJ_PARTICIPANT_ORIGIN;
        return receipt;
    }

    staged.return_actor_key = expected_origin_key;
    staged.consumed_return_cause_id = return_cause_id;
    staged.return_commit_id = next_commit_id;
    staged.returned = 1u;
    *journey = staged;

    receipt = receipt_base(journey);
    receipt.status = PRJ_STATUS_OK;
    receipt.active_primitive = active_result.primitive;
    receipt.origin_primitive = origin_result.primitive;
    return receipt;
}

uint64_t prj_fingerprint(const prj_return_journey_v0 *journey) {
    uint64_t hash = UINT64_C(1469598103934665603);
    if (journey == NULL) return 0u;
    hash_feed(&hash, &journey->region_key, sizeof(journey->region_key));
    hash_feed(&hash, &journey->traveler_key, sizeof(journey->traveler_key));
    hash_feed(&hash, &journey->origin_zone_key, sizeof(journey->origin_zone_key));
    hash_feed(&hash, &journey->active_zone_key, sizeof(journey->active_zone_key));
    hash_feed(&hash, &journey->outbound_traversal_commit_id,
              sizeof(journey->outbound_traversal_commit_id));
    hash_feed(&hash, &journey->active_actor_key, sizeof(journey->active_actor_key));
    hash_feed(&hash, &journey->return_actor_key, sizeof(journey->return_actor_key));
    hash_feed(&hash, &journey->consumed_return_cause_id,
              sizeof(journey->consumed_return_cause_id));
    hash_feed(&hash, &journey->return_commit_id, sizeof(journey->return_commit_id));
    hash_feed(&hash, &journey->origin_resident, sizeof(journey->origin_resident));
    hash_feed(&hash, &journey->origin_evicted, sizeof(journey->origin_evicted));
    hash_feed(&hash, &journey->returned, sizeof(journey->returned));
    if (journey->origin_resident != 0u) {
        uint64_t phase = pww_phase_fingerprint(&journey->origin.phase);
        uint64_t world = wc_world_fingerprint(&journey->origin.world);
        hash_feed(&hash, &phase, sizeof(phase));
        hash_feed(&hash, &world, sizeof(world));
    } else {
        hash_feed(&hash, &journey->origin_snapshot_receipt,
                  sizeof(journey->origin_snapshot_receipt));
        hash_feed(&hash, journey->origin_snapshot,
                  (size_t)journey->origin_snapshot_bytes);
    }
    {
        uint64_t phase = pww_phase_fingerprint(&journey->active.phase);
        uint64_t world = wc_world_fingerprint(&journey->active.world);
        hash_feed(&hash, &phase, sizeof(phase));
        hash_feed(&hash, &world, sizeof(world));
    }
    return hash;
}

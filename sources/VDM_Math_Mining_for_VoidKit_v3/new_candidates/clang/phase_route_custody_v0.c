#include "phase_route_custody_v0.h"

#include <limits.h>
#include <string.h>

static void hash_feed(uint64_t *hash, const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t *)data;
    size_t i;
    for (i = 0u; i < size; ++i) {
        *hash ^= bytes[i];
        *hash *= UINT64_C(1099511628211);
    }
}

static int zones_unique(const uint64_t *zone_keys, size_t count) {
    size_t i;
    size_t j;
    for (i = 0u; i < count; ++i) {
        if (zone_keys[i] == 0u) return 0;
        for (j = i + 1u; j < count; ++j) {
            if (zone_keys[i] == zone_keys[j]) return 0;
        }
    }
    return 1;
}

static prt_receipt_v0 receipt_base(const prt_route_v0 *route,
                                   uint32_t source_index,
                                   uint32_t destination_index) {
    prt_receipt_v0 receipt;
    memset(&receipt, 0, sizeof(receipt));
    receipt.source_index = source_index;
    receipt.destination_index = destination_index;
    if (route == NULL) return receipt;
    receipt.route_commit_id = route->route_commit_id;
    receipt.active_index = route->active_index;
    receipt.traveler_key = route->traveler_key;
    receipt.actor_key = route->actor_key;
    if (source_index < route->route_length) {
        const prt_route_slot_v0 *slot = &route->slots[source_index];
        if (slot->resident != 0u) {
            receipt.source_local_commit_id = slot->cell.world.accepted_transition_id;
            receipt.source_phase_fingerprint = pww_phase_fingerprint(&slot->cell.phase);
            receipt.source_world_fingerprint = wc_world_fingerprint(&slot->cell.world);
        } else {
            receipt.source_local_commit_id = slot->snapshot_receipt.source_local_commit_id;
            receipt.source_phase_fingerprint = slot->snapshot_receipt.phase_fingerprint;
            receipt.source_world_fingerprint = slot->snapshot_receipt.world_fingerprint;
            receipt.snapshot_bytes = slot->snapshot_bytes;
        }
    }
    if (destination_index < route->route_length) {
        const prt_route_slot_v0 *slot = &route->slots[destination_index];
        if (slot->resident != 0u) {
            receipt.destination_local_commit_id = slot->cell.world.accepted_transition_id;
            receipt.destination_phase_fingerprint = pww_phase_fingerprint(&slot->cell.phase);
            receipt.destination_world_fingerprint = wc_world_fingerprint(&slot->cell.world);
        } else {
            receipt.destination_local_commit_id = slot->snapshot_receipt.source_local_commit_id;
            receipt.destination_phase_fingerprint = slot->snapshot_receipt.phase_fingerprint;
            receipt.destination_world_fingerprint = slot->snapshot_receipt.world_fingerprint;
            receipt.snapshot_bytes = slot->snapshot_bytes;
        }
    }
    return receipt;
}

static uint32_t map_source_failure(const pww_result_v0 *result,
                                   prt_receipt_v0 *receipt) {
    receipt->fault_participant = PRT_PARTICIPANT_SOURCE;
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PRT_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PRT_STATUS_SOURCE_WORLD_FAILURE;
    return PRT_STATUS_SOURCE_PHASE_FAILURE;
}

static uint32_t map_destination_failure(const pww_result_v0 *result,
                                        prt_receipt_v0 *receipt) {
    receipt->fault_participant = PRT_PARTICIPANT_DESTINATION;
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PRT_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PRT_STATUS_DESTINATION_WORLD_FAILURE;
    return PRT_STATUS_DESTINATION_PHASE_FAILURE;
}

uint32_t prt_route_init(prt_route_v0 *route,
                        uint64_t route_key,
                        uint64_t traveler_key,
                        const uint64_t *zone_keys,
                        size_t route_length,
                        uint32_t pair_limb_limit) {
    size_t i;
    if (route == NULL || zone_keys == NULL || route_key == 0u || traveler_key == 0u ||
        route_length < 2u || route_length > PRT_ROUTE_MAX_SLOTS ||
        pair_limb_limit == 0u || !zones_unique(zone_keys, route_length))
        return PRT_STATUS_INVALID_ARGUMENT;
    memset(route, 0, sizeof(*route));
    route->route_key = route_key;
    route->traveler_key = traveler_key;
    route->route_length = (uint32_t)route_length;
    route->active_index = 0u;
    for (i = 0u; i < route_length; ++i) {
        if (pww_cell_init(&route->slots[i].cell,
                          zone_keys[i], pair_limb_limit) != PWW_STATUS_OK) {
            memset(route, 0, sizeof(*route));
            return PRT_STATUS_INVALID_ARGUMENT;
        }
        route->slots[i].zone_key = zone_keys[i];
        route->slots[i].resident = 1u;
        route->slots[i].retained = 1u;
    }
    return PRT_STATUS_OK;
}

prt_receipt_v0 prt_bootstrap(prt_route_v0 *route,
                             uint64_t bootstrap_cause_id,
                             uint32_t source_sequence,
                             uint16_t actor_health) {
    prt_route_v0 staged;
    prt_receipt_v0 receipt = receipt_base(route, 0u, 0u);
    wc_cause_v0 cause;
    wc_intent_v0 intent;
    pww_result_v0 result;
    uint64_t expected_key;
    int slot;
    if (route == NULL || bootstrap_cause_id == 0u || source_sequence == 0u || actor_health == 0u) {
        receipt.status = PRT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (route->bootstrapped != 0u) {
        receipt.status = PRT_STATUS_ALREADY_BOOTSTRAPPED;
        return receipt;
    }
    staged = *route;
    expected_key = staged.slots[0].cell.world.next_object_key;
    cause = wc_external_input(source_sequence, staged.traveler_key, bootstrap_cause_id);
    intent = wc_spawn_actor(source_sequence, actor_health, WC_NON_SPATIAL_SITE);
    result = pww_cell_transact(&staged.slots[0].cell, &cause, 1u, &intent, 1u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&result, &receipt);
        return receipt;
    }
    slot = wc_world_resolve(&staged.slots[0].cell.world, expected_key);
    if (slot < 0 || staged.slots[0].cell.world.kind[slot] != WC_KIND_ACTOR ||
        staged.slots[0].cell.world.actor_health[slot] != actor_health) {
        receipt.status = PRT_STATUS_DESTINATION_SPAWN_REJECTED;
        return receipt;
    }
    staged.actor_key = expected_key;
    staged.bootstrapped = 1u;
    *route = staged;
    receipt = receipt_base(route, 0u, 0u);
    receipt.status = PRT_STATUS_OK;
    receipt.source_primitive = result.primitive;
    return receipt;
}

prt_receipt_v0 prt_advance_active(prt_route_v0 *route,
                                  uint32_t source_sequence,
                                  uint64_t payload0,
                                  uint64_t payload1) {
    prt_route_v0 staged;
    prt_receipt_v0 receipt;
    wc_cause_v0 cause;
    pww_result_v0 result;
    uint32_t active;
    if (route == NULL || source_sequence == 0u) {
        receipt = receipt_base(route, 0u, 0u);
        receipt.status = PRT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    active = route->active_index;
    receipt = receipt_base(route, active, active);
    if (route->bootstrapped == 0u) {
        receipt.status = PRT_STATUS_NOT_BOOTSTRAPPED;
        return receipt;
    }
    if (active >= route->route_length || route->slots[active].resident == 0u) {
        receipt.status = PRT_STATUS_SLOT_NOT_RESIDENT;
        return receipt;
    }
    staged = *route;
    cause = wc_external_input(source_sequence, payload0, payload1);
    result = pww_cell_transact(&staged.slots[active].cell, &cause, 1u, NULL, 0u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&result, &receipt);
        return receipt;
    }
    *route = staged;
    receipt = receipt_base(route, active, active);
    receipt.status = PRT_STATUS_OK;
    receipt.source_primitive = result.primitive;
    return receipt;
}

prt_receipt_v0 prt_evict_slot(prt_route_v0 *route, uint32_t slot_index) {
    prt_route_v0 staged;
    prt_receipt_v0 receipt = receipt_base(route, slot_index, slot_index);
    pcs_receipt_v1 snapshot_receipt;
    uint32_t status;
    if (route == NULL) {
        receipt.status = PRT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (slot_index >= route->route_length) {
        receipt.status = PRT_STATUS_SLOT_RANGE;
        return receipt;
    }
    if (slot_index == route->active_index) {
        receipt.status = PRT_STATUS_ACTIVE_SLOT;
        return receipt;
    }
    if (route->slots[slot_index].resident == 0u) {
        receipt.status = PRT_STATUS_SLOT_NOT_RESIDENT;
        return receipt;
    }
    if (route->slots[slot_index].cell.world.object_count != 0u) {
        receipt.status = PRT_STATUS_SLOT_NOT_EMPTY;
        return receipt;
    }
    staged = *route;
    status = pcs_snapshot_encode_v1(&staged.slots[slot_index].cell,
                                    staged.slots[slot_index].snapshot,
                                    sizeof(staged.slots[slot_index].snapshot),
                                    &snapshot_receipt);
    if (status != PCS_STATUS_OK) {
        receipt.status = PRT_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    staged.slots[slot_index].snapshot_receipt = snapshot_receipt;
    staged.slots[slot_index].snapshot_bytes = snapshot_receipt.snapshot_bytes;
    memset(&staged.slots[slot_index].cell, 0,
           sizeof(staged.slots[slot_index].cell));
    staged.slots[slot_index].resident = 0u;
    *route = staged;
    receipt = receipt_base(route, slot_index, slot_index);
    receipt.status = PRT_STATUS_OK;
    receipt.snapshot_bytes = snapshot_receipt.snapshot_bytes;
    return receipt;
}

prt_receipt_v0 prt_restore_slot(prt_route_v0 *route, uint32_t slot_index) {
    prt_route_v0 staged;
    prt_receipt_v0 receipt = receipt_base(route, slot_index, slot_index);
    pww_cell_v0 candidate;
    pcs_receipt_v1 decoded;
    uint32_t status;
    if (route == NULL) {
        receipt.status = PRT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (slot_index >= route->route_length) {
        receipt.status = PRT_STATUS_SLOT_RANGE;
        return receipt;
    }
    if (route->slots[slot_index].resident != 0u) {
        receipt.status = PRT_STATUS_SLOT_ALREADY_RESIDENT;
        return receipt;
    }
    if (route->slots[slot_index].snapshot_bytes == 0u) {
        receipt.status = PRT_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    memset(&candidate, 0, sizeof(candidate));
    status = pcs_snapshot_decode_v1(route->slots[slot_index].snapshot,
                                    (size_t)route->slots[slot_index].snapshot_bytes,
                                    &candidate, &decoded);
    if (status != PCS_STATUS_OK) {
        receipt.status = PRT_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    if (decoded.source_zone_key != route->slots[slot_index].zone_key ||
        decoded.source_local_commit_id != route->slots[slot_index].snapshot_receipt.source_local_commit_id ||
        decoded.phase_fingerprint != route->slots[slot_index].snapshot_receipt.phase_fingerprint ||
        decoded.world_fingerprint != route->slots[slot_index].snapshot_receipt.world_fingerprint) {
        receipt.status = PRT_STATUS_SNAPSHOT_MISMATCH;
        return receipt;
    }
    staged = *route;
    staged.slots[slot_index].cell = candidate;
    memset(staged.slots[slot_index].snapshot, 0,
           sizeof(staged.slots[slot_index].snapshot));
    memset(&staged.slots[slot_index].snapshot_receipt, 0,
           sizeof(staged.slots[slot_index].snapshot_receipt));
    staged.slots[slot_index].snapshot_bytes = 0u;
    staged.slots[slot_index].resident = 1u;
    *route = staged;
    receipt = receipt_base(route, slot_index, slot_index);
    receipt.status = PRT_STATUS_OK;
    return receipt;
}

uint32_t prt_provision_slot_pair_limbs(prt_route_v0 *route,
                                       uint32_t slot_index,
                                       uint32_t new_limit) {
    if (route == NULL || slot_index >= route->route_length ||
        route->slots[slot_index].resident == 0u)
        return PRT_STATUS_INVALID_ARGUMENT;
    if (pww_cell_provision_pair_limbs(&route->slots[slot_index].cell,
                                      new_limit) != PWW_STATUS_OK)
        return PRT_STATUS_INVALID_ARGUMENT;
    return PRT_STATUS_OK;
}

prt_receipt_v0 prt_handoff(prt_route_v0 *route,
                           uint64_t handoff_cause_id,
                           uint32_t destination_index,
                           uint64_t expected_source_local_commit_id,
                           uint64_t expected_destination_local_commit_id,
                           uint32_t source_sequence,
                           uint32_t destination_sequence) {
    prt_route_v0 staged;
    prt_receipt_v0 receipt;
    wc_cause_v0 source_cause;
    wc_cause_v0 destination_cause;
    wc_intent_v0 source_intent;
    wc_intent_v0 destination_intent;
    pww_result_v0 source_result;
    pww_result_v0 destination_result;
    uint32_t source_index;
    uint64_t expected_destination_key;
    uint64_t next_commit_id;
    int source_actor_slot;
    int destination_actor_slot;
    uint16_t health;

    source_index = route == NULL ? 0u : route->active_index;
    receipt = receipt_base(route, source_index, destination_index);
    if (route == NULL || handoff_cause_id == 0u || source_sequence == 0u ||
        destination_sequence == 0u) {
        receipt.status = PRT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (route->bootstrapped == 0u) {
        receipt.status = PRT_STATUS_NOT_BOOTSTRAPPED;
        return receipt;
    }
    if (route->consumed_handoff_cause_id == handoff_cause_id) {
        receipt.status = PRT_STATUS_DUPLICATE_IGNORED;
        return receipt;
    }
    if (destination_index >= route->route_length) {
        receipt.status = PRT_STATUS_SLOT_RANGE;
        return receipt;
    }
    if (destination_index == source_index ||
        (destination_index + 1u != source_index && source_index + 1u != destination_index)) {
        receipt.status = PRT_STATUS_NOT_ADJACENT;
        return receipt;
    }
    if (route->slots[source_index].resident == 0u ||
        route->slots[destination_index].resident == 0u) {
        receipt.status = PRT_STATUS_SLOT_NOT_RESIDENT;
        return receipt;
    }
    if (route->slots[source_index].cell.world.accepted_transition_id !=
            expected_source_local_commit_id ||
        route->slots[destination_index].cell.world.accepted_transition_id !=
            expected_destination_local_commit_id) {
        receipt.status = PRT_STATUS_STALE_VERSION;
        return receipt;
    }
    source_actor_slot = wc_world_resolve(&route->slots[source_index].cell.world,
                                         route->actor_key);
    if (source_actor_slot < 0 ||
        route->slots[source_index].cell.world.kind[source_actor_slot] != WC_KIND_ACTOR) {
        receipt.status = PRT_STATUS_ACTOR_MISSING;
        return receipt;
    }
    health = route->slots[source_index].cell.world.actor_health[source_actor_slot];
    if (route->route_commit_id == UINT64_MAX) {
        receipt.status = PRT_STATUS_COMMIT_EXHAUSTED;
        return receipt;
    }
    next_commit_id = route->route_commit_id + 1u;
    staged = *route;

    source_cause = wc_external_input(source_sequence,
                                     staged.traveler_key,
                                     staged.slots[destination_index].zone_key);
    source_intent = wc_despawn(source_sequence, staged.actor_key);
    source_result = pww_cell_transact(&staged.slots[source_index].cell,
                                      &source_cause, 1u,
                                      &source_intent, 1u);
    if (source_result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&source_result, &receipt);
        return receipt;
    }

    expected_destination_key = staged.slots[destination_index].cell.world.next_object_key;
    destination_cause = wc_external_input(destination_sequence,
                                          staged.traveler_key,
                                          handoff_cause_id);
    destination_intent = wc_spawn_actor(destination_sequence,
                                        health, WC_NON_SPATIAL_SITE);
    destination_result = pww_cell_transact(&staged.slots[destination_index].cell,
                                           &destination_cause, 1u,
                                           &destination_intent, 1u);
    if (destination_result.status != PWW_STATUS_OK) {
        receipt.status = map_destination_failure(&destination_result, &receipt);
        return receipt;
    }
    destination_actor_slot = wc_world_resolve(
        &staged.slots[destination_index].cell.world,
        expected_destination_key);
    if (destination_actor_slot < 0 ||
        staged.slots[destination_index].cell.world.kind[destination_actor_slot] != WC_KIND_ACTOR ||
        staged.slots[destination_index].cell.world.actor_health[destination_actor_slot] != health) {
        receipt.status = PRT_STATUS_DESTINATION_SPAWN_REJECTED;
        receipt.fault_participant = PRT_PARTICIPANT_DESTINATION;
        return receipt;
    }

    staged.active_index = destination_index;
    staged.actor_key = expected_destination_key;
    staged.consumed_handoff_cause_id = handoff_cause_id;
    staged.route_commit_id = next_commit_id;
    *route = staged;

    receipt = receipt_base(route, source_index, destination_index);
    receipt.status = PRT_STATUS_OK;
    receipt.source_primitive = source_result.primitive;
    receipt.destination_primitive = destination_result.primitive;
    return receipt;
}

uint64_t prt_route_fingerprint(const prt_route_v0 *route) {
    uint64_t hash = UINT64_C(1469598103934665603);
    uint32_t i;
    if (route == NULL) return 0u;
    hash_feed(&hash, &route->route_key, sizeof(route->route_key));
    hash_feed(&hash, &route->traveler_key, sizeof(route->traveler_key));
    hash_feed(&hash, &route->actor_key, sizeof(route->actor_key));
    hash_feed(&hash, &route->route_commit_id, sizeof(route->route_commit_id));
    hash_feed(&hash, &route->consumed_handoff_cause_id,
              sizeof(route->consumed_handoff_cause_id));
    hash_feed(&hash, &route->route_length, sizeof(route->route_length));
    hash_feed(&hash, &route->active_index, sizeof(route->active_index));
    hash_feed(&hash, &route->bootstrapped, sizeof(route->bootstrapped));
    for (i = 0u; i < route->route_length; ++i) {
        const prt_route_slot_v0 *slot = &route->slots[i];
        hash_feed(&hash, &slot->zone_key, sizeof(slot->zone_key));
        hash_feed(&hash, &slot->resident, sizeof(slot->resident));
        hash_feed(&hash, &slot->retained, sizeof(slot->retained));
        if (slot->resident != 0u) {
            uint64_t phase = pww_phase_fingerprint(&slot->cell.phase);
            uint64_t world = wc_world_fingerprint(&slot->cell.world);
            hash_feed(&hash, &phase, sizeof(phase));
            hash_feed(&hash, &world, sizeof(world));
        } else {
            hash_feed(&hash, &slot->snapshot_receipt,
                      sizeof(slot->snapshot_receipt));
            hash_feed(&hash, slot->snapshot, (size_t)slot->snapshot_bytes);
        }
    }
    return hash;
}

#include "phase_zone_traversal_v0.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

static void hash_feed(uint64_t *hash, const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t *)data;
    size_t i;
    for (i = 0u; i < size; ++i) {
        *hash ^= bytes[i];
        *hash *= UINT64_C(1099511628211);
    }
}

static pe_source_version_v0 cell_source(uint64_t region_key, const pww_cell_v0 *cell) {
    pe_source_version_v0 source;
    memset(&source, 0, sizeof(source));
    if (cell == NULL) return source;
    source.region_key = region_key;
    source.region_coordination_id = 0u;
    source.zone_key = cell->world.zone_key;
    source.local_commit_id = cell->world.accepted_transition_id;
    source.phase_fingerprint = pww_phase_fingerprint(&cell->phase);
    source.world_fingerprint = wc_world_fingerprint(&cell->world);
    return source;
}

static int source_equal(const pe_source_version_v0 *left,
                        const pe_source_version_v0 *right) {
    return left->region_key == right->region_key &&
           left->region_coordination_id == right->region_coordination_id &&
           left->zone_key == right->zone_key &&
           left->local_commit_id == right->local_commit_id &&
           left->phase_fingerprint == right->phase_fingerprint &&
           left->world_fingerprint == right->world_fingerprint;
}

static const pc_completion_record_v0 *find_completion(
    const pc_completion_stream_v0 *stream,
    uint64_t admission_id) {
    size_t index;
    if (stream == NULL || admission_id == 0u) return NULL;
    for (index = 0u; index < stream->admitted_count; ++index) {
        if (stream->admitted[index].admission_id == admission_id)
            return &stream->admitted[index];
    }
    return NULL;
}

static pt_traversal_receipt_v0 receipt_base(const pt_streamed_traversal_v0 *traversal) {
    pt_traversal_receipt_v0 receipt;
    memset(&receipt, 0, sizeof(receipt));
    if (traversal == NULL) return receipt;
    receipt.traversal_commit_id = traversal->traversal_commit_id;
    receipt.source_local_commit_id = traversal->source.world.accepted_transition_id;
    receipt.destination_local_commit_id = traversal->destination.world.accepted_transition_id;
    receipt.stream_effect_id = traversal->stream_effect_id;
    receipt.completion_admission_id = traversal->consumed_admission_id;
    receipt.traveler_key = traversal->traveler_key;
    receipt.source_actor_key = traversal->source_actor_key;
    receipt.destination_actor_key = traversal->destination_actor_key;
    receipt.source_phase_fingerprint = pww_phase_fingerprint(&traversal->source.phase);
    receipt.source_world_fingerprint = wc_world_fingerprint(&traversal->source.world);
    receipt.destination_phase_fingerprint = pww_phase_fingerprint(&traversal->destination.phase);
    receipt.destination_world_fingerprint = wc_world_fingerprint(&traversal->destination.world);
    return receipt;
}

static uint32_t map_source_failure(const pww_result_v0 *result,
                                   pt_traversal_receipt_v0 *receipt) {
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PT_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PT_STATUS_SOURCE_WORLD_FAILURE;
    return PT_STATUS_SOURCE_PHASE_FAILURE;
}

static uint32_t map_destination_failure(const pww_result_v0 *result,
                                        pt_traversal_receipt_v0 *receipt) {
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PT_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PT_STATUS_DESTINATION_WORLD_FAILURE;
    return PT_STATUS_DESTINATION_PHASE_FAILURE;
}

uint32_t pt_traversal_init(
    pt_streamed_traversal_v0 *traversal,
    uint64_t region_key,
    uint64_t traveler_key,
    uint64_t source_zone_key,
    uint64_t destination_zone_key,
    uint64_t destination_asset_key,
    uint64_t content_version,
    uint64_t variant,
    uint32_t source_pair_limb_limit,
    uint32_t destination_pair_limb_limit) {
    if (traversal == NULL || region_key == 0u || traveler_key == 0u ||
        source_zone_key == 0u || destination_zone_key == 0u ||
        source_zone_key == destination_zone_key || destination_asset_key == 0u)
        return PT_STATUS_INVALID_ARGUMENT;
    memset(traversal, 0, sizeof(*traversal));
    if (pww_cell_init(&traversal->source, source_zone_key,
                      source_pair_limb_limit) != PWW_STATUS_OK)
        return PT_STATUS_INVALID_ARGUMENT;
    if (pww_cell_init(&traversal->destination, destination_zone_key,
                      destination_pair_limb_limit) != PWW_STATUS_OK)
        return PT_STATUS_INVALID_ARGUMENT;
    if (pe_pipeline_init(&traversal->effects, PT_EFFECT_CAPACITY) != PE_STATUS_OK)
        return PT_STATUS_INVALID_ARGUMENT;
    if (pc_completion_stream_init(&traversal->completions,
                                  PT_COMPLETION_CAPACITY,
                                  PT_COMPLETION_CAPACITY) != PC_STATUS_OK)
        return PT_STATUS_INVALID_ARGUMENT;
    traversal->region_key = region_key;
    traversal->traveler_key = traveler_key;
    traversal->destination_asset_key = destination_asset_key;
    traversal->content_version = content_version;
    traversal->variant = variant;
    return PT_STATUS_OK;
}

pt_traversal_receipt_v0 pt_bootstrap_source(
    pt_streamed_traversal_v0 *traversal,
    uint32_t source_sequence,
    uint16_t actor_health) {
    pt_streamed_traversal_v0 staged;
    pt_traversal_receipt_v0 receipt;
    wc_cause_v0 cause;
    wc_intent_v0 intent;
    pww_result_v0 result;

    receipt = receipt_base(traversal);
    if (traversal == NULL || source_sequence == 0u || actor_health == 0u) {
        receipt.status = PT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (traversal->source_bootstrapped != 0u) {
        receipt.status = PT_STATUS_ALREADY_BOOTSTRAPPED;
        return receipt;
    }
    staged = *traversal;
    cause = wc_external_input(source_sequence, staged.traveler_key,
                              staged.destination_asset_key);
    intent = wc_spawn_actor(source_sequence, actor_health, WC_NON_SPATIAL_SITE);
    result = pww_cell_transact(&staged.source, &cause, 1u, &intent, 1u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&result, &receipt);
        return receipt;
    }
    if (result.rejected_requests != 0u || staged.source.world.object_count != 1u ||
        staged.source.world.object_key[0] == 0u) {
        receipt.status = PT_STATUS_SOURCE_ACTOR_MISSING;
        return receipt;
    }
    staged.source_actor_key = staged.source.world.object_key[0];
    staged.source_bootstrapped = 1u;
    *traversal = staged;
    receipt = receipt_base(traversal);
    receipt.status = PT_STATUS_OK;
    receipt.source_primitive = result.primitive;
    return receipt;
}

pt_traversal_receipt_v0 pt_advance_source(
    pt_streamed_traversal_v0 *traversal,
    uint32_t source_sequence,
    uint64_t payload0,
    uint64_t payload1) {
    pt_streamed_traversal_v0 staged;
    pt_traversal_receipt_v0 receipt;
    wc_cause_v0 cause;
    pww_result_v0 result;

    receipt = receipt_base(traversal);
    if (traversal == NULL || source_sequence == 0u) {
        receipt.status = PT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (traversal->source_bootstrapped == 0u || traversal->traversed != 0u) {
        receipt.status = PT_STATUS_NOT_BOOTSTRAPPED;
        return receipt;
    }
    staged = *traversal;
    cause = wc_external_input(source_sequence, payload0, payload1);
    result = pww_cell_transact(&staged.source, &cause, 1u, NULL, 0u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&result, &receipt);
        return receipt;
    }
    *traversal = staged;
    receipt = receipt_base(traversal);
    receipt.status = PT_STATUS_OK;
    receipt.source_primitive = result.primitive;
    return receipt;
}

pt_traversal_receipt_v0 pt_advance_destination(
    pt_streamed_traversal_v0 *traversal,
    uint32_t source_sequence,
    uint64_t payload0,
    uint64_t payload1) {
    pt_streamed_traversal_v0 staged;
    pt_traversal_receipt_v0 receipt;
    wc_cause_v0 cause;
    pww_result_v0 result;

    receipt = receipt_base(traversal);
    if (traversal == NULL || source_sequence == 0u) {
        receipt.status = PT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (traversal->traversed != 0u) {
        receipt.status = PT_STATUS_ALREADY_TRAVERSED;
        return receipt;
    }
    staged = *traversal;
    cause = wc_external_input(source_sequence, payload0, payload1);
    result = pww_cell_transact(&staged.destination, &cause, 1u, NULL, 0u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_destination_failure(&result, &receipt);
        return receipt;
    }
    *traversal = staged;
    receipt = receipt_base(traversal);
    receipt.status = PT_STATUS_OK;
    receipt.destination_primitive = result.primitive;
    return receipt;
}

pt_traversal_receipt_v0 pt_request_destination(
    pt_streamed_traversal_v0 *traversal,
    uint32_t source_sequence) {
    pt_streamed_traversal_v0 staged;
    pt_traversal_receipt_v0 receipt;
    wc_cause_v0 cause;
    pww_result_v0 result;
    pe_source_version_v0 source;
    pe_effect_request_v0 request;
    pe_emit_receipt_v0 emitted;
    pe_consume_receipt_v0 consumed;
    pc_registration_receipt_v0 registered;

    receipt = receipt_base(traversal);
    if (traversal == NULL || source_sequence == 0u) {
        receipt.status = PT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (traversal->source_bootstrapped == 0u) {
        receipt.status = PT_STATUS_NOT_BOOTSTRAPPED;
        return receipt;
    }
    if (traversal->destination_requested != 0u) {
        receipt.status = PT_STATUS_ALREADY_REQUESTED;
        return receipt;
    }
    staged = *traversal;
    cause = wc_external_input(source_sequence, staged.traveler_key,
                              staged.destination_asset_key);
    result = pww_cell_transact(&staged.destination, &cause, 1u, NULL, 0u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_destination_failure(&result, &receipt);
        return receipt;
    }
    source = cell_source(staged.region_key, &staged.destination);
    memset(&request, 0, sizeof(request));
    request.kind = PE_EFFECT_STREAM_REQUEST;
    request.flags = PE_FLAG_CRITICAL;
    request.subject_key = staged.destination_asset_key;
    request.payload0 = staged.content_version;
    request.payload1 = staged.variant;
    emitted = pe_emit_batch(&staged.effects, &source, &request, 1u);
    if (emitted.status != PE_STATUS_OK) {
        receipt.status = PT_STATUS_EFFECT_FAILURE;
        return receipt;
    }
    consumed = pe_consume_next(&staged.effects, &source, 1u);
    if (consumed.status != PE_STATUS_READY) {
        receipt.status = PT_STATUS_EFFECT_FAILURE;
        return receipt;
    }
    registered = pc_register_stream_effect(&staged.completions, &consumed.effect);
    if (registered.status != PC_STATUS_OK) {
        receipt.status = PT_STATUS_EFFECT_FAILURE;
        return receipt;
    }
    staged.stream_effect_id = consumed.effect.effect_id;
    staged.destination_requested = 1u;
    *traversal = staged;
    receipt = receipt_base(traversal);
    receipt.status = PT_STATUS_OK;
    receipt.destination_primitive = result.primitive;
    return receipt;
}

pc_admission_receipt_v0 pt_admit_completion(
    pt_streamed_traversal_v0 *traversal,
    const pc_completion_input_v0 *input) {
    if (traversal == NULL) return pc_admit_completion(NULL, input);
    return pc_admit_completion(&traversal->completions, input);
}

uint32_t pt_provision_destination_pair_limbs(
    pt_streamed_traversal_v0 *traversal,
    uint32_t new_limit) {
    if (traversal == NULL) return PT_STATUS_INVALID_ARGUMENT;
    return pww_cell_provision_pair_limbs(&traversal->destination, new_limit) == PWW_STATUS_OK
               ? PT_STATUS_OK
               : PT_STATUS_INVALID_ARGUMENT;
}

pt_traversal_receipt_v0 pt_traverse(
    pt_streamed_traversal_v0 *traversal,
    uint64_t completion_admission_id,
    uint32_t source_sequence,
    uint32_t destination_sequence) {
    pt_streamed_traversal_v0 staged;
    pt_traversal_receipt_v0 receipt;
    const pc_completion_record_v0 *record;
    pe_source_version_v0 destination_source;
    int source_slot;
    uint16_t health;
    wc_cause_v0 source_cause;
    wc_cause_v0 destination_cause;
    wc_intent_v0 source_intent;
    wc_intent_v0 destination_intent;
    pww_result_v0 source_result;
    pww_result_v0 destination_result;
    uint64_t destination_key;

    receipt = receipt_base(traversal);
    if (traversal == NULL || completion_admission_id == 0u ||
        source_sequence == 0u || destination_sequence == 0u) {
        receipt.status = PT_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (traversal->source_bootstrapped == 0u || traversal->source_actor_key == 0u) {
        receipt.status = PT_STATUS_NOT_BOOTSTRAPPED;
        return receipt;
    }
    if (traversal->destination_requested == 0u) {
        receipt.status = PT_STATUS_NOT_REQUESTED;
        return receipt;
    }
    if (traversal->traversed != 0u) {
        receipt.status = traversal->consumed_admission_id == completion_admission_id
                             ? PT_STATUS_DUPLICATE_IGNORED
                             : PT_STATUS_ALREADY_TRAVERSED;
        return receipt;
    }
    record = find_completion(&traversal->completions, completion_admission_id);
    if (record == NULL) {
        receipt.status = PT_STATUS_COMPLETION_UNKNOWN;
        return receipt;
    }
    if (record->effect_id != traversal->stream_effect_id ||
        record->asset_key != traversal->destination_asset_key ||
        record->content_version != traversal->content_version ||
        record->variant != traversal->variant ||
        record->result != PC_RESULT_READY) {
        receipt.status = PT_STATUS_COMPLETION_NOT_READY;
        return receipt;
    }
    destination_source = cell_source(traversal->region_key, &traversal->destination);
    if (!source_equal(&destination_source, &record->source)) {
        receipt.status = PT_STATUS_COMPLETION_STALE;
        return receipt;
    }
    source_slot = wc_world_resolve(&traversal->source.world,
                                   traversal->source_actor_key);
    if (source_slot < 0 || traversal->source.world.kind[source_slot] != WC_KIND_ACTOR) {
        receipt.status = PT_STATUS_SOURCE_ACTOR_MISSING;
        return receipt;
    }
    health = traversal->source.world.actor_health[source_slot];
    if (traversal->traversal_commit_id == UINT64_MAX) {
        receipt.status = PT_STATUS_COMMIT_EXHAUSTED;
        return receipt;
    }

    staged = *traversal;
    source_cause = wc_external_input(source_sequence, staged.traveler_key,
                                     staged.destination.world.zone_key);
    source_intent = wc_despawn(source_sequence, staged.source_actor_key);
    source_result = pww_cell_transact(&staged.source,
                                      &source_cause, 1u,
                                      &source_intent, 1u);
    if (source_result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&source_result, &receipt);
        return receipt;
    }

    destination_key = staged.destination.world.next_object_key;
    destination_cause = wc_external_input(destination_sequence,
                                          staged.traveler_key,
                                          record->admission_id);
    destination_intent = wc_spawn_actor(destination_sequence, health,
                                        WC_NON_SPATIAL_SITE);
    destination_result = pww_cell_transact(&staged.destination,
                                           &destination_cause, 1u,
                                           &destination_intent, 1u);
    if (destination_result.status != PWW_STATUS_OK) {
        receipt.status = map_destination_failure(&destination_result, &receipt);
        return receipt;
    }
    if (destination_result.rejected_requests != 0u ||
        wc_world_resolve(&staged.destination.world, destination_key) < 0) {
        receipt.status = PT_STATUS_DESTINATION_SPAWN_REJECTED;
        return receipt;
    }
    {
        int destination_slot = wc_world_resolve(&staged.destination.world,
                                                destination_key);
        if (destination_slot < 0 ||
            staged.destination.world.actor_health[destination_slot] != health) {
            receipt.status = PT_STATUS_DESTINATION_SPAWN_REJECTED;
            return receipt;
        }
    }

    staged.destination_actor_key = destination_key;
    staged.consumed_admission_id = completion_admission_id;
    staged.traversal_commit_id += 1u;
    staged.traversed = 1u;
    *traversal = staged;
    receipt = receipt_base(traversal);
    receipt.status = PT_STATUS_OK;
    receipt.source_primitive = source_result.primitive;
    receipt.destination_primitive = destination_result.primitive;
    receipt.completion_admission_id = completion_admission_id;
    return receipt;
}

uint64_t pt_traversal_fingerprint(const pt_streamed_traversal_v0 *traversal) {
    uint64_t hash = UINT64_C(14695981039346656037);
    if (traversal == NULL) return 0u;
    hash_feed(&hash, &traversal->region_key, sizeof(traversal->region_key));
    hash_feed(&hash, &traversal->traveler_key, sizeof(traversal->traveler_key));
    hash_feed(&hash, &traversal->source, sizeof(traversal->source));
    hash_feed(&hash, &traversal->destination, sizeof(traversal->destination));
    hash_feed(&hash, &traversal->effects, sizeof(traversal->effects));
    hash_feed(&hash, &traversal->completions, sizeof(traversal->completions));
    hash_feed(&hash, &traversal->destination_asset_key,
              sizeof(traversal->destination_asset_key));
    hash_feed(&hash, &traversal->content_version,
              sizeof(traversal->content_version));
    hash_feed(&hash, &traversal->variant, sizeof(traversal->variant));
    hash_feed(&hash, &traversal->stream_effect_id,
              sizeof(traversal->stream_effect_id));
    hash_feed(&hash, &traversal->consumed_admission_id,
              sizeof(traversal->consumed_admission_id));
    hash_feed(&hash, &traversal->traversal_commit_id,
              sizeof(traversal->traversal_commit_id));
    hash_feed(&hash, &traversal->source_actor_key,
              sizeof(traversal->source_actor_key));
    hash_feed(&hash, &traversal->destination_actor_key,
              sizeof(traversal->destination_actor_key));
    hash_feed(&hash, &traversal->source_bootstrapped,
              sizeof(traversal->source_bootstrapped));
    hash_feed(&hash, &traversal->destination_requested,
              sizeof(traversal->destination_requested));
    hash_feed(&hash, &traversal->traversed,
              sizeof(traversal->traversed));
    return hash;
}

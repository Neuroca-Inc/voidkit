#include "phase_streamed_slice_v0.h"

#include <stddef.h>
#include <string.h>

static pe_source_version_v0 current_source(const ps_streamed_zone_slice_v0 *slice) {
    pe_source_version_v0 source;
    memset(&source, 0, sizeof(source));
    if (slice == NULL) return source;
    source.region_key = slice->region_key;
    source.region_coordination_id = 0u;
    source.zone_key = slice->cell.world.zone_key;
    source.local_commit_id = slice->cell.world.accepted_transition_id;
    source.phase_fingerprint = pww_phase_fingerprint(&slice->cell.phase);
    source.world_fingerprint = wc_world_fingerprint(&slice->cell.world);
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

static ps_slice_receipt_v0 receipt_base(const ps_streamed_zone_slice_v0 *slice) {
    ps_slice_receipt_v0 receipt;
    pe_source_version_v0 source = current_source(slice);
    memset(&receipt, 0, sizeof(receipt));
    if (slice != NULL) {
        receipt.local_commit_id = slice->cell.world.accepted_transition_id;
        receipt.stream_effect_id = slice->stream_effect_id;
        receipt.completion_admission_id = slice->consumed_admission_id;
    }
    receipt.phase_fingerprint = source.phase_fingerprint;
    receipt.world_fingerprint = source.world_fingerprint;
    return receipt;
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

uint32_t ps_slice_init(
    ps_streamed_zone_slice_v0 *slice,
    uint64_t region_key,
    uint64_t zone_key,
    uint64_t asset_key,
    uint64_t content_version,
    uint64_t variant,
    uint32_t pair_limb_limit) {
    if (slice == NULL || region_key == 0u || zone_key == 0u || asset_key == 0u)
        return PS_STATUS_INVALID_ARGUMENT;
    memset(slice, 0, sizeof(*slice));
    if (pww_cell_init(&slice->cell, zone_key, pair_limb_limit) != PWW_STATUS_OK)
        return PS_STATUS_INVALID_ARGUMENT;
    if (pe_pipeline_init(&slice->effects, PS_SLICE_EFFECT_CAPACITY) != PE_STATUS_OK)
        return PS_STATUS_INVALID_ARGUMENT;
    if (pc_completion_stream_init(&slice->completions,
                                  PS_SLICE_COMPLETION_CAPACITY,
                                  PS_SLICE_COMPLETION_CAPACITY) != PC_STATUS_OK)
        return PS_STATUS_INVALID_ARGUMENT;
    slice->region_key = region_key;
    slice->asset_key = asset_key;
    slice->content_version = content_version;
    slice->variant = variant;
    return PS_STATUS_OK;
}

ps_slice_receipt_v0 ps_slice_bootstrap_actor(
    ps_streamed_zone_slice_v0 *slice,
    uint32_t source_sequence,
    uint16_t actor_health) {
    ps_streamed_zone_slice_v0 staged;
    ps_slice_receipt_v0 receipt;
    wc_cause_v0 cause;
    wc_intent_v0 intent;
    pww_result_v0 phase_result;
    pe_source_version_v0 source;
    pe_effect_request_v0 request;
    pe_emit_receipt_v0 emit;
    pe_consume_receipt_v0 consumed;
    pc_registration_receipt_v0 registered;

    if (slice == NULL || source_sequence == 0u || actor_health == 0u) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (slice->actor_spawned != 0u) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_ALREADY_BOOTSTRAPPED;
        return receipt;
    }

    staged = *slice;
    cause = wc_external_input(source_sequence, staged.asset_key,
                              staged.content_version);
    intent = wc_spawn_actor(source_sequence, actor_health, WC_NON_SPATIAL_SITE);
    phase_result = pww_cell_transact(&staged.cell, &cause, 1u, &intent, 1u);
    if (phase_result.status != PWW_STATUS_OK) {
        receipt = receipt_base(slice);
        if (phase_result.status == PWW_STATUS_PROVISION_REQUIRED) {
            receipt.status = PS_STATUS_PROVISION_REQUIRED;
            receipt.required_pair_limbs = phase_result.required_pair_limbs;
        } else if (phase_result.status == PWW_STATUS_WORLD_FAULT) {
            receipt.status = PS_STATUS_WORLD_FAILURE;
        } else {
            receipt.status = PS_STATUS_PHASE_FAILURE;
        }
        return receipt;
    }

    if (staged.cell.world.object_count != 1u ||
        staged.cell.world.object_key[0] == 0u) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_WORLD_FAILURE;
        return receipt;
    }
    staged.actor_key = staged.cell.world.object_key[0];
    staged.actor_spawned = 1u;

    source = current_source(&staged);
    memset(&request, 0, sizeof(request));
    request.kind = PE_EFFECT_STREAM_REQUEST;
    request.flags = PE_FLAG_CRITICAL;
    request.subject_key = staged.asset_key;
    request.payload0 = staged.content_version;
    request.payload1 = staged.variant;
    emit = pe_emit_batch(&staged.effects, &source, &request, 1u);
    if (emit.status != PE_STATUS_OK) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_EFFECT_FAILURE;
        return receipt;
    }
    consumed = pe_consume_next(&staged.effects, &source, 1u);
    if (consumed.status != PE_STATUS_READY) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_EFFECT_FAILURE;
        return receipt;
    }
    registered = pc_register_stream_effect(&staged.completions, &consumed.effect);
    if (registered.status != PC_STATUS_OK) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_EFFECT_FAILURE;
        return receipt;
    }
    staged.stream_effect_id = consumed.effect.effect_id;
    staged.request_dispatched = 1u;

    *slice = staged;
    receipt = receipt_base(slice);
    receipt.status = PS_STATUS_OK;
    receipt.primitive = phase_result.primitive;
    return receipt;
}

pc_admission_receipt_v0 ps_slice_admit_completion(
    ps_streamed_zone_slice_v0 *slice,
    const pc_completion_input_v0 *input) {
    if (slice == NULL)
        return pc_admit_completion(NULL, input);
    return pc_admit_completion(&slice->completions, input);
}

ps_slice_receipt_v0 ps_slice_advance_local_cause(
    ps_streamed_zone_slice_v0 *slice,
    uint32_t source_sequence,
    uint64_t payload0,
    uint64_t payload1) {
    ps_streamed_zone_slice_v0 staged;
    ps_slice_receipt_v0 receipt;
    wc_cause_v0 cause;
    pww_result_v0 phase_result;

    if (slice == NULL || source_sequence == 0u) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (slice->actor_spawned == 0u) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_NOT_BOOTSTRAPPED;
        return receipt;
    }
    staged = *slice;
    cause = wc_external_input(source_sequence, payload0, payload1);
    phase_result = pww_cell_transact(&staged.cell, &cause, 1u, NULL, 0u);
    if (phase_result.status != PWW_STATUS_OK) {
        receipt = receipt_base(slice);
        if (phase_result.status == PWW_STATUS_PROVISION_REQUIRED) {
            receipt.status = PS_STATUS_PROVISION_REQUIRED;
            receipt.required_pair_limbs = phase_result.required_pair_limbs;
        } else if (phase_result.status == PWW_STATUS_WORLD_FAULT) {
            receipt.status = PS_STATUS_WORLD_FAILURE;
        } else {
            receipt.status = PS_STATUS_PHASE_FAILURE;
        }
        return receipt;
    }
    *slice = staged;
    receipt = receipt_base(slice);
    receipt.status = PS_STATUS_OK;
    receipt.primitive = phase_result.primitive;
    return receipt;
}

ps_slice_receipt_v0 ps_slice_activate_zone(
    ps_streamed_zone_slice_v0 *slice,
    uint64_t completion_admission_id,
    uint32_t source_sequence,
    int32_t target_x,
    int32_t target_y,
    int32_t target_z) {
    const pc_completion_record_v0 *record;
    pe_source_version_v0 source;
    ps_streamed_zone_slice_v0 staged;
    ps_slice_receipt_v0 receipt;
    wc_cause_v0 cause;
    wc_intent_v0 intent;
    pww_result_v0 phase_result;

    if (slice == NULL || completion_admission_id == 0u || source_sequence == 0u) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (slice->actor_spawned == 0u || slice->request_dispatched == 0u) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_NOT_BOOTSTRAPPED;
        return receipt;
    }
    if (slice->zone_active != 0u) {
        receipt = receipt_base(slice);
        receipt.status = slice->consumed_admission_id == completion_admission_id
                             ? PS_STATUS_DUPLICATE_IGNORED
                             : PS_STATUS_ALREADY_ACTIVE;
        return receipt;
    }

    record = find_completion(&slice->completions, completion_admission_id);
    if (record == NULL) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_COMPLETION_UNKNOWN;
        return receipt;
    }
    if (record->effect_id != slice->stream_effect_id ||
        record->asset_key != slice->asset_key ||
        record->content_version != slice->content_version ||
        record->variant != slice->variant ||
        record->result != PC_RESULT_READY) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_COMPLETION_NOT_READY;
        return receipt;
    }

    source = current_source(slice);
    if (!source_equal(&source, &record->source)) {
        receipt = receipt_base(slice);
        receipt.status = PS_STATUS_COMPLETION_STALE;
        return receipt;
    }

    staged = *slice;
    cause = wc_external_input(source_sequence, record->asset_key,
                              record->admission_id);
    intent = wc_replace_kinematics(source_sequence, staged.actor_key,
                                   target_x, target_y, target_z,
                                   0, 0, 0);
    phase_result = pww_cell_transact(&staged.cell, &cause, 1u, &intent, 1u);
    if (phase_result.status != PWW_STATUS_OK) {
        receipt = receipt_base(slice);
        if (phase_result.status == PWW_STATUS_PROVISION_REQUIRED) {
            receipt.status = PS_STATUS_PROVISION_REQUIRED;
            receipt.required_pair_limbs = phase_result.required_pair_limbs;
        } else if (phase_result.status == PWW_STATUS_WORLD_FAULT) {
            receipt.status = PS_STATUS_WORLD_FAILURE;
        } else {
            receipt.status = PS_STATUS_PHASE_FAILURE;
        }
        return receipt;
    }

    staged.zone_active = 1u;
    staged.consumed_admission_id = completion_admission_id;
    *slice = staged;
    receipt = receipt_base(slice);
    receipt.status = PS_STATUS_OK;
    receipt.primitive = phase_result.primitive;
    return receipt;
}

uint64_t ps_slice_fingerprint(const ps_streamed_zone_slice_v0 *slice) {
    const uint8_t *bytes;
    size_t index;
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    if (slice == NULL) return 0u;
    bytes = (const uint8_t *)slice;
    for (index = 0u; index < sizeof(*slice); ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

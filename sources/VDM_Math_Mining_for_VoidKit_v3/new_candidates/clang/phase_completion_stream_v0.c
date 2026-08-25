#include "phase_completion_stream_v0.h"

#include <limits.h>
#include <string.h>

static int source_equal(
    const pe_source_version_v0 *left,
    const pe_source_version_v0 *right) {
    return left->region_key == right->region_key &&
           left->region_coordination_id == right->region_coordination_id &&
           left->zone_key == right->zone_key &&
           left->local_commit_id == right->local_commit_id &&
           left->phase_fingerprint == right->phase_fingerprint &&
           left->world_fingerprint == right->world_fingerprint;
}

static int result_valid(uint32_t result) {
    return result == PC_RESULT_READY ||
           result == PC_RESULT_NOT_FOUND ||
           result == PC_RESULT_FAILED;
}

static int pending_matches_effect(
    const pc_pending_request_v0 *pending,
    const pe_effect_record_v0 *effect) {
    return pending->effect_id == effect->effect_id &&
           source_equal(&pending->source, &effect->source) &&
           pending->asset_key == effect->request.subject_key &&
           pending->content_version == effect->request.payload0 &&
           pending->variant == effect->request.payload1;
}

static int input_matches_pending(
    const pc_completion_input_v0 *input,
    const pc_pending_request_v0 *pending) {
    return input->effect_id == pending->effect_id &&
           input->asset_key == pending->asset_key &&
           input->content_version == pending->content_version &&
           input->variant == pending->variant;
}

static int input_matches_record(
    const pc_completion_input_v0 *input,
    const pc_completion_record_v0 *record) {
    return input->effect_id == record->effect_id &&
           input->asset_key == record->asset_key &&
           input->content_version == record->content_version &&
           input->variant == record->variant &&
           input->result == record->result &&
           input->content_hash == record->content_hash &&
           input->byte_length == record->byte_length;
}

static int record_matches_pending(
    const pc_completion_record_v0 *record,
    const pc_pending_request_v0 *pending) {
    return record->effect_id == pending->effect_id &&
           source_equal(&record->source, &pending->source) &&
           record->asset_key == pending->asset_key &&
           record->content_version == pending->content_version &&
           record->variant == pending->variant;
}

static size_t find_pending(
    const pc_completion_stream_v0 *stream,
    uint64_t effect_id) {
    size_t index;
    for (index = 0u; index < stream->pending_count; ++index) {
        if (stream->pending[index].effect_id == effect_id) return index;
    }
    return SIZE_MAX;
}

static size_t find_admitted(
    const pc_completion_stream_v0 *stream,
    uint64_t effect_id) {
    size_t index;
    for (index = 0u; index < stream->admitted_count; ++index) {
        if (stream->admitted[index].effect_id == effect_id) return index;
    }
    return SIZE_MAX;
}

static void remove_pending(pc_completion_stream_v0 *stream, size_t index) {
    size_t cursor = index + 1u;
    while (cursor < stream->pending_count) {
        stream->pending[cursor - 1u] = stream->pending[cursor];
        cursor += 1u;
    }
    stream->pending_count -= 1u;
    memset(&stream->pending[stream->pending_count], 0, sizeof(stream->pending[0]));
}

static pc_registration_receipt_v0 registration_receipt(
    uint32_t status,
    const pc_completion_stream_v0 *stream,
    uint64_t effect_id) {
    pc_registration_receipt_v0 receipt;
    memset(&receipt, 0, sizeof(receipt));
    receipt.status = status;
    receipt.effect_id = effect_id;
    if (stream != NULL) receipt.pending_count = stream->pending_count;
    return receipt;
}

static pc_admission_receipt_v0 admission_receipt(
    uint32_t status,
    const pc_completion_stream_v0 *stream,
    const pc_completion_record_v0 *record) {
    pc_admission_receipt_v0 receipt;
    memset(&receipt, 0, sizeof(receipt));
    receipt.status = status;
    if (stream != NULL) {
        receipt.pending_count = stream->pending_count;
        receipt.admitted_count = stream->admitted_count;
    }
    if (record != NULL) receipt.record = *record;
    return receipt;
}

uint32_t pc_completion_stream_init(
    pc_completion_stream_v0 *stream,
    uint32_t pending_capacity,
    uint32_t admitted_capacity) {
    if (stream == NULL || pending_capacity == 0u || admitted_capacity == 0u ||
        pending_capacity > PC_MAX_PENDING_REQUESTS ||
        admitted_capacity > PC_MAX_ADMITTED_COMPLETIONS) {
        return PC_STATUS_INVALID_ARGUMENT;
    }
    memset(stream, 0, sizeof(*stream));
    stream->next_admission_id = 1u;
    stream->pending_capacity = pending_capacity;
    stream->admitted_capacity = admitted_capacity;
    return PC_STATUS_OK;
}

pc_registration_receipt_v0 pc_register_stream_effect(
    pc_completion_stream_v0 *stream,
    const pe_effect_record_v0 *effect) {
    size_t index;
    pc_pending_request_v0 pending;

    if (stream == NULL || effect == NULL || effect->effect_id == 0u ||
        effect->request.kind != PE_EFFECT_STREAM_REQUEST ||
        effect->source.region_key == 0u || effect->source.zone_key == 0u) {
        return registration_receipt(PC_STATUS_INVALID_ARGUMENT, stream, 0u);
    }

    index = find_pending(stream, effect->effect_id);
    if (index != SIZE_MAX) {
        return registration_receipt(
            pending_matches_effect(&stream->pending[index], effect)
                ? PC_STATUS_DUPLICATE_IGNORED
                : PC_STATUS_MISMATCH,
            stream,
            effect->effect_id);
    }

    index = find_admitted(stream, effect->effect_id);
    if (index != SIZE_MAX) {
        const pc_completion_record_v0 *record = &stream->admitted[index];
        int same = source_equal(&record->source, &effect->source) &&
                   record->asset_key == effect->request.subject_key &&
                   record->content_version == effect->request.payload0 &&
                   record->variant == effect->request.payload1;
        return registration_receipt(
            same ? PC_STATUS_DUPLICATE_IGNORED : PC_STATUS_MISMATCH,
            stream,
            effect->effect_id);
    }

    if (stream->pending_count >= stream->pending_capacity) {
        return registration_receipt(PC_STATUS_FULL, stream, effect->effect_id);
    }

    memset(&pending, 0, sizeof(pending));
    pending.effect_id = effect->effect_id;
    pending.source = effect->source;
    pending.asset_key = effect->request.subject_key;
    pending.content_version = effect->request.payload0;
    pending.variant = effect->request.payload1;
    stream->pending[stream->pending_count] = pending;
    stream->pending_count += 1u;
    return registration_receipt(PC_STATUS_OK, stream, effect->effect_id);
}

pc_admission_receipt_v0 pc_admit_completion(
    pc_completion_stream_v0 *stream,
    const pc_completion_input_v0 *input) {
    size_t index;
    pc_completion_record_v0 record;

    if (stream == NULL || input == NULL || input->effect_id == 0u ||
        input->asset_key == 0u || !result_valid(input->result)) {
        return admission_receipt(PC_STATUS_INVALID_ARGUMENT, stream, NULL);
    }

    index = find_admitted(stream, input->effect_id);
    if (index != SIZE_MAX) {
        const pc_completion_record_v0 *existing = &stream->admitted[index];
        return admission_receipt(
            input_matches_record(input, existing)
                ? PC_STATUS_DUPLICATE_IGNORED
                : PC_STATUS_MISMATCH,
            stream,
            existing);
    }

    index = find_pending(stream, input->effect_id);
    if (index == SIZE_MAX) {
        return admission_receipt(PC_STATUS_UNKNOWN_REQUEST, stream, NULL);
    }
    if (!input_matches_pending(input, &stream->pending[index])) {
        return admission_receipt(PC_STATUS_MISMATCH, stream, NULL);
    }
    if (stream->admitted_count >= stream->admitted_capacity) {
        return admission_receipt(PC_STATUS_FULL, stream, NULL);
    }
    if (stream->next_admission_id == 0u || stream->next_admission_id == UINT64_MAX) {
        return admission_receipt(PC_STATUS_ADMISSION_ID_EXHAUSTED, stream, NULL);
    }

    memset(&record, 0, sizeof(record));
    record.admission_id = stream->next_admission_id;
    record.effect_id = input->effect_id;
    record.source = stream->pending[index].source;
    record.asset_key = input->asset_key;
    record.content_version = input->content_version;
    record.variant = input->variant;
    record.result = input->result;
    record.content_hash = input->content_hash;
    record.byte_length = input->byte_length;

    stream->admitted[stream->admitted_count] = record;
    stream->admitted_count += 1u;
    stream->next_admission_id += 1u;
    remove_pending(stream, index);
    return admission_receipt(PC_STATUS_ACCEPTED, stream, &record);
}

pc_admission_receipt_v0 pc_replay_completion(
    pc_completion_stream_v0 *stream,
    const pc_completion_record_v0 *record) {
    size_t index;

    if (stream == NULL || record == NULL || record->admission_id == 0u ||
        record->effect_id == 0u || record->asset_key == 0u ||
        !result_valid(record->result)) {
        return admission_receipt(PC_STATUS_INVALID_ARGUMENT, stream, NULL);
    }
    if (record->admission_id != stream->next_admission_id) {
        return admission_receipt(PC_STATUS_MISMATCH, stream, NULL);
    }
    if (find_admitted(stream, record->effect_id) != SIZE_MAX) {
        return admission_receipt(PC_STATUS_MISMATCH, stream, NULL);
    }

    index = find_pending(stream, record->effect_id);
    if (index == SIZE_MAX) {
        return admission_receipt(PC_STATUS_UNKNOWN_REQUEST, stream, NULL);
    }
    if (!record_matches_pending(record, &stream->pending[index])) {
        return admission_receipt(PC_STATUS_MISMATCH, stream, NULL);
    }
    if (stream->admitted_count >= stream->admitted_capacity) {
        return admission_receipt(PC_STATUS_FULL, stream, NULL);
    }
    if (stream->next_admission_id == UINT64_MAX) {
        return admission_receipt(PC_STATUS_ADMISSION_ID_EXHAUSTED, stream, NULL);
    }

    stream->admitted[stream->admitted_count] = *record;
    stream->admitted_count += 1u;
    stream->next_admission_id += 1u;
    remove_pending(stream, index);
    return admission_receipt(PC_STATUS_REPLAYED, stream, record);
}

static uint64_t mix_u64(uint64_t hash, uint64_t value) {
    unsigned int byte;
    for (byte = 0u; byte < 8u; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t pc_completion_stream_fingerprint(const pc_completion_stream_v0 *stream) {
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t index;
    if (stream == NULL) return 0u;
    hash = mix_u64(hash, stream->next_admission_id);
    hash = mix_u64(hash, stream->pending_count);
    hash = mix_u64(hash, stream->admitted_count);
    for (index = 0u; index < stream->pending_count; ++index) {
        const pc_pending_request_v0 *pending = &stream->pending[index];
        hash = mix_u64(hash, pending->effect_id);
        hash = mix_u64(hash, pending->source.region_key);
        hash = mix_u64(hash, pending->source.region_coordination_id);
        hash = mix_u64(hash, pending->source.zone_key);
        hash = mix_u64(hash, pending->source.local_commit_id);
        hash = mix_u64(hash, pending->source.phase_fingerprint);
        hash = mix_u64(hash, pending->source.world_fingerprint);
        hash = mix_u64(hash, pending->asset_key);
        hash = mix_u64(hash, pending->content_version);
        hash = mix_u64(hash, pending->variant);
    }
    for (index = 0u; index < stream->admitted_count; ++index) {
        const pc_completion_record_v0 *record = &stream->admitted[index];
        hash = mix_u64(hash, record->admission_id);
        hash = mix_u64(hash, record->effect_id);
        hash = mix_u64(hash, record->source.region_key);
        hash = mix_u64(hash, record->source.region_coordination_id);
        hash = mix_u64(hash, record->source.zone_key);
        hash = mix_u64(hash, record->source.local_commit_id);
        hash = mix_u64(hash, record->source.phase_fingerprint);
        hash = mix_u64(hash, record->source.world_fingerprint);
        hash = mix_u64(hash, record->asset_key);
        hash = mix_u64(hash, record->content_version);
        hash = mix_u64(hash, record->variant);
        hash = mix_u64(hash, record->result);
        hash = mix_u64(hash, record->content_hash);
        hash = mix_u64(hash, record->byte_length);
    }
    return hash;
}

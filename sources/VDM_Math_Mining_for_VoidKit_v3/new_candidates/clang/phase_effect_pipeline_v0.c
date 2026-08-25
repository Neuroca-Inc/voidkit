#include "phase_effect_pipeline_v0.h"

#include <limits.h>
#include <string.h>

static int source_valid(const pe_source_version_v0 *source) {
    return source != NULL && source->region_key != 0u && source->zone_key != 0u;
}

static int request_valid(const pe_effect_request_v0 *request) {
    if (request == NULL) return 0;
    return request->kind == PE_EFFECT_RENDER_COMMANDS ||
           request->kind == PE_EFFECT_AUDIO_COMMANDS ||
           request->kind == PE_EFFECT_STREAM_REQUEST;
}

static int request_less(
    const pe_effect_request_v0 *left,
    const pe_effect_request_v0 *right) {
    if (left->kind != right->kind) return left->kind < right->kind;
    if (left->channel != right->channel) return left->channel < right->channel;
    if (left->subject_key != right->subject_key) return left->subject_key < right->subject_key;
    if (left->flags != right->flags) return left->flags < right->flags;
    if (left->payload0 != right->payload0) return left->payload0 < right->payload0;
    return left->payload1 < right->payload1;
}

static void sort_requests(pe_effect_request_v0 *requests, size_t count) {
    size_t index;
    for (index = 1u; index < count; ++index) {
        pe_effect_request_v0 value = requests[index];
        size_t cursor = index;
        while (cursor != 0u && request_less(&value, &requests[cursor - 1u])) {
            requests[cursor] = requests[cursor - 1u];
            cursor -= 1u;
        }
        requests[cursor] = value;
    }
}

static const pe_source_version_v0 *find_current_source(
    const pe_effect_record_v0 *record,
    const pe_source_version_v0 *current_versions,
    size_t current_version_count) {
    size_t index;
    for (index = 0u; index < current_version_count; ++index) {
        if (current_versions[index].region_key == record->source.region_key &&
            current_versions[index].zone_key == record->source.zone_key) {
            return &current_versions[index];
        }
    }
    return NULL;
}

static uint32_t classify_source(
    const pe_effect_record_v0 *record,
    const pe_source_version_v0 *current) {
    if (current == NULL) return PE_STATUS_SOURCE_NOT_FOUND;
    if (current->region_coordination_id < record->source.region_coordination_id ||
        current->local_commit_id < record->source.local_commit_id) {
        return PE_STATUS_NOT_READY;
    }
    if (current->region_coordination_id == record->source.region_coordination_id &&
        current->local_commit_id == record->source.local_commit_id &&
        current->phase_fingerprint == record->source.phase_fingerprint &&
        current->world_fingerprint == record->source.world_fingerprint) {
        return PE_STATUS_READY;
    }
    return PE_STATUS_STALE_DISCARDED;
}

static void remove_first(pe_pipeline_v0 *pipeline) {
    if (pipeline->pending_count > 1u) {
        memmove(
            &pipeline->pending[0],
            &pipeline->pending[1],
            (size_t)(pipeline->pending_count - 1u) * sizeof(pipeline->pending[0]));
    }
    pipeline->pending_count -= 1u;
    memset(&pipeline->pending[pipeline->pending_count], 0, sizeof(pipeline->pending[0]));
}

uint32_t pe_pipeline_init(pe_pipeline_v0 *pipeline, uint32_t capacity) {
    if (pipeline == NULL || capacity == 0u || capacity > PE_MAX_PENDING_EFFECTS) {
        return PE_STATUS_INVALID_ARGUMENT;
    }
    memset(pipeline, 0, sizeof(*pipeline));
    pipeline->next_effect_id = 1u;
    pipeline->capacity = capacity;
    return PE_STATUS_OK;
}

pe_emit_receipt_v0 pe_emit_batch(
    pe_pipeline_v0 *pipeline,
    const pe_source_version_v0 *source,
    const pe_effect_request_v0 *requests,
    size_t request_count) {
    pe_emit_receipt_v0 receipt;
    pe_pipeline_v0 staged;
    pe_effect_request_v0 sorted[PE_MAX_EMIT_BATCH];
    size_t index;

    memset(&receipt, 0, sizeof(receipt));
    if (pipeline == NULL || !source_valid(source)) {
        receipt.status = PE_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    receipt.source = *source;
    if (request_count == 0u) {
        receipt.status = PE_STATUS_NO_WORK;
        return receipt;
    }
    if (requests == NULL || request_count > PE_MAX_EMIT_BATCH) {
        receipt.status = PE_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    for (index = 0u; index < request_count; ++index) {
        if (!request_valid(&requests[index])) {
            receipt.status = PE_STATUS_INVALID_ARGUMENT;
            return receipt;
        }
        sorted[index] = requests[index];
    }
    if ((size_t)pipeline->pending_count + request_count > pipeline->capacity) {
        receipt.status = PE_STATUS_FULL;
        return receipt;
    }
    if (pipeline->next_effect_id == 0u ||
        request_count > (size_t)(UINT64_MAX - pipeline->next_effect_id)) {
        receipt.status = PE_STATUS_EFFECT_ID_EXHAUSTED;
        return receipt;
    }

    sort_requests(sorted, request_count);
    staged = *pipeline;
    receipt.first_effect_id = staged.next_effect_id;
    for (index = 0u; index < request_count; ++index) {
        pe_effect_record_v0 *record = &staged.pending[staged.pending_count];
        memset(record, 0, sizeof(*record));
        record->effect_id = staged.next_effect_id;
        record->source = *source;
        record->request = sorted[index];
        staged.next_effect_id += 1u;
        staged.pending_count += 1u;
    }
    receipt.last_effect_id = staged.next_effect_id - 1u;
    receipt.published_count = (uint32_t)request_count;
    receipt.status = PE_STATUS_OK;
    *pipeline = staged;
    return receipt;
}

pe_consume_receipt_v0 pe_consume_next(
    pe_pipeline_v0 *pipeline,
    const pe_source_version_v0 *current_versions,
    size_t current_version_count) {
    pe_consume_receipt_v0 receipt;
    const pe_source_version_v0 *current;
    uint32_t classification;

    memset(&receipt, 0, sizeof(receipt));
    if (pipeline == NULL || (current_version_count != 0u && current_versions == NULL)) {
        receipt.status = PE_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (pipeline->pending_count == 0u) {
        receipt.status = PE_STATUS_NO_WORK;
        return receipt;
    }

    receipt.effect = pipeline->pending[0];
    current = find_current_source(&receipt.effect, current_versions, current_version_count);
    classification = classify_source(&receipt.effect, current);
    receipt.status = classification;
    if (classification == PE_STATUS_READY || classification == PE_STATUS_STALE_DISCARDED) {
        remove_first(pipeline);
    }
    receipt.remaining_count = pipeline->pending_count;
    return receipt;
}

uint64_t pe_pipeline_fingerprint(const pe_pipeline_v0 *pipeline) {
    uint64_t hash = UINT64_C(1469598103934665603);
    const uint8_t *bytes;
    size_t size;
    size_t index;
    if (pipeline == NULL) return 0u;
    bytes = (const uint8_t *)pipeline;
    size = offsetof(pe_pipeline_v0, pending) +
           (size_t)pipeline->pending_count * sizeof(pipeline->pending[0]);
    for (index = 0u; index < size; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

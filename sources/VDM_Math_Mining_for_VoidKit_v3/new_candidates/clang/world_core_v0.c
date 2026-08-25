#include "world_core_v0.h"

#include <limits.h>
#include <string.h>

_Static_assert(sizeof(wc_intent_v0) == 64u, "mutation intent must be 64 bytes");

static uint64_t pack_i32_pair(int32_t low, int32_t high) {
    return (uint64_t)(uint32_t)low | ((uint64_t)(uint32_t)high << 32u);
}

static int32_t unpack_low(uint64_t value) { return (int32_t)(uint32_t)value; }
static int32_t unpack_high(uint64_t value) { return (int32_t)(uint32_t)(value >> 32u); }

void wc_world_init(wc_world_v0 *world, uint64_t zone_key) {
    size_t i;
    memset(world, 0, sizeof(*world));
    world->next_object_key = 1u;
    world->zone_key = zone_key;
    for (i = 0; i < WC_CP7_OBJECT_CAPACITY; ++i) {
        world->generation[i] = 1u;
        world->surface_site[i] = WC_NON_SPATIAL_SITE;
        world->next_in_site[i] = UINT32_MAX;
    }
}

wc_cause_v0 wc_external_input(uint32_t source_sequence, uint64_t payload0, uint64_t payload1) {
    wc_cause_v0 cause;
    memset(&cause, 0, sizeof(cause));
    cause.kind = WC_CAUSE_EXTERNAL_INPUT;
    cause.source_sequence = source_sequence;
    cause.payload0 = payload0;
    cause.payload1 = payload1;
    return cause;
}

wc_intent_v0 wc_spawn_actor(uint32_t source_sequence, uint16_t health, uint16_t site) {
    wc_intent_v0 intent;
    memset(&intent, 0, sizeof(intent));
    intent.kind = WC_INTENT_SPAWN_OBJECT;
    intent.reducer = WC_REDUCER_LIFECYCLE;
    intent.source_sequence = source_sequence;
    intent.aux = (uint64_t)WC_KIND_ACTOR | ((uint64_t)site << 16u);
    intent.value0 = health;
    return intent;
}

wc_intent_v0 wc_spawn_item(uint32_t source_sequence, uint16_t item_type, uint16_t stack, uint16_t site) {
    wc_intent_v0 intent;
    memset(&intent, 0, sizeof(intent));
    intent.kind = WC_INTENT_SPAWN_OBJECT;
    intent.reducer = WC_REDUCER_LIFECYCLE;
    intent.source_sequence = source_sequence;
    intent.aux = (uint64_t)WC_KIND_ITEM | ((uint64_t)site << 16u);
    intent.value0 = (uint64_t)item_type | ((uint64_t)stack << 16u);
    return intent;
}

wc_intent_v0 wc_despawn(uint32_t source_sequence, uint64_t target) {
    wc_intent_v0 intent;
    memset(&intent, 0, sizeof(intent));
    intent.kind = WC_INTENT_DESPAWN_OBJECT;
    intent.reducer = WC_REDUCER_LIFECYCLE;
    intent.source_sequence = source_sequence;
    intent.target = target;
    return intent;
}

wc_intent_v0 wc_replace_kinematics(uint32_t source_sequence, uint64_t target,
                                    int32_t px, int32_t py, int32_t pz,
                                    int32_t vx, int32_t vy, int32_t vz) {
    wc_intent_v0 intent;
    memset(&intent, 0, sizeof(intent));
    intent.kind = WC_INTENT_REPLACE_KINEMATICS;
    intent.reducer = WC_REDUCER_EXCLUSIVE;
    intent.source_sequence = source_sequence;
    intent.target = target;
    intent.value0 = pack_i32_pair(px, py);
    intent.value1 = pack_i32_pair(pz, vx);
    intent.aux = pack_i32_pair(vy, vz);
    return intent;
}

wc_intent_v0 wc_apply_health_delta(uint32_t source_sequence, uint64_t target, int64_t delta) {
    wc_intent_v0 intent;
    memset(&intent, 0, sizeof(intent));
    intent.kind = WC_INTENT_APPLY_HEALTH_DELTA;
    intent.reducer = WC_REDUCER_ADDITIVE;
    intent.source_sequence = source_sequence;
    intent.target = target;
    intent.value0 = (uint64_t)delta;
    return intent;
}

static int cause_cmp(const wc_cause_v0 *a, const wc_cause_v0 *b) {
#define CMP_FIELD(field) do { if (a->field < b->field) return -1; if (a->field > b->field) return 1; } while (0)
    CMP_FIELD(lane_rank);
    CMP_FIELD(subsystem_rank);
    CMP_FIELD(source_zone);
    CMP_FIELD(source_object);
    CMP_FIELD(source_sequence);
    CMP_FIELD(kind);
#undef CMP_FIELD
    return 0;
}

static int intent_cmp(const wc_intent_v0 *a, const wc_intent_v0 *b) {
#define CMP_FIELD(field) do { if (a->field < b->field) return -1; if (a->field > b->field) return 1; } while (0)
    CMP_FIELD(lane_rank);
    CMP_FIELD(subsystem_rank);
    CMP_FIELD(source_zone);
    CMP_FIELD(source_object);
    CMP_FIELD(source_sequence);
    CMP_FIELD(kind);
#undef CMP_FIELD
    return 0;
}

static void sort_causes(wc_cause_v0 *values, size_t count) {
    size_t i;
    for (i = 1; i < count; ++i) {
        wc_cause_v0 value = values[i];
        size_t j = i;
        while (j > 0 && cause_cmp(&value, &values[j - 1]) < 0) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = value;
    }
}

static void sort_intents(wc_intent_v0 *values, size_t count) {
    size_t i;
    for (i = 1; i < count; ++i) {
        wc_intent_v0 value = values[i];
        size_t j = i;
        while (j > 0 && intent_cmp(&value, &values[j - 1]) < 0) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = value;
    }
}

int wc_world_resolve(const wc_world_v0 *world, uint64_t key) {
    size_t i;
    if (key == 0u) return -1;
    for (i = 0; i < WC_CP7_OBJECT_CAPACITY; ++i) {
        if (world->alive[i] && world->object_key[i] == key) return (int)i;
    }
    return -1;
}

wc_handle_v0 wc_world_handle(const wc_world_v0 *world, uint64_t key) {
    wc_handle_v0 handle = {UINT32_MAX, 0u};
    int slot = wc_world_resolve(world, key);
    if (slot >= 0) {
        handle.slot = (uint32_t)slot;
        handle.generation = world->generation[slot];
    }
    return handle;
}

int wc_world_validate_handle(const wc_world_v0 *world, wc_handle_v0 handle, uint64_t *key_out) {
    if (handle.slot >= WC_CP7_OBJECT_CAPACITY || !world->alive[handle.slot] ||
        world->generation[handle.slot] != handle.generation) return 0;
    if (key_out != NULL) *key_out = world->object_key[handle.slot];
    return 1;
}

static int valid_position(int32_t value) {
    return value >= 0 && value < (32 << 16);
}

static int validate_intent(const wc_intent_v0 *intent) {
    uint16_t kind;
    uint16_t site;
    if (intent->reserved != 0u) return 0;
    switch (intent->kind) {
        case WC_INTENT_DESPAWN_OBJECT:
            return intent->reducer == WC_REDUCER_LIFECYCLE && intent->target != 0u;
        case WC_INTENT_SPAWN_OBJECT:
            kind = (uint16_t)(intent->aux & 0xffffu);
            site = (uint16_t)((intent->aux >> 16u) & 0xffffu);
            return intent->reducer == WC_REDUCER_LIFECYCLE && intent->target == 0u &&
                   intent->flags <= UINT16_MAX &&
                   (kind == WC_KIND_ACTOR || kind == WC_KIND_ITEM) &&
                   (site == WC_NON_SPATIAL_SITE || site < 1024u);
        case WC_INTENT_REPLACE_KINEMATICS:
            return intent->reducer == WC_REDUCER_EXCLUSIVE && intent->target != 0u &&
                   valid_position(unpack_low(intent->value0)) &&
                   valid_position(unpack_high(intent->value0)) &&
                   valid_position(unpack_low(intent->value1));
        case WC_INTENT_APPLY_HEALTH_DELTA:
            return intent->reducer == WC_REDUCER_ADDITIVE && intent->target != 0u;
        default:
            return 0;
    }
}

static wc_transition_result_v0 fault_result(const wc_world_v0 *world, uint32_t fault) {
    wc_transition_result_v0 result;
    result.status = WC_STATUS_FAULT;
    result.fault = fault;
    result.accepted_intents = 0u;
    result.rejected_requests = 0u;
    result.transition_id = world->accepted_transition_id;
    result.fingerprint = wc_world_fingerprint(world);
    return result;
}

wc_transition_result_v0 wc_world_transact(wc_world_v0 *world,
                                          const wc_cause_v0 *causes, size_t cause_count,
                                          const wc_intent_v0 *intents, size_t intent_count) {
    wc_cause_v0 sorted_causes[WC_CP7_CAUSE_CAPACITY];
    wc_intent_v0 sorted_intents[WC_CP7_INTENT_CAPACITY];
    wc_world_v0 next;
    uint8_t simulated_alive[WC_CP7_OBJECT_CAPACITY];
    uint8_t lifecycle[WC_CP7_OBJECT_CAPACITY] = {0};
    uint8_t exclusive[WC_CP7_OBJECT_CAPACITY] = {0};
    uint8_t health_touched[WC_CP7_OBJECT_CAPACITY] = {0};
    int64_t health_delta[WC_CP7_OBJECT_CAPACITY] = {0};
    uint32_t plan_slot[WC_CP7_INTENT_CAPACITY];
    uint64_t plan_key[WC_CP7_INTENT_CAPACITY] = {0};
    uint64_t accepted_spawns = 0u;
    uint32_t rejected = 0u;
    size_t i;

    if (cause_count == 0u) {
        wc_transition_result_v0 result;
        if (intent_count != 0u) return fault_result(world, WC_FAULT_CAUSE_MISSING);
        result.status = WC_STATUS_NO_WORK;
        result.fault = WC_FAULT_NONE;
        result.accepted_intents = 0u;
        result.rejected_requests = 0u;
        result.transition_id = world->accepted_transition_id;
        result.fingerprint = wc_world_fingerprint(world);
        return result;
    }
    if (cause_count > WC_CP7_CAUSE_CAPACITY) return fault_result(world, WC_FAULT_CAUSE_CAPACITY);
    if (intent_count > WC_CP7_INTENT_CAPACITY) return fault_result(world, WC_FAULT_INTENT_CAPACITY);
    if (world->accepted_transition_id == UINT64_MAX) return fault_result(world, WC_FAULT_TRANSITION_EXHAUSTED);

    memcpy(sorted_causes, causes, cause_count * sizeof(*causes));
    sort_causes(sorted_causes, cause_count);
    for (i = 0; i < cause_count; ++i) {
        if (sorted_causes[i].kind < 1u || sorted_causes[i].kind > 5u)
            return fault_result(world, WC_FAULT_INVALID_CAUSE);
        if (i > 0u && sorted_causes[i].lane_rank == sorted_causes[i - 1u].lane_rank &&
            sorted_causes[i].subsystem_rank == sorted_causes[i - 1u].subsystem_rank &&
            sorted_causes[i].source_zone == sorted_causes[i - 1u].source_zone &&
            sorted_causes[i].source_object == sorted_causes[i - 1u].source_object &&
            sorted_causes[i].source_sequence == sorted_causes[i - 1u].source_sequence)
            return fault_result(world, WC_FAULT_DUPLICATE_CAUSE);
    }

    memcpy(sorted_intents, intents, intent_count * sizeof(*intents));
    sort_intents(sorted_intents, intent_count);
    memcpy(simulated_alive, world->alive, sizeof(simulated_alive));
    for (i = 0; i < intent_count; ++i) plan_slot[i] = UINT32_MAX;

    for (i = 0; i < intent_count; ++i) {
        wc_intent_v0 *intent = &sorted_intents[i];
        int slot;
        if (i > 0u && intent->lane_rank == sorted_intents[i - 1u].lane_rank &&
            intent->subsystem_rank == sorted_intents[i - 1u].subsystem_rank &&
            intent->source_zone == sorted_intents[i - 1u].source_zone &&
            intent->source_object == sorted_intents[i - 1u].source_object &&
            intent->source_sequence == sorted_intents[i - 1u].source_sequence)
            return fault_result(world, WC_FAULT_DUPLICATE_INTENT);
        if (!validate_intent(intent)) return fault_result(world, WC_FAULT_INVALID_INTENT);
        if (intent->kind == WC_INTENT_SPAWN_OBJECT) {
            size_t candidate;
            slot = -1;
            for (candidate = 0; candidate < WC_CP7_OBJECT_CAPACITY; ++candidate) {
                if (!simulated_alive[candidate]) { slot = (int)candidate; break; }
            }
            if (slot < 0) {
                plan_slot[i] = UINT32_MAX - 1u;
                ++rejected;
            } else {
                if (world->next_object_key > UINT64_MAX - accepted_spawns)
                    return fault_result(world, WC_FAULT_KEY_EXHAUSTED);
                plan_slot[i] = (uint32_t)slot;
                plan_key[i] = world->next_object_key + accepted_spawns;
                simulated_alive[slot] = 1u;
                ++accepted_spawns;
            }
            continue;
        }
        slot = wc_world_resolve(world, intent->target);
        if (slot < 0) return fault_result(world, WC_FAULT_OBJECT_NOT_FOUND);
        plan_slot[i] = (uint32_t)slot;
        if (intent->kind == WC_INTENT_DESPAWN_OBJECT) {
            if (lifecycle[slot]) return fault_result(world, WC_FAULT_DUPLICATE_LIFECYCLE);
            lifecycle[slot] = 1u;
            simulated_alive[slot] = 0u;
        } else if (intent->kind == WC_INTENT_REPLACE_KINEMATICS) {
            if (exclusive[slot]) return fault_result(world, WC_FAULT_EXCLUSIVE_CONFLICT);
            exclusive[slot] = 1u;
        } else if (intent->kind == WC_INTENT_APPLY_HEALTH_DELTA) {
            int64_t delta;
            if (world->kind[slot] != WC_KIND_ACTOR)
                return fault_result(world, WC_FAULT_INVALID_INTENT);
            delta = (int64_t)intent->value0;
            if ((delta > 0 && health_delta[slot] > INT64_MAX - delta) ||
                (delta < 0 && health_delta[slot] < INT64_MIN - delta))
                return fault_result(world, WC_FAULT_WIDE_OVERFLOW);
            health_delta[slot] += delta;
            health_touched[slot] = 1u;
        }
    }

    for (i = 0; i < WC_CP7_OBJECT_CAPACITY; ++i) {
        if (lifecycle[i] && (exclusive[i] || health_touched[i]))
            return fault_result(world, WC_FAULT_LIFECYCLE_WRITE_CONFLICT);
    }
    if (accepted_spawns > UINT64_MAX - world->next_object_key)
        return fault_result(world, WC_FAULT_KEY_EXHAUSTED);

    next = *world;
    for (i = 0; i < intent_count; ++i) {
        wc_intent_v0 *intent = &sorted_intents[i];
        if (intent->kind == WC_INTENT_DESPAWN_OBJECT) {
            uint32_t slot = plan_slot[i];
            next.alive[slot] = 0u;
            next.object_key[slot] = 0u;
            next.kind[slot] = 0u;
            next.flags[slot] = 0u;
            next.surface_site[slot] = WC_NON_SPATIAL_SITE;
            next.next_in_site[slot] = UINT32_MAX;
            next.pos_x[slot] = next.pos_y[slot] = next.pos_z[slot] = 0;
            next.vel_x[slot] = next.vel_y[slot] = next.vel_z[slot] = 0;
            next.orientation[slot] = 0u;
            next.actor_health[slot] = next.item_type[slot] = next.item_stack[slot] = 0u;
            next.generation[slot] += 1u;
            if (next.generation[slot] == 0u) next.generation[slot] = 1u;
            --next.object_count;
        }
    }
    for (i = 0; i < intent_count; ++i) {
        wc_intent_v0 *intent = &sorted_intents[i];
        if (intent->kind == WC_INTENT_SPAWN_OBJECT && plan_slot[i] != UINT32_MAX - 1u) {
            uint32_t slot = plan_slot[i];
            uint16_t kind = (uint16_t)(intent->aux & 0xffffu);
            next.alive[slot] = 1u;
            next.object_key[slot] = plan_key[i];
            next.kind[slot] = kind;
            next.flags[slot] = (uint16_t)intent->flags;
            next.surface_site[slot] = (uint16_t)((intent->aux >> 16u) & 0xffffu);
            next.next_in_site[slot] = UINT32_MAX;
            next.orientation[slot] = 0u;
            if (kind == WC_KIND_ACTOR) next.actor_health[slot] = (uint16_t)intent->value0;
            else {
                next.item_type[slot] = (uint16_t)intent->value0;
                next.item_stack[slot] = (uint16_t)(intent->value0 >> 16u);
            }
            ++next.object_count;
        }
    }
    for (i = 0; i < intent_count; ++i) {
        wc_intent_v0 *intent = &sorted_intents[i];
        if (intent->kind == WC_INTENT_REPLACE_KINEMATICS) {
            uint32_t slot = plan_slot[i];
            next.pos_x[slot] = unpack_low(intent->value0);
            next.pos_y[slot] = unpack_high(intent->value0);
            next.pos_z[slot] = unpack_low(intent->value1);
            next.vel_x[slot] = unpack_high(intent->value1);
            next.vel_y[slot] = unpack_low(intent->aux);
            next.vel_z[slot] = unpack_high(intent->aux);
        }
    }
    for (i = 0; i < WC_CP7_OBJECT_CAPACITY; ++i) {
        if (health_touched[i]) {
            int64_t value = (int64_t)next.actor_health[i] + health_delta[i];
            if (value <= 0) next.actor_health[i] = 0u;
            else if (value >= UINT16_MAX) next.actor_health[i] = UINT16_MAX;
            else next.actor_health[i] = (uint16_t)value;
        }
    }
    next.next_object_key += accepted_spawns;
    ++next.accepted_transition_id;
    *world = next;

    {
        wc_transition_result_v0 result;
        result.status = WC_STATUS_OK;
        result.fault = WC_FAULT_NONE;
        result.accepted_intents = (uint32_t)intent_count - rejected;
        result.rejected_requests = rejected;
        result.transition_id = world->accepted_transition_id;
        result.fingerprint = wc_world_fingerprint(world);
        return result;
    }
}

uint64_t wc_world_fingerprint(const wc_world_v0 *world) {
    const uint8_t *bytes = (const uint8_t *)world;
    uint64_t hash = UINT64_C(14695981039346656037);
    size_t i;
    for (i = 0; i < sizeof(*world); ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint32_t wc_xi_port_status(void) { return WC_XI_NOT_DERIVED; }

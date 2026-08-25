#include "phase_graph_custody_v0.h"

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

static int edge_less(const pgg_edge_v0 *left, const pgg_edge_v0 *right) {
    if (left->node_a != right->node_a) return left->node_a < right->node_a;
    return left->node_b < right->node_b;
}

static uint32_t canonicalize_edges(pgg_graph_v0 *graph,
                                   const uint32_t *edge_pairs,
                                   size_t edge_count) {
    size_t i;
    size_t j;
    for (i = 0u; i < edge_count; ++i) {
        uint32_t a = edge_pairs[i * 2u];
        uint32_t b = edge_pairs[i * 2u + 1u];
        if (a >= graph->node_count || b >= graph->node_count || a == b)
            return PGG_STATUS_INVALID_ARGUMENT;
        if (a > b) {
            uint32_t temp = a;
            a = b;
            b = temp;
        }
        graph->edges[i].node_a = a;
        graph->edges[i].node_b = b;
        graph->edges[i].edge_key = 0u;
    }
    for (i = 1u; i < edge_count; ++i) {
        pgg_edge_v0 key = graph->edges[i];
        j = i;
        while (j > 0u && edge_less(&key, &graph->edges[j - 1u])) {
            graph->edges[j] = graph->edges[j - 1u];
            --j;
        }
        graph->edges[j] = key;
    }
    for (i = 0u; i < edge_count; ++i) {
        if (i > 0u && graph->edges[i - 1u].node_a == graph->edges[i].node_a &&
            graph->edges[i - 1u].node_b == graph->edges[i].node_b)
            return PGG_STATUS_DUPLICATE_EDGE;
        graph->edges[i].edge_key = (uint64_t)i + 1u;
    }
    return PGG_STATUS_OK;
}

static int graph_connected(const pgg_graph_v0 *graph) {
    uint8_t visited[PGG_GRAPH_MAX_NODES] = {0u};
    uint32_t changed = 1u;
    uint32_t i;
    uint32_t count = 0u;
    visited[0] = 1u;
    while (changed != 0u) {
        changed = 0u;
        for (i = 0u; i < graph->edge_count; ++i) {
            const pgg_edge_v0 *edge = &graph->edges[i];
            if (visited[edge->node_a] != 0u && visited[edge->node_b] == 0u) {
                visited[edge->node_b] = 1u;
                changed = 1u;
            }
            if (visited[edge->node_b] != 0u && visited[edge->node_a] == 0u) {
                visited[edge->node_a] = 1u;
                changed = 1u;
            }
        }
    }
    for (i = 0u; i < graph->node_count; ++i) count += visited[i] != 0u;
    return count == graph->node_count;
}

uint32_t pgg_find_edge(const pgg_graph_v0 *graph,
                       uint32_t left,
                       uint32_t right) {
    uint32_t a;
    uint32_t b;
    uint32_t i;
    if (graph == NULL || left >= graph->node_count || right >= graph->node_count || left == right)
        return PGG_EDGE_NONE;
    a = left < right ? left : right;
    b = left < right ? right : left;
    for (i = 0u; i < graph->edge_count; ++i) {
        if (graph->edges[i].node_a == a && graph->edges[i].node_b == b) return i;
    }
    return PGG_EDGE_NONE;
}

static pgg_receipt_v0 receipt_base(const pgg_graph_v0 *graph,
                                   uint32_t source_index,
                                   uint32_t destination_index) {
    pgg_receipt_v0 receipt;
    uint32_t edge_index;
    memset(&receipt, 0, sizeof(receipt));
    receipt.source_index = source_index;
    receipt.destination_index = destination_index;
    receipt.edge_index = PGG_EDGE_NONE;
    if (graph == NULL) return receipt;
    receipt.graph_commit_id = graph->graph_commit_id;
    receipt.active_index = graph->active_index;
    receipt.traveler_key = graph->traveler_key;
    receipt.actor_key = graph->actor_key;
    edge_index = pgg_find_edge(graph, source_index, destination_index);
    if (edge_index != PGG_EDGE_NONE) {
        receipt.edge_index = edge_index;
        receipt.edge_key = graph->edges[edge_index].edge_key;
    }
    if (source_index < graph->node_count) {
        const pgg_node_v0 *node = &graph->nodes[source_index];
        if (node->resident != 0u) {
            receipt.source_local_commit_id = node->cell.world.accepted_transition_id;
            receipt.source_phase_fingerprint = pww_phase_fingerprint(&node->cell.phase);
            receipt.source_world_fingerprint = wc_world_fingerprint(&node->cell.world);
        } else {
            receipt.source_local_commit_id = node->snapshot_receipt.source_local_commit_id;
            receipt.source_phase_fingerprint = node->snapshot_receipt.phase_fingerprint;
            receipt.source_world_fingerprint = node->snapshot_receipt.world_fingerprint;
            receipt.snapshot_bytes = node->snapshot_bytes;
        }
    }
    if (destination_index < graph->node_count) {
        const pgg_node_v0 *node = &graph->nodes[destination_index];
        if (node->resident != 0u) {
            receipt.destination_local_commit_id = node->cell.world.accepted_transition_id;
            receipt.destination_phase_fingerprint = pww_phase_fingerprint(&node->cell.phase);
            receipt.destination_world_fingerprint = wc_world_fingerprint(&node->cell.world);
        } else {
            receipt.destination_local_commit_id = node->snapshot_receipt.source_local_commit_id;
            receipt.destination_phase_fingerprint = node->snapshot_receipt.phase_fingerprint;
            receipt.destination_world_fingerprint = node->snapshot_receipt.world_fingerprint;
            receipt.snapshot_bytes = node->snapshot_bytes;
        }
    }
    return receipt;
}

static uint32_t map_source_failure(const pww_result_v0 *result,
                                   pgg_receipt_v0 *receipt) {
    receipt->fault_participant = PGG_PARTICIPANT_SOURCE;
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PGG_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PGG_STATUS_SOURCE_WORLD_FAILURE;
    return PGG_STATUS_SOURCE_PHASE_FAILURE;
}

static uint32_t map_destination_failure(const pww_result_v0 *result,
                                        pgg_receipt_v0 *receipt) {
    receipt->fault_participant = PGG_PARTICIPANT_DESTINATION;
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PGG_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PGG_STATUS_DESTINATION_WORLD_FAILURE;
    return PGG_STATUS_DESTINATION_PHASE_FAILURE;
}

uint32_t pgg_graph_init(pgg_graph_v0 *graph,
                        uint64_t graph_key,
                        uint64_t traveler_key,
                        const uint64_t *zone_keys,
                        size_t node_count,
                        const uint32_t *edge_pairs,
                        size_t edge_count,
                        uint32_t pair_limb_limit) {
    size_t i;
    uint32_t status;
    if (graph == NULL || zone_keys == NULL || edge_pairs == NULL ||
        graph_key == 0u || traveler_key == 0u ||
        node_count < 3u || node_count > PGG_GRAPH_MAX_NODES ||
        edge_count < node_count - 1u || edge_count > PGG_GRAPH_MAX_EDGES ||
        pair_limb_limit == 0u || !zones_unique(zone_keys, node_count))
        return PGG_STATUS_INVALID_ARGUMENT;
    memset(graph, 0, sizeof(*graph));
    graph->graph_key = graph_key;
    graph->traveler_key = traveler_key;
    graph->node_count = (uint32_t)node_count;
    graph->edge_count = (uint32_t)edge_count;
    graph->active_index = 0u;
    status = canonicalize_edges(graph, edge_pairs, edge_count);
    if (status != PGG_STATUS_OK) {
        memset(graph, 0, sizeof(*graph));
        return status;
    }
    if (!graph_connected(graph)) {
        memset(graph, 0, sizeof(*graph));
        return PGG_STATUS_GRAPH_DISCONNECTED;
    }
    for (i = 0u; i < node_count; ++i) {
        if (pww_cell_init(&graph->nodes[i].cell,
                          zone_keys[i], pair_limb_limit) != PWW_STATUS_OK) {
            memset(graph, 0, sizeof(*graph));
            return PGG_STATUS_INVALID_ARGUMENT;
        }
        graph->nodes[i].zone_key = zone_keys[i];
        graph->nodes[i].resident = 1u;
        graph->nodes[i].retained = 1u;
    }
    return PGG_STATUS_OK;
}

pgg_receipt_v0 pgg_bootstrap(pgg_graph_v0 *graph,
                             uint64_t bootstrap_cause_id,
                             uint32_t source_sequence,
                             uint16_t actor_health) {
    pgg_graph_v0 staged;
    pgg_receipt_v0 receipt = receipt_base(graph, 0u, 0u);
    wc_cause_v0 cause;
    wc_intent_v0 intent;
    pww_result_v0 result;
    uint64_t expected_key;
    int slot;
    if (graph == NULL || bootstrap_cause_id == 0u || source_sequence == 0u || actor_health == 0u) {
        receipt.status = PGG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (graph->bootstrapped != 0u) {
        receipt.status = PGG_STATUS_ALREADY_BOOTSTRAPPED;
        return receipt;
    }
    staged = *graph;
    expected_key = staged.nodes[0].cell.world.next_object_key;
    cause = wc_external_input(source_sequence, staged.traveler_key, bootstrap_cause_id);
    intent = wc_spawn_actor(source_sequence, actor_health, WC_NON_SPATIAL_SITE);
    result = pww_cell_transact(&staged.nodes[0].cell, &cause, 1u, &intent, 1u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&result, &receipt);
        return receipt;
    }
    slot = wc_world_resolve(&staged.nodes[0].cell.world, expected_key);
    if (slot < 0 || staged.nodes[0].cell.world.kind[slot] != WC_KIND_ACTOR ||
        staged.nodes[0].cell.world.actor_health[slot] != actor_health) {
        receipt.status = PGG_STATUS_DESTINATION_SPAWN_REJECTED;
        return receipt;
    }
    staged.actor_key = expected_key;
    staged.bootstrapped = 1u;
    *graph = staged;
    receipt = receipt_base(graph, 0u, 0u);
    receipt.status = PGG_STATUS_OK;
    receipt.source_primitive = result.primitive;
    return receipt;
}

pgg_receipt_v0 pgg_advance_active(pgg_graph_v0 *graph,
                                  uint32_t source_sequence,
                                  uint64_t payload0,
                                  uint64_t payload1) {
    pgg_graph_v0 staged;
    pgg_receipt_v0 receipt;
    wc_cause_v0 cause;
    pww_result_v0 result;
    uint32_t active;
    if (graph == NULL || source_sequence == 0u) {
        receipt = receipt_base(graph, 0u, 0u);
        receipt.status = PGG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    active = graph->active_index;
    receipt = receipt_base(graph, active, active);
    if (graph->bootstrapped == 0u) {
        receipt.status = PGG_STATUS_NOT_BOOTSTRAPPED;
        return receipt;
    }
    if (active >= graph->node_count || graph->nodes[active].resident == 0u) {
        receipt.status = PGG_STATUS_NODE_NOT_RESIDENT;
        return receipt;
    }
    staged = *graph;
    cause = wc_external_input(source_sequence, payload0, payload1);
    result = pww_cell_transact(&staged.nodes[active].cell, &cause, 1u, NULL, 0u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&result, &receipt);
        return receipt;
    }
    *graph = staged;
    receipt = receipt_base(graph, active, active);
    receipt.status = PGG_STATUS_OK;
    receipt.source_primitive = result.primitive;
    return receipt;
}

pgg_receipt_v0 pgg_evict_node(pgg_graph_v0 *graph, uint32_t node_index) {
    pgg_graph_v0 staged;
    pgg_receipt_v0 receipt = receipt_base(graph, node_index, node_index);
    pcs_receipt_v1 snapshot_receipt;
    uint32_t status;
    if (graph == NULL) {
        receipt.status = PGG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (node_index >= graph->node_count) {
        receipt.status = PGG_STATUS_NODE_RANGE;
        return receipt;
    }
    if (node_index == graph->active_index) {
        receipt.status = PGG_STATUS_ACTIVE_NODE;
        return receipt;
    }
    if (graph->nodes[node_index].resident == 0u) {
        receipt.status = PGG_STATUS_NODE_NOT_RESIDENT;
        return receipt;
    }
    if (graph->nodes[node_index].cell.world.object_count != 0u) {
        receipt.status = PGG_STATUS_NODE_NOT_EMPTY;
        return receipt;
    }
    staged = *graph;
    status = pcs_snapshot_encode_v1(&staged.nodes[node_index].cell,
                                    staged.nodes[node_index].snapshot,
                                    sizeof(staged.nodes[node_index].snapshot),
                                    &snapshot_receipt);
    if (status != PCS_STATUS_OK) {
        receipt.status = PGG_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    staged.nodes[node_index].snapshot_receipt = snapshot_receipt;
    staged.nodes[node_index].snapshot_bytes = snapshot_receipt.snapshot_bytes;
    memset(&staged.nodes[node_index].cell, 0, sizeof(staged.nodes[node_index].cell));
    staged.nodes[node_index].resident = 0u;
    *graph = staged;
    receipt = receipt_base(graph, node_index, node_index);
    receipt.status = PGG_STATUS_OK;
    receipt.snapshot_bytes = snapshot_receipt.snapshot_bytes;
    return receipt;
}

pgg_receipt_v0 pgg_restore_node(pgg_graph_v0 *graph, uint32_t node_index) {
    pgg_graph_v0 staged;
    pgg_receipt_v0 receipt = receipt_base(graph, node_index, node_index);
    pww_cell_v0 candidate;
    pcs_receipt_v1 decoded;
    uint32_t status;
    if (graph == NULL) {
        receipt.status = PGG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (node_index >= graph->node_count) {
        receipt.status = PGG_STATUS_NODE_RANGE;
        return receipt;
    }
    if (graph->nodes[node_index].resident != 0u) {
        receipt.status = PGG_STATUS_NODE_ALREADY_RESIDENT;
        return receipt;
    }
    if (graph->nodes[node_index].snapshot_bytes == 0u) {
        receipt.status = PGG_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    memset(&candidate, 0, sizeof(candidate));
    status = pcs_snapshot_decode_v1(graph->nodes[node_index].snapshot,
                                    (size_t)graph->nodes[node_index].snapshot_bytes,
                                    &candidate, &decoded);
    if (status != PCS_STATUS_OK) {
        receipt.status = PGG_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    if (decoded.source_zone_key != graph->nodes[node_index].zone_key ||
        decoded.source_local_commit_id != graph->nodes[node_index].snapshot_receipt.source_local_commit_id ||
        decoded.phase_fingerprint != graph->nodes[node_index].snapshot_receipt.phase_fingerprint ||
        decoded.world_fingerprint != graph->nodes[node_index].snapshot_receipt.world_fingerprint) {
        receipt.status = PGG_STATUS_SNAPSHOT_MISMATCH;
        return receipt;
    }
    staged = *graph;
    staged.nodes[node_index].cell = candidate;
    memset(staged.nodes[node_index].snapshot, 0, sizeof(staged.nodes[node_index].snapshot));
    memset(&staged.nodes[node_index].snapshot_receipt, 0,
           sizeof(staged.nodes[node_index].snapshot_receipt));
    staged.nodes[node_index].snapshot_bytes = 0u;
    staged.nodes[node_index].resident = 1u;
    *graph = staged;
    receipt = receipt_base(graph, node_index, node_index);
    receipt.status = PGG_STATUS_OK;
    return receipt;
}

uint32_t pgg_provision_node_pair_limbs(pgg_graph_v0 *graph,
                                       uint32_t node_index,
                                       uint32_t new_limit) {
    if (graph == NULL || node_index >= graph->node_count ||
        graph->nodes[node_index].resident == 0u)
        return PGG_STATUS_INVALID_ARGUMENT;
    if (pww_cell_provision_pair_limbs(&graph->nodes[node_index].cell,
                                      new_limit) != PWW_STATUS_OK)
        return PGG_STATUS_INVALID_ARGUMENT;
    return PGG_STATUS_OK;
}

pgg_receipt_v0 pgg_handoff(pgg_graph_v0 *graph,
                           uint64_t handoff_cause_id,
                           uint32_t destination_index,
                           uint64_t expected_source_local_commit_id,
                           uint64_t expected_destination_local_commit_id,
                           uint32_t source_sequence,
                           uint32_t destination_sequence) {
    pgg_graph_v0 staged;
    pgg_receipt_v0 receipt;
    wc_cause_v0 source_cause;
    wc_cause_v0 destination_cause;
    wc_intent_v0 source_intent;
    wc_intent_v0 destination_intent;
    pww_result_v0 source_result;
    pww_result_v0 destination_result;
    uint32_t source_index;
    uint32_t edge_index;
    uint64_t expected_destination_key;
    uint64_t next_commit_id;
    int source_actor_slot;
    int destination_actor_slot;
    uint16_t health;

    source_index = graph == NULL ? 0u : graph->active_index;
    receipt = receipt_base(graph, source_index, destination_index);
    if (graph == NULL || handoff_cause_id == 0u || source_sequence == 0u ||
        destination_sequence == 0u) {
        receipt.status = PGG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (graph->bootstrapped == 0u) {
        receipt.status = PGG_STATUS_NOT_BOOTSTRAPPED;
        return receipt;
    }
    if (graph->consumed_handoff_cause_id == handoff_cause_id) {
        receipt.status = PGG_STATUS_DUPLICATE_IGNORED;
        return receipt;
    }
    if (destination_index >= graph->node_count) {
        receipt.status = PGG_STATUS_NODE_RANGE;
        return receipt;
    }
    edge_index = pgg_find_edge(graph, source_index, destination_index);
    if (edge_index == PGG_EDGE_NONE) {
        receipt.status = PGG_STATUS_EDGE_NOT_FOUND;
        return receipt;
    }
    receipt.edge_index = edge_index;
    receipt.edge_key = graph->edges[edge_index].edge_key;
    if (graph->nodes[source_index].resident == 0u ||
        graph->nodes[destination_index].resident == 0u) {
        receipt.status = PGG_STATUS_NODE_NOT_RESIDENT;
        return receipt;
    }
    if (graph->nodes[source_index].cell.world.accepted_transition_id !=
            expected_source_local_commit_id ||
        graph->nodes[destination_index].cell.world.accepted_transition_id !=
            expected_destination_local_commit_id) {
        receipt.status = PGG_STATUS_STALE_VERSION;
        return receipt;
    }
    source_actor_slot = wc_world_resolve(&graph->nodes[source_index].cell.world,
                                         graph->actor_key);
    if (source_actor_slot < 0 ||
        graph->nodes[source_index].cell.world.kind[source_actor_slot] != WC_KIND_ACTOR) {
        receipt.status = PGG_STATUS_ACTOR_MISSING;
        return receipt;
    }
    health = graph->nodes[source_index].cell.world.actor_health[source_actor_slot];
    if (graph->graph_commit_id == UINT64_MAX) {
        receipt.status = PGG_STATUS_COMMIT_EXHAUSTED;
        return receipt;
    }
    next_commit_id = graph->graph_commit_id + 1u;
    staged = *graph;

    source_cause = wc_external_input(source_sequence,
                                     staged.traveler_key,
                                     staged.edges[edge_index].edge_key);
    source_intent = wc_despawn(source_sequence, staged.actor_key);
    source_result = pww_cell_transact(&staged.nodes[source_index].cell,
                                      &source_cause, 1u,
                                      &source_intent, 1u);
    if (source_result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&source_result, &receipt);
        return receipt;
    }

    expected_destination_key = staged.nodes[destination_index].cell.world.next_object_key;
    destination_cause = wc_external_input(destination_sequence,
                                          staged.traveler_key,
                                          handoff_cause_id);
    destination_intent = wc_spawn_actor(destination_sequence,
                                        health, WC_NON_SPATIAL_SITE);
    destination_result = pww_cell_transact(&staged.nodes[destination_index].cell,
                                           &destination_cause, 1u,
                                           &destination_intent, 1u);
    if (destination_result.status != PWW_STATUS_OK) {
        receipt.status = map_destination_failure(&destination_result, &receipt);
        return receipt;
    }
    destination_actor_slot = wc_world_resolve(&staged.nodes[destination_index].cell.world,
                                              expected_destination_key);
    if (destination_actor_slot < 0 ||
        staged.nodes[destination_index].cell.world.kind[destination_actor_slot] != WC_KIND_ACTOR ||
        staged.nodes[destination_index].cell.world.actor_health[destination_actor_slot] != health) {
        receipt.status = PGG_STATUS_DESTINATION_SPAWN_REJECTED;
        receipt.fault_participant = PGG_PARTICIPANT_DESTINATION;
        return receipt;
    }

    staged.active_index = destination_index;
    staged.actor_key = expected_destination_key;
    staged.consumed_handoff_cause_id = handoff_cause_id;
    staged.graph_commit_id = next_commit_id;
    *graph = staged;

    receipt = receipt_base(graph, source_index, destination_index);
    receipt.status = PGG_STATUS_OK;
    receipt.source_primitive = source_result.primitive;
    receipt.destination_primitive = destination_result.primitive;
    return receipt;
}

uint64_t pgg_graph_fingerprint(const pgg_graph_v0 *graph) {
    uint64_t hash = UINT64_C(1469598103934665603);
    uint32_t i;
    if (graph == NULL) return 0u;
    hash_feed(&hash, &graph->graph_key, sizeof(graph->graph_key));
    hash_feed(&hash, &graph->traveler_key, sizeof(graph->traveler_key));
    hash_feed(&hash, &graph->actor_key, sizeof(graph->actor_key));
    hash_feed(&hash, &graph->graph_commit_id, sizeof(graph->graph_commit_id));
    hash_feed(&hash, &graph->consumed_handoff_cause_id,
              sizeof(graph->consumed_handoff_cause_id));
    hash_feed(&hash, &graph->node_count, sizeof(graph->node_count));
    hash_feed(&hash, &graph->edge_count, sizeof(graph->edge_count));
    hash_feed(&hash, &graph->active_index, sizeof(graph->active_index));
    hash_feed(&hash, &graph->bootstrapped, sizeof(graph->bootstrapped));
    for (i = 0u; i < graph->edge_count; ++i) {
        hash_feed(&hash, &graph->edges[i], sizeof(graph->edges[i]));
    }
    for (i = 0u; i < graph->node_count; ++i) {
        const pgg_node_v0 *node = &graph->nodes[i];
        hash_feed(&hash, &node->zone_key, sizeof(node->zone_key));
        hash_feed(&hash, &node->resident, sizeof(node->resident));
        hash_feed(&hash, &node->retained, sizeof(node->retained));
        if (node->resident != 0u) {
            uint64_t phase = pww_phase_fingerprint(&node->cell.phase);
            uint64_t world = wc_world_fingerprint(&node->cell.world);
            hash_feed(&hash, &phase, sizeof(phase));
            hash_feed(&hash, &world, sizeof(world));
        } else {
            hash_feed(&hash, &node->snapshot_receipt, sizeof(node->snapshot_receipt));
            hash_feed(&hash, node->snapshot, (size_t)node->snapshot_bytes);
        }
    }
    return hash;
}

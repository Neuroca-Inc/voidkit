#include "phase_multi_graph_v0.h"

#include <limits.h>
#include <string.h>

typedef struct pmg_candidate_v0 {
    pmg_request_v0 request;
    uint32_t source_node;
    uint32_t edge_index;
    uint64_t edge_key;
} pmg_candidate_v0;

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

static int edge_less(const pmg_edge_v0 *left, const pmg_edge_v0 *right) {
    if (left->node_a != right->node_a) return left->node_a < right->node_a;
    return left->node_b < right->node_b;
}

static uint32_t canonicalize_edges(pmg_graph_v0 *graph,
                                   const uint32_t *edge_pairs,
                                   size_t edge_count) {
    size_t i;
    size_t j;
    for (i = 0u; i < edge_count; ++i) {
        uint32_t a = edge_pairs[i * 2u];
        uint32_t b = edge_pairs[i * 2u + 1u];
        if (a >= graph->node_count || b >= graph->node_count || a == b)
            return PMG_STATUS_INVALID_ARGUMENT;
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
        pmg_edge_v0 key = graph->edges[i];
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
            return PMG_STATUS_DUPLICATE_EDGE;
        graph->edges[i].edge_key = (uint64_t)i + 1u;
    }
    return PMG_STATUS_OK;
}

static int graph_connected(const pmg_graph_v0 *graph) {
    uint8_t visited[PMG_MAX_NODES] = {0u};
    uint32_t changed = 1u;
    uint32_t i;
    uint32_t count = 0u;
    visited[0] = 1u;
    while (changed != 0u) {
        changed = 0u;
        for (i = 0u; i < graph->edge_count; ++i) {
            const pmg_edge_v0 *edge = &graph->edges[i];
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

uint32_t pmg_find_edge(const pmg_graph_v0 *graph, uint32_t left, uint32_t right) {
    uint32_t a;
    uint32_t b;
    uint32_t i;
    if (graph == NULL || left >= graph->node_count || right >= graph->node_count || left == right)
        return PMG_EDGE_NONE;
    a = left < right ? left : right;
    b = left < right ? right : left;
    for (i = 0u; i < graph->edge_count; ++i) {
        if (graph->edges[i].node_a == a && graph->edges[i].node_b == b) return i;
    }
    return PMG_EDGE_NONE;
}

uint32_t pmg_find_traveler(const pmg_graph_v0 *graph, uint64_t traveler_key) {
    uint32_t i;
    if (graph == NULL || traveler_key == 0u) return PMG_TRAVELER_NONE;
    for (i = 0u; i < graph->traveler_count; ++i) {
        if (graph->travelers[i].active != 0u &&
            graph->travelers[i].traveler_key == traveler_key)
            return i;
    }
    return PMG_TRAVELER_NONE;
}

static pmg_receipt_v0 receipt_base(const pmg_graph_v0 *graph,
                                   uint32_t traveler_index) {
    pmg_receipt_v0 receipt;
    memset(&receipt, 0, sizeof(receipt));
    receipt.traveler_index = traveler_index;
    receipt.source_node = PMG_NODE_NONE;
    receipt.destination_node = PMG_NODE_NONE;
    receipt.edge_index = PMG_EDGE_NONE;
    if (graph == NULL) return receipt;
    receipt.graph_commit_id = graph->graph_commit_id;
    receipt.claim_set_id = graph->active_claim_set_id;
    receipt.claims_pending = graph->claims_pending;
    if (traveler_index < graph->traveler_count &&
        graph->travelers[traveler_index].active != 0u) {
        const pmg_traveler_v0 *traveler = &graph->travelers[traveler_index];
        receipt.traveler_key = traveler->traveler_key;
        receipt.actor_key = traveler->actor_key;
        receipt.source_node = traveler->current_node;
        receipt.request_key = traveler->claim_request_key;
        if (traveler->claim_active != 0u) {
            receipt.destination_node = traveler->claim_destination_node;
            receipt.edge_index = traveler->claim_edge_index;
            if (traveler->claim_edge_index < graph->edge_count)
                receipt.edge_key = graph->edges[traveler->claim_edge_index].edge_key;
        }
        if (traveler->current_node < graph->node_count &&
            graph->nodes[traveler->current_node].resident != 0u) {
            const pmg_node_v0 *node = &graph->nodes[traveler->current_node];
            receipt.source_local_commit_id = node->cell.world.accepted_transition_id;
            receipt.source_phase_fingerprint = pww_phase_fingerprint(&node->cell.phase);
            receipt.source_world_fingerprint = wc_world_fingerprint(&node->cell.world);
        }
        if (traveler->claim_active != 0u &&
            traveler->claim_destination_node < graph->node_count &&
            graph->nodes[traveler->claim_destination_node].resident != 0u) {
            const pmg_node_v0 *node = &graph->nodes[traveler->claim_destination_node];
            receipt.destination_local_commit_id = node->cell.world.accepted_transition_id;
            receipt.destination_phase_fingerprint = pww_phase_fingerprint(&node->cell.phase);
            receipt.destination_world_fingerprint = wc_world_fingerprint(&node->cell.world);
        }
    }
    return receipt;
}

static uint32_t map_source_failure(const pww_result_v0 *result,
                                   pmg_receipt_v0 *receipt) {
    receipt->fault_participant = PMG_PARTICIPANT_SOURCE;
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PMG_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PMG_STATUS_SOURCE_WORLD_FAILURE;
    return PMG_STATUS_SOURCE_PHASE_FAILURE;
}

static uint32_t map_destination_failure(const pww_result_v0 *result,
                                        pmg_receipt_v0 *receipt) {
    receipt->fault_participant = PMG_PARTICIPANT_DESTINATION;
    if (result->status == PWW_STATUS_PROVISION_REQUIRED) {
        receipt->required_pair_limbs = result->required_pair_limbs;
        return PMG_STATUS_PROVISION_REQUIRED;
    }
    if (result->status == PWW_STATUS_WORLD_FAULT)
        return PMG_STATUS_DESTINATION_WORLD_FAILURE;
    return PMG_STATUS_DESTINATION_PHASE_FAILURE;
}

uint32_t pmg_graph_init(pmg_graph_v0 *graph,
                        uint64_t graph_key,
                        const uint64_t *zone_keys,
                        size_t node_count,
                        const uint32_t *edge_pairs,
                        size_t edge_count,
                        uint32_t pair_limb_limit) {
    size_t i;
    uint32_t status;
    if (graph == NULL || zone_keys == NULL || edge_pairs == NULL || graph_key == 0u ||
        node_count < 3u || node_count > PMG_MAX_NODES ||
        edge_count < node_count - 1u || edge_count > PMG_MAX_EDGES ||
        pair_limb_limit == 0u || !zones_unique(zone_keys, node_count))
        return PMG_STATUS_INVALID_ARGUMENT;
    memset(graph, 0, sizeof(*graph));
    graph->graph_key = graph_key;
    graph->node_count = (uint32_t)node_count;
    graph->edge_count = (uint32_t)edge_count;
    status = canonicalize_edges(graph, edge_pairs, edge_count);
    if (status != PMG_STATUS_OK) {
        memset(graph, 0, sizeof(*graph));
        return status;
    }
    if (!graph_connected(graph)) {
        memset(graph, 0, sizeof(*graph));
        return PMG_STATUS_GRAPH_DISCONNECTED;
    }
    for (i = 0u; i < node_count; ++i) {
        if (pww_cell_init(&graph->nodes[i].cell, zone_keys[i], pair_limb_limit) != PWW_STATUS_OK) {
            memset(graph, 0, sizeof(*graph));
            return PMG_STATUS_INVALID_ARGUMENT;
        }
        graph->nodes[i].zone_key = zone_keys[i];
        graph->nodes[i].resident = 1u;
        graph->nodes[i].retained = 1u;
    }
    return PMG_STATUS_OK;
}

pmg_receipt_v0 pmg_bootstrap_traveler(pmg_graph_v0 *graph,
                                      uint64_t traveler_key,
                                      uint32_t node_index,
                                      uint64_t bootstrap_cause_id,
                                      uint32_t source_sequence,
                                      uint16_t actor_health) {
    pmg_graph_v0 staged;
    pmg_receipt_v0 receipt = receipt_base(graph, PMG_TRAVELER_NONE);
    wc_cause_v0 cause;
    wc_intent_v0 intent;
    pww_result_v0 result;
    uint64_t expected_key;
    uint32_t slot;
    int actor_slot;
    if (graph == NULL || traveler_key == 0u || bootstrap_cause_id == 0u ||
        source_sequence == 0u || actor_health == 0u) {
        receipt.status = PMG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (pmg_find_traveler(graph, traveler_key) != PMG_TRAVELER_NONE) {
        receipt.status = PMG_STATUS_TRAVELER_EXISTS;
        return receipt;
    }
    if (graph->traveler_count >= PMG_MAX_TRAVELERS) {
        receipt.status = PMG_STATUS_TRAVELER_LIMIT;
        return receipt;
    }
    if (node_index >= graph->node_count) {
        receipt.status = PMG_STATUS_NODE_RANGE;
        return receipt;
    }
    if (graph->nodes[node_index].resident == 0u) {
        receipt.status = PMG_STATUS_NODE_NOT_RESIDENT;
        return receipt;
    }
    if (graph->node_claim_owner[node_index] != 0u) {
        receipt.status = PMG_STATUS_CLAIMS_PENDING;
        return receipt;
    }
    staged = *graph;
    slot = staged.traveler_count;
    expected_key = staged.nodes[node_index].cell.world.next_object_key;
    cause = wc_external_input(source_sequence, traveler_key, bootstrap_cause_id);
    intent = wc_spawn_actor(source_sequence, actor_health, WC_NON_SPATIAL_SITE);
    result = pww_cell_transact(&staged.nodes[node_index].cell, &cause, 1u, &intent, 1u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&result, &receipt);
        return receipt;
    }
    actor_slot = wc_world_resolve(&staged.nodes[node_index].cell.world, expected_key);
    if (actor_slot < 0 ||
        staged.nodes[node_index].cell.world.kind[actor_slot] != WC_KIND_ACTOR ||
        staged.nodes[node_index].cell.world.actor_health[actor_slot] != actor_health) {
        receipt.status = PMG_STATUS_DESTINATION_SPAWN_REJECTED;
        return receipt;
    }
    memset(&staged.travelers[slot], 0, sizeof(staged.travelers[slot]));
    staged.travelers[slot].traveler_key = traveler_key;
    staged.travelers[slot].actor_key = expected_key;
    staged.travelers[slot].current_node = node_index;
    staged.travelers[slot].claim_edge_index = PMG_EDGE_NONE;
    staged.travelers[slot].claim_destination_node = PMG_NODE_NONE;
    staged.travelers[slot].active = 1u;
    staged.nodes[node_index].occupant_mask |= UINT32_C(1) << slot;
    staged.traveler_count += 1u;
    *graph = staged;
    receipt = receipt_base(graph, slot);
    receipt.status = PMG_STATUS_OK;
    receipt.source_primitive = result.primitive;
    return receipt;
}

pmg_receipt_v0 pmg_advance_node(pmg_graph_v0 *graph,
                                uint32_t node_index,
                                uint32_t source_sequence,
                                uint64_t payload0,
                                uint64_t payload1) {
    pmg_graph_v0 staged;
    pmg_receipt_v0 receipt = receipt_base(graph, PMG_TRAVELER_NONE);
    wc_cause_v0 cause;
    pww_result_v0 result;
    if (graph == NULL || source_sequence == 0u) {
        receipt.status = PMG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (node_index >= graph->node_count) {
        receipt.status = PMG_STATUS_NODE_RANGE;
        return receipt;
    }
    if (graph->nodes[node_index].resident == 0u) {
        receipt.status = PMG_STATUS_NODE_NOT_RESIDENT;
        return receipt;
    }
    if (graph->node_claim_owner[node_index] != 0u) {
        receipt.status = PMG_STATUS_CLAIMS_PENDING;
        return receipt;
    }
    staged = *graph;
    cause = wc_external_input(source_sequence, payload0, payload1);
    result = pww_cell_transact(&staged.nodes[node_index].cell, &cause, 1u, NULL, 0u);
    if (result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&result, &receipt);
        return receipt;
    }
    *graph = staged;
    memset(&receipt, 0, sizeof(receipt));
    receipt.status = PMG_STATUS_OK;
    receipt.source_node = node_index;
    receipt.source_primitive = result.primitive;
    receipt.source_local_commit_id = result.local_commit_id;
    receipt.source_phase_fingerprint = result.phase_fingerprint;
    receipt.source_world_fingerprint = result.world_fingerprint;
    receipt.graph_commit_id = graph->graph_commit_id;
    receipt.claims_pending = graph->claims_pending;
    return receipt;
}

static int candidate_less(const pmg_candidate_v0 *left,
                          const pmg_candidate_v0 *right) {
    if (left->edge_key != right->edge_key) return left->edge_key < right->edge_key;
    if (left->request.traveler_key != right->request.traveler_key)
        return left->request.traveler_key < right->request.traveler_key;
    return left->request.request_key < right->request.request_key;
}

static int duplicate_requests(const pmg_request_v0 *requests, size_t count) {
    size_t i;
    size_t j;
    for (i = 0u; i < count; ++i) {
        for (j = i + 1u; j < count; ++j) {
            if (requests[i].request_key == requests[j].request_key ||
                requests[i].traveler_key == requests[j].traveler_key)
                return 1;
        }
    }
    return 0;
}

uint32_t pmg_admit_claims(pmg_graph_v0 *graph,
                          uint64_t claim_set_id,
                          const pmg_request_v0 *requests,
                          size_t request_count,
                          pmg_claim_result_v0 *results,
                          size_t result_capacity) {
    pmg_graph_v0 staged;
    pmg_candidate_v0 candidates[PMG_MAX_REQUESTS];
    size_t i;
    size_t j;
    uint32_t grants = 0u;
    if (graph == NULL || requests == NULL || results == NULL || claim_set_id == 0u ||
        request_count == 0u || request_count > PMG_MAX_REQUESTS ||
        result_capacity < request_count)
        return PMG_STATUS_INVALID_ARGUMENT;
    memset(results, 0, result_capacity * sizeof(*results));
    if (graph->claims_pending != 0u) return PMG_STATUS_CLAIMS_PENDING;
    if (duplicate_requests(requests, request_count)) return PMG_STATUS_DUPLICATE_REQUEST;
    for (i = 0u; i < request_count; ++i) {
        uint32_t traveler_index = pmg_find_traveler(graph, requests[i].traveler_key);
        uint32_t source_node = PMG_NODE_NONE;
        uint32_t edge_index = PMG_EDGE_NONE;
        uint64_t edge_key = UINT64_MAX;
        if (traveler_index != PMG_TRAVELER_NONE) {
            source_node = graph->travelers[traveler_index].current_node;
            edge_index = pmg_find_edge(graph, source_node, requests[i].destination_node);
            if (edge_index != PMG_EDGE_NONE) edge_key = graph->edges[edge_index].edge_key;
        }
        candidates[i].request = requests[i];
        candidates[i].source_node = source_node;
        candidates[i].edge_index = edge_index;
        candidates[i].edge_key = edge_key;
    }
    for (i = 1u; i < request_count; ++i) {
        pmg_candidate_v0 key = candidates[i];
        j = i;
        while (j > 0u && candidate_less(&key, &candidates[j - 1u])) {
            candidates[j] = candidates[j - 1u];
            --j;
        }
        candidates[j] = key;
    }
    staged = *graph;
    for (i = 0u; i < request_count; ++i) {
        const pmg_candidate_v0 *candidate = &candidates[i];
        const pmg_request_v0 *request = &candidate->request;
        pmg_claim_result_v0 *result = &results[i];
        uint32_t traveler_index = pmg_find_traveler(&staged, request->traveler_key);
        result->status = PMG_STATUS_INVALID_ARGUMENT;
        result->source_node = candidate->source_node;
        result->destination_node = request->destination_node;
        result->edge_index = candidate->edge_index;
        result->edge_key = candidate->edge_key == UINT64_MAX ? 0u : candidate->edge_key;
        result->request_key = request->request_key;
        result->traveler_key = request->traveler_key;
        result->claim_set_id = claim_set_id;
        if (request->request_key == 0u || request->traveler_key == 0u ||
            request->handoff_cause_id == 0u || request->source_sequence == 0u ||
            request->destination_sequence == 0u)
            continue;
        if (traveler_index == PMG_TRAVELER_NONE) {
            result->status = PMG_STATUS_TRAVELER_NOT_FOUND;
            continue;
        }
        if (staged.travelers[traveler_index].consumed_handoff_cause_id ==
            request->handoff_cause_id) {
            result->status = PMG_STATUS_DUPLICATE_IGNORED;
            continue;
        }
        if (request->destination_node >= staged.node_count) {
            result->status = PMG_STATUS_NODE_RANGE;
            continue;
        }
        if (candidate->edge_index == PMG_EDGE_NONE) {
            result->status = PMG_STATUS_EDGE_NOT_FOUND;
            continue;
        }
        if (staged.nodes[candidate->source_node].resident == 0u ||
            staged.nodes[request->destination_node].resident == 0u) {
            result->status = PMG_STATUS_NODE_NOT_RESIDENT;
            continue;
        }
        if (staged.nodes[candidate->source_node].cell.world.accepted_transition_id !=
                request->expected_source_local_commit_id ||
            staged.nodes[request->destination_node].cell.world.accepted_transition_id !=
                request->expected_destination_local_commit_id) {
            result->status = PMG_STATUS_STALE_VERSION;
            continue;
        }
        if (staged.edge_claim_owner[candidate->edge_index] != 0u) {
            result->status = PMG_STATUS_EDGE_CONFLICT_LOST;
            continue;
        }
        if (staged.node_claim_owner[candidate->source_node] != 0u ||
            staged.node_claim_owner[request->destination_node] != 0u) {
            result->status = PMG_STATUS_NODE_CONFLICT_LOST;
            continue;
        }
        staged.edge_claim_owner[candidate->edge_index] = request->traveler_key;
        staged.node_claim_owner[candidate->source_node] = request->traveler_key;
        staged.node_claim_owner[request->destination_node] = request->traveler_key;
        staged.travelers[traveler_index].claim_set_id = claim_set_id;
        staged.travelers[traveler_index].claim_request_key = request->request_key;
        staged.travelers[traveler_index].claim_handoff_cause_id = request->handoff_cause_id;
        staged.travelers[traveler_index].expected_source_local_commit_id =
            request->expected_source_local_commit_id;
        staged.travelers[traveler_index].expected_destination_local_commit_id =
            request->expected_destination_local_commit_id;
        staged.travelers[traveler_index].claim_edge_index = candidate->edge_index;
        staged.travelers[traveler_index].claim_destination_node = request->destination_node;
        staged.travelers[traveler_index].source_sequence = request->source_sequence;
        staged.travelers[traveler_index].destination_sequence = request->destination_sequence;
        staged.travelers[traveler_index].claim_active = 1u;
        result->status = PMG_STATUS_OK;
        grants += 1u;
    }
    if (grants != 0u) {
        staged.active_claim_set_id = claim_set_id;
        staged.claims_pending = grants;
        *graph = staged;
    }
    return PMG_STATUS_OK;
}

pmg_receipt_v0 pmg_publish_claim(pmg_graph_v0 *graph, uint64_t traveler_key) {
    pmg_graph_v0 staged;
    pmg_receipt_v0 receipt;
    pmg_traveler_v0 *traveler;
    wc_cause_v0 source_cause;
    wc_cause_v0 destination_cause;
    wc_intent_v0 source_intent;
    wc_intent_v0 destination_intent;
    pww_result_v0 source_result;
    pww_result_v0 destination_result;
    uint32_t traveler_index = pmg_find_traveler(graph, traveler_key);
    uint32_t source_node;
    uint32_t destination_node;
    uint32_t edge_index;
    uint64_t edge_key;
    uint64_t expected_destination_key;
    uint64_t next_commit_id;
    int source_actor_slot;
    int destination_actor_slot;
    uint16_t health;
    receipt = receipt_base(graph, traveler_index);
    if (graph == NULL || traveler_key == 0u) {
        receipt.status = PMG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (traveler_index == PMG_TRAVELER_NONE) {
        receipt.status = PMG_STATUS_TRAVELER_NOT_FOUND;
        return receipt;
    }
    traveler = &graph->travelers[traveler_index];
    if (traveler->claim_active == 0u) {
        receipt.status = PMG_STATUS_NO_CLAIM;
        return receipt;
    }
    source_node = traveler->current_node;
    destination_node = traveler->claim_destination_node;
    edge_index = traveler->claim_edge_index;
    edge_key = graph->edges[edge_index].edge_key;
    receipt.source_node = source_node;
    receipt.destination_node = destination_node;
    receipt.edge_index = edge_index;
    receipt.edge_key = edge_key;
    if (graph->nodes[source_node].resident == 0u ||
        graph->nodes[destination_node].resident == 0u) {
        receipt.status = PMG_STATUS_NODE_NOT_RESIDENT;
        return receipt;
    }
    if (graph->node_claim_owner[source_node] != traveler_key ||
        graph->node_claim_owner[destination_node] != traveler_key ||
        graph->edge_claim_owner[edge_index] != traveler_key) {
        receipt.status = PMG_STATUS_NO_CLAIM;
        return receipt;
    }
    if (graph->nodes[source_node].cell.world.accepted_transition_id !=
            traveler->expected_source_local_commit_id ||
        graph->nodes[destination_node].cell.world.accepted_transition_id !=
            traveler->expected_destination_local_commit_id) {
        receipt.status = PMG_STATUS_STALE_VERSION;
        return receipt;
    }
    source_actor_slot = wc_world_resolve(&graph->nodes[source_node].cell.world,
                                         traveler->actor_key);
    if (source_actor_slot < 0 ||
        graph->nodes[source_node].cell.world.kind[source_actor_slot] != WC_KIND_ACTOR) {
        receipt.status = PMG_STATUS_ACTOR_MISSING;
        return receipt;
    }
    health = graph->nodes[source_node].cell.world.actor_health[source_actor_slot];
    if (graph->graph_commit_id == UINT64_MAX) {
        receipt.status = PMG_STATUS_COMMIT_EXHAUSTED;
        return receipt;
    }
    next_commit_id = graph->graph_commit_id + 1u;
    staged = *graph;
    traveler = &staged.travelers[traveler_index];

    source_cause = wc_external_input(traveler->source_sequence,
                                     traveler->traveler_key,
                                     edge_key);
    source_intent = wc_despawn(traveler->source_sequence, traveler->actor_key);
    source_result = pww_cell_transact(&staged.nodes[source_node].cell,
                                      &source_cause, 1u,
                                      &source_intent, 1u);
    if (source_result.status != PWW_STATUS_OK) {
        receipt.status = map_source_failure(&source_result, &receipt);
        return receipt;
    }

    expected_destination_key = staged.nodes[destination_node].cell.world.next_object_key;
    destination_cause = wc_external_input(traveler->destination_sequence,
                                          traveler->traveler_key,
                                          traveler->claim_handoff_cause_id);
    destination_intent = wc_spawn_actor(traveler->destination_sequence,
                                        health, WC_NON_SPATIAL_SITE);
    destination_result = pww_cell_transact(&staged.nodes[destination_node].cell,
                                           &destination_cause, 1u,
                                           &destination_intent, 1u);
    if (destination_result.status != PWW_STATUS_OK) {
        receipt.status = map_destination_failure(&destination_result, &receipt);
        return receipt;
    }
    destination_actor_slot = wc_world_resolve(&staged.nodes[destination_node].cell.world,
                                              expected_destination_key);
    if (destination_actor_slot < 0 ||
        staged.nodes[destination_node].cell.world.kind[destination_actor_slot] != WC_KIND_ACTOR ||
        staged.nodes[destination_node].cell.world.actor_health[destination_actor_slot] != health) {
        receipt.status = PMG_STATUS_DESTINATION_SPAWN_REJECTED;
        receipt.fault_participant = PMG_PARTICIPANT_DESTINATION;
        return receipt;
    }

    staged.nodes[source_node].occupant_mask &= ~(UINT32_C(1) << traveler_index);
    staged.nodes[destination_node].occupant_mask |= UINT32_C(1) << traveler_index;
    staged.node_claim_owner[source_node] = 0u;
    staged.node_claim_owner[destination_node] = 0u;
    staged.edge_claim_owner[edge_index] = 0u;
    traveler = &staged.travelers[traveler_index];
    traveler->current_node = destination_node;
    traveler->actor_key = expected_destination_key;
    traveler->consumed_handoff_cause_id = traveler->claim_handoff_cause_id;
    traveler->claim_set_id = 0u;
    traveler->claim_request_key = 0u;
    traveler->claim_handoff_cause_id = 0u;
    traveler->expected_source_local_commit_id = 0u;
    traveler->expected_destination_local_commit_id = 0u;
    traveler->claim_edge_index = PMG_EDGE_NONE;
    traveler->claim_destination_node = PMG_NODE_NONE;
    traveler->source_sequence = 0u;
    traveler->destination_sequence = 0u;
    traveler->claim_active = 0u;
    staged.graph_commit_id = next_commit_id;
    staged.claims_pending -= 1u;
    if (staged.claims_pending == 0u) staged.active_claim_set_id = 0u;
    *graph = staged;

    receipt = receipt_base(graph, traveler_index);
    receipt.status = PMG_STATUS_OK;
    receipt.source_node = source_node;
    receipt.destination_node = destination_node;
    receipt.edge_index = edge_index;
    receipt.edge_key = edge_key;
    receipt.source_primitive = source_result.primitive;
    receipt.destination_primitive = destination_result.primitive;
    receipt.source_local_commit_id = source_result.local_commit_id;
    receipt.destination_local_commit_id = destination_result.local_commit_id;
    receipt.source_phase_fingerprint = source_result.phase_fingerprint;
    receipt.source_world_fingerprint = source_result.world_fingerprint;
    receipt.destination_phase_fingerprint = destination_result.phase_fingerprint;
    receipt.destination_world_fingerprint = destination_result.world_fingerprint;
    return receipt;
}

pmg_receipt_v0 pmg_evict_node(pmg_graph_v0 *graph, uint32_t node_index) {
    pmg_graph_v0 staged;
    pmg_receipt_v0 receipt = receipt_base(graph, PMG_TRAVELER_NONE);
    pcs_receipt_v1 snapshot_receipt;
    uint32_t status;
    if (graph == NULL) {
        receipt.status = PMG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (node_index >= graph->node_count) {
        receipt.status = PMG_STATUS_NODE_RANGE;
        return receipt;
    }
    if (graph->nodes[node_index].resident == 0u) {
        receipt.status = PMG_STATUS_NODE_NOT_RESIDENT;
        return receipt;
    }
    if (graph->nodes[node_index].occupant_mask != 0u) {
        receipt.status = PMG_STATUS_NODE_NOT_EMPTY;
        return receipt;
    }
    if (graph->node_claim_owner[node_index] != 0u) {
        receipt.status = PMG_STATUS_CLAIMS_PENDING;
        return receipt;
    }
    staged = *graph;
    status = pcs_snapshot_encode_v1(&staged.nodes[node_index].cell,
                                    staged.nodes[node_index].snapshot,
                                    sizeof(staged.nodes[node_index].snapshot),
                                    &snapshot_receipt);
    if (status != PCS_STATUS_OK) {
        receipt.status = PMG_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    memset(&staged.nodes[node_index].cell, 0, sizeof(staged.nodes[node_index].cell));
    staged.nodes[node_index].snapshot_receipt = snapshot_receipt;
    staged.nodes[node_index].snapshot_bytes = snapshot_receipt.snapshot_bytes;
    staged.nodes[node_index].resident = 0u;
    *graph = staged;
    receipt.status = PMG_STATUS_OK;
    receipt.source_node = node_index;
    return receipt;
}

pmg_receipt_v0 pmg_restore_node(pmg_graph_v0 *graph, uint32_t node_index) {
    pmg_graph_v0 staged;
    pmg_receipt_v0 receipt = receipt_base(graph, PMG_TRAVELER_NONE);
    pww_cell_v0 candidate;
    pcs_receipt_v1 decoded;
    uint32_t status;
    if (graph == NULL) {
        receipt.status = PMG_STATUS_INVALID_ARGUMENT;
        return receipt;
    }
    if (node_index >= graph->node_count) {
        receipt.status = PMG_STATUS_NODE_RANGE;
        return receipt;
    }
    if (graph->nodes[node_index].resident != 0u) {
        receipt.status = PMG_STATUS_NODE_ALREADY_RESIDENT;
        return receipt;
    }
    if (graph->nodes[node_index].snapshot_bytes == 0u) {
        receipt.status = PMG_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    memset(&candidate, 0, sizeof(candidate));
    status = pcs_snapshot_decode_v1(graph->nodes[node_index].snapshot,
                                    (size_t)graph->nodes[node_index].snapshot_bytes,
                                    &candidate, &decoded);
    if (status != PCS_STATUS_OK) {
        receipt.status = PMG_STATUS_SNAPSHOT_FAILURE;
        return receipt;
    }
    if (decoded.source_zone_key != graph->nodes[node_index].zone_key ||
        decoded.source_local_commit_id != graph->nodes[node_index].snapshot_receipt.source_local_commit_id ||
        decoded.phase_fingerprint != graph->nodes[node_index].snapshot_receipt.phase_fingerprint ||
        decoded.world_fingerprint != graph->nodes[node_index].snapshot_receipt.world_fingerprint) {
        receipt.status = PMG_STATUS_SNAPSHOT_MISMATCH;
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
    receipt.status = PMG_STATUS_OK;
    receipt.source_node = node_index;
    return receipt;
}

uint32_t pmg_provision_node_pair_limbs(pmg_graph_v0 *graph,
                                       uint32_t node_index,
                                       uint32_t new_limit) {
    if (graph == NULL || node_index >= graph->node_count ||
        graph->nodes[node_index].resident == 0u)
        return PMG_STATUS_INVALID_ARGUMENT;
    if (pww_cell_provision_pair_limbs(&graph->nodes[node_index].cell,
                                      new_limit) != PWW_STATUS_OK)
        return PMG_STATUS_INVALID_ARGUMENT;
    return PMG_STATUS_OK;
}

uint64_t pmg_graph_fingerprint(const pmg_graph_v0 *graph) {
    uint64_t hash = UINT64_C(1469598103934665603);
    uint32_t i;
    if (graph == NULL) return 0u;
    hash_feed(&hash, &graph->graph_key, sizeof(graph->graph_key));
    hash_feed(&hash, &graph->graph_commit_id, sizeof(graph->graph_commit_id));
    hash_feed(&hash, &graph->active_claim_set_id, sizeof(graph->active_claim_set_id));
    hash_feed(&hash, &graph->node_count, sizeof(graph->node_count));
    hash_feed(&hash, &graph->edge_count, sizeof(graph->edge_count));
    hash_feed(&hash, &graph->traveler_count, sizeof(graph->traveler_count));
    hash_feed(&hash, &graph->claims_pending, sizeof(graph->claims_pending));
    for (i = 0u; i < graph->edge_count; ++i) {
        hash_feed(&hash, &graph->edges[i], sizeof(graph->edges[i]));
        hash_feed(&hash, &graph->edge_claim_owner[i], sizeof(graph->edge_claim_owner[i]));
    }
    for (i = 0u; i < graph->node_count; ++i) {
        const pmg_node_v0 *node = &graph->nodes[i];
        hash_feed(&hash, &node->zone_key, sizeof(node->zone_key));
        hash_feed(&hash, &node->occupant_mask, sizeof(node->occupant_mask));
        hash_feed(&hash, &node->resident, sizeof(node->resident));
        hash_feed(&hash, &node->retained, sizeof(node->retained));
        hash_feed(&hash, &graph->node_claim_owner[i], sizeof(graph->node_claim_owner[i]));
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
    for (i = 0u; i < graph->traveler_count; ++i) {
        hash_feed(&hash, &graph->travelers[i], sizeof(graph->travelers[i]));
    }
    return hash;
}

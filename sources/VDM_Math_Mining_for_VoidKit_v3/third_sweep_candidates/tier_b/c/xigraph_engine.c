#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/pcvdm/xigraph.h"

extern uint64_t xi_graph_tick_asm(XiState *states,
                                  uint64_t count,
                                  uint64_t *carry_indices,
                                  uint64_t carry_capacity,
                                  uint64_t state_stride);

static uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }

static uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static uint64_t abs_diff_u64(uint64_t a, uint64_t b) {
    return a > b ? a - b : b - a;
}

static uint64_t phase_distance(uint64_t a, uint64_t b) {
    uint64_t da = (a & 63u);
    uint64_t db = (b & 63u);
    uint64_t d = abs_diff_u64(da, db);
    return d < (64u - d) ? d : (64u - d);
}

static int edge_id_between(const XiGraph *g, uint32_t a, uint32_t b) {
    if (a >= g->node_count || b >= g->node_count) return -1;
    const XgNodeMeta *m = &g->meta[a];
    for (uint32_t k = 0; k < m->degree; ++k) {
        if (m->adj[k].dst == b) return (int)m->adj[k].edge_id;
    }
    return -1;
}

static uint64_t compatibility_score(const XiGraph *g, uint32_t a, uint32_t b) {
    const XiState *sa = &g->states[a];
    const XiState *sb = &g->states[b];
    uint64_t score = 0;
    if (a == b || !g->meta[b].active) return 0;
    if (sa->A == sb->A) score += g->config.same_a_bonus + 32u;
    if (phase_distance(sa->theta_ticks, sb->theta_ticks) <= g->config.phase_window) score += 16u;
    if (gcd_u64(sa->uv ? sa->uv : 1u, sb->uv ? sb->uv : 1u) > 1u) score += 8u;
    if (sa->window_ready && sb->window_ready) score += 4u;
    if (edge_id_between(g, a, b) >= 0) score += 24u;
    return score;
}

static int attach_adj(XiGraph *g, uint32_t src, uint32_t dst, uint32_t edge_id) {
    XgNodeMeta *m = &g->meta[src];
    if (m->degree >= XG_MAX_DEGREE) return -1;
    m->adj[m->degree].dst = dst;
    m->adj[m->degree].edge_id = edge_id;
    m->degree++;
    return 0;
}

static int create_edge(XiGraph *g,
                       uint32_t src,
                       uint32_t dst,
                       uint64_t strength,
                       uint32_t flags) {
    if (src == dst || src >= g->node_count || dst >= g->node_count) return -1;
    if (g->edge_count >= g->edge_capacity) return -1;
    if (g->meta[src].degree >= XG_MAX_DEGREE) return -1;
    if (g->meta[dst].degree >= XG_MAX_DEGREE) return -1;

    uint32_t edge_id = (uint32_t)g->edge_count;
    XgEdge *e = &g->edges[edge_id];
    e->src = src;
    e->dst = dst;
    e->flags = flags;
    e->reserved = 0;
    e->strength = strength;
    e->phase_offset = abs_diff_u64(g->states[src].theta_ticks, g->states[dst].theta_ticks);
    e->last_transport_step = g->global_step;
    e->absorbed_debt = 0;

    if (attach_adj(g, src, dst, edge_id) != 0) return -1;
    if (attach_adj(g, dst, src, edge_id) != 0) {
        g->meta[src].degree--;
        return -1;
    }

    g->meta[src].total_edge_strength += strength;
    g->meta[dst].total_edge_strength += strength;
    g->edge_count++;
    g->edge_birth_count++;
    return (int)edge_id;
}

static uint64_t absorb_through_edge(XiGraph *g, uint32_t node, uint32_t edge_id, uint64_t debt) {
    if (edge_id >= g->edge_count || debt == 0) return 0;
    XgEdge *e = &g->edges[edge_id];
    uint64_t capacity = g->config.edge_strength_unit + (e->strength >> 3);
    uint64_t cap_max = g->config.edge_strength_unit * 4u;
    if (capacity > cap_max) capacity = cap_max;
    uint64_t absorbed = min_u64(debt, capacity);
    e->absorbed_debt += absorbed;
    e->last_transport_step = g->global_step;
    g->meta[node].debt_absorbed += absorbed;
    return absorbed;
}

static int strengthen_or_create_edge(XiGraph *g, uint32_t src, uint32_t dst, uint64_t score) {
    int existing = edge_id_between(g, src, dst);
    uint64_t delta = g->config.edge_strength_unit + score;
    if (existing >= 0) {
        XgEdge *e = &g->edges[existing];
        e->strength += delta;
        e->flags |= XG_EDGE_FLAG_REINFORCED;
        e->last_transport_step = g->global_step;
        g->meta[e->src].total_edge_strength += delta;
        g->meta[e->dst].total_edge_strength += delta;
        g->edge_reinforce_count++;
        return existing;
    }
    return create_edge(g, src, dst, delta, XG_EDGE_FLAG_SECONDARY);
}

static int pick_best_candidate(const XiGraph *g,
                               uint32_t src,
                               const uint8_t *used,
                               uint32_t *best_out,
                               uint64_t *score_out) {
    uint64_t best_score = 0;
    uint32_t best = XG_INVALID_INDEX;
    for (uint32_t j = 0; j < g->node_count; ++j) {
        if (j == src || used[j] || !g->meta[j].active) continue;
        uint64_t score = compatibility_score(g, src, j);
        if (score > best_score) {
            best_score = score;
            best = j;
        }
    }
    if (best == XG_INVALID_INDEX || best_score == 0) return -1;
    *best_out = best;
    *score_out = best_score;
    return 0;
}

static uint64_t attempt_2d_resolution(XiGraph *g, uint32_t src, uint64_t debt) {
    if (debt <= g->config.resolution_threshold || g->node_count <= 1) return debt;
    uint8_t *used = calloc((size_t)g->node_count, sizeof(uint8_t));
    if (!used) return debt;

    uint32_t attempts = g->config.max_edge_attempts;
    for (uint32_t n = 0; n < attempts && debt > g->config.resolution_threshold; ++n) {
        uint32_t dst;
        uint64_t score;
        if (pick_best_candidate(g, src, used, &dst, &score) != 0) break;
        used[dst] = 1;
        int eid = strengthen_or_create_edge(g, src, dst, score);
        if (eid < 0) continue;
        uint64_t absorbed = absorb_through_edge(g, src, (uint32_t)eid, debt);
        debt -= absorbed;
        g->meta[src].resolution_count++;
    }

    free(used);
    return debt;
}

static uint64_t node_birth_live_word(const XiGraph *g, uint32_t parent) {
    uint64_t shift = (g->states[parent].A + g->global_step + g->node_count) & 63u;
    if (shift == 63u) return (1ULL << 62u) | 1u;
    return 1ULL << shift;
}

static int birth_node(XiGraph *g, uint32_t parent, uint64_t residual_debt) {
    if (!g->config.allow_node_birth || g->node_count >= g->node_capacity) return -1;
    uint32_t child = (uint32_t)g->node_count;
    XiState *ps = &g->states[parent];
    g->states[child] = xi_state_seed(node_birth_live_word(g, parent), ps->A, ps->theta_ticks, ps->floor_den);

    XgNodeMeta *m = &g->meta[child];
    memset(m, 0, sizeof(*m));
    m->node_id = g->next_node_id++;
    m->debt = residual_debt;
    m->birth_step = g->global_step;
    m->active = 1;
    for (uint32_t k = 0; k < XG_MAX_DEGREE; ++k) {
        m->adj[k].dst = XG_INVALID_INDEX;
        m->adj[k].edge_id = XG_INVALID_INDEX;
    }

    g->node_count++;
    g->node_birth_count++;
    g->meta[parent].birth_count++;

    uint64_t primary_strength = g->config.edge_strength_unit + residual_debt + 64u;
    int primary = create_edge(g, parent, child, primary_strength, XG_EDGE_FLAG_PRIMARY | XG_EDGE_FLAG_BIRTH);
    if (primary < 0) return 0;

    for (uint32_t n = 0; n < g->config.secondary_birth_links; ++n) {
        uint32_t best = XG_INVALID_INDEX;
        uint64_t best_score = 0;
        for (uint32_t j = 0; j + 1 < g->node_count; ++j) {
            if (j == parent || edge_id_between(g, child, j) >= 0) continue;
            uint64_t score = compatibility_score(g, child, j);
            if (score > best_score) {
                best = j;
                best_score = score;
            }
        }
        if (best == XG_INVALID_INDEX || best_score == 0) break;
        (void)create_edge(g, child, best, g->config.edge_strength_unit + best_score, XG_EDGE_FLAG_SECONDARY | XG_EDGE_FLAG_BIRTH);
    }
    return 0;
}

static int handle_carry(XiGraph *g, uint32_t node) {
    if (node >= g->node_count) return -1;
    XiState *s = &g->states[node];
    XgNodeMeta *m = &g->meta[node];
    uint64_t debt = g->config.initial_debt + 1u + (s->A & 15u) + (s->uv & 7u);
    m->debt += debt;
    m->last_carry_step = g->global_step;
    m->carry_count++;
    g->carry_event_count++;

    uint64_t residual = attempt_2d_resolution(g, node, m->debt);
    m->debt = residual;
    if (residual <= g->config.resolution_threshold) {
        m->debt = 0;
        return 0;
    }
    if (birth_node(g, node, residual) == 0) {
        m->debt = 0;
        return 0;
    }
    g->unresolved_carry_count++;
    return -1;
}

XgConfig xg_default_config(void) {
    XgConfig cfg;
    cfg.floor_den = 4096;
    cfg.initial_debt = 16;
    cfg.resolution_threshold = 2;
    cfg.edge_strength_unit = 8;
    cfg.same_a_bonus = 16;
    cfg.phase_window = 8;
    cfg.max_edge_attempts = 4;
    cfg.secondary_birth_links = 2;
    cfg.allow_node_birth = 1;
    cfg.reserved = 0;
    return cfg;
}

int xg_init(XiGraph *g, uint64_t node_capacity, uint64_t edge_capacity, XgConfig cfg) {
    if (!g || node_capacity == 0 || edge_capacity == 0) return -1;
    memset(g, 0, sizeof(*g));
    g->states = calloc((size_t)node_capacity, sizeof(XiState));
    g->meta = calloc((size_t)node_capacity, sizeof(XgNodeMeta));
    g->edges = calloc((size_t)edge_capacity, sizeof(XgEdge));
    g->carry_events = calloc((size_t)node_capacity, sizeof(uint64_t));
    if (!g->states || !g->meta || !g->edges || !g->carry_events) {
        xg_free(g);
        return -1;
    }
    g->node_capacity = node_capacity;
    g->edge_capacity = edge_capacity;
    g->carry_capacity = node_capacity;
    g->next_node_id = 1;
    g->config = cfg;
    return 0;
}

void xg_free(XiGraph *g) {
    if (!g) return;
    free(g->states);
    free(g->meta);
    free(g->edges);
    free(g->carry_events);
    memset(g, 0, sizeof(*g));
}

int xg_seed_nodes(XiGraph *g, uint64_t initial_nodes) {
    if (!g || initial_nodes == 0 || initial_nodes > g->node_capacity) return -1;
    for (uint64_t i = 0; i < initial_nodes; ++i) {
        uint64_t shift = ((i * 13u) + 1u) & 62u;
        uint64_t live = 1ULL << shift;
        g->states[i] = xi_state_seed(live, i + 1u, i & 15u, g->config.floor_den);
        XgNodeMeta *m = &g->meta[i];
        memset(m, 0, sizeof(*m));
        m->node_id = g->next_node_id++;
        m->birth_step = 0;
        m->active = 1;
        for (uint32_t k = 0; k < XG_MAX_DEGREE; ++k) {
            m->adj[k].dst = XG_INVALID_INDEX;
            m->adj[k].edge_id = XG_INVALID_INDEX;
        }
    }
    g->node_count = initial_nodes;
    return 0;
}

int xg_tick(XiGraph *g) {
    if (!g || !g->states || !g->carry_events) return -1;
    uint64_t start_count = g->node_count;
    uint64_t n = xi_graph_tick_asm(g->states, start_count, g->carry_events,
                                   g->carry_capacity, sizeof(XiState));
    g->global_step++;
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t idx = g->carry_events[i];
        if (idx < start_count) (void)handle_carry(g, (uint32_t)idx);
    }
    return 0;
}

int xg_run(XiGraph *g, uint64_t ticks) {
    for (uint64_t t = 0; t < ticks; ++t) {
        if (xg_tick(g) != 0) return -1;
    }
    return 0;
}

void xg_summary(const XiGraph *g, XgSummary *out) {
    if (!g || !out) return;
    memset(out, 0, sizeof(*out));
    out->ticks = g->global_step;
    out->nodes = g->node_count;
    out->edges = g->edge_count;
    out->carries = g->carry_event_count;
    out->edge_births = g->edge_birth_count;
    out->edge_reinforcements = g->edge_reinforce_count;
    out->node_births = g->node_birth_count;
    out->unresolved_carries = g->unresolved_carry_count;
    uint64_t degree_sum = 0;
    for (uint64_t i = 0; i < g->node_count; ++i) {
        degree_sum += g->meta[i].degree;
        if (g->meta[i].degree > out->max_degree) out->max_degree = g->meta[i].degree;
    }
    out->average_degree = g->node_count ? (double)degree_sum / (double)g->node_count : 0.0;
}

int xg_validate_invariants(const XiGraph *g, FILE *err) {
    if (!g) return -1;
    int ok = 1;
    for (uint64_t i = 0; i < g->node_count; ++i) {
        const XiState *s = &g->states[i];
        const XgNodeMeta *m = &g->meta[i];
        if (!m->active || m->degree > XG_MAX_DEGREE) ok = 0;
        if (s->kappa != (s->theta_ticks >> 2)) ok = 0;
        if (s->uv != s->u * s->v) ok = 0;
        if (s->r_den != s->uv) ok = 0;
        if (s->c_den != 2u * s->uv) ok = 0;
        if (!ok && err) fprintf(err, "node invariant failed at %" PRIu64 "\n", i);
        for (uint32_t k = 0; k < m->degree; ++k) {
            if (m->adj[k].dst >= g->node_count || m->adj[k].edge_id >= g->edge_count) {
                if (err) fprintf(err, "adj invariant failed at node %" PRIu64 " slot %u\n", i, k);
                ok = 0;
            }
        }
    }
    for (uint64_t e = 0; e < g->edge_count; ++e) {
        if (g->edges[e].src >= g->node_count || g->edges[e].dst >= g->node_count) {
            if (err) fprintf(err, "edge endpoint invariant failed at edge %" PRIu64 "\n", e);
            ok = 0;
        }
    }
    return ok ? 0 : -1;
}

int xg_write_trace_csv(const XiGraph *g, const char *path) {
    if (!g || !path) return -1;
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "node_index,node_id,step,A,theta_ticks,kappa,u,v,uv,carry_count,debt,debt_absorbed,degree,total_edge_strength,birth_step,last_carry_step,birth_count\n");
    for (uint64_t i = 0; i < g->node_count; ++i) {
        const XiState *s = &g->states[i];
        const XgNodeMeta *m = &g->meta[i];
        fprintf(f,
                "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu32 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                i, m->node_id, s->step, s->A, s->theta_ticks, s->kappa, s->u, s->v, s->uv,
                m->carry_count, m->debt, m->debt_absorbed, m->degree, m->total_edge_strength,
                m->birth_step, m->last_carry_step, m->birth_count);
    }
    fclose(f);
    return 0;
}

int xg_write_summary_json(const XiGraph *g, const char *path) {
    if (!g || !path) return -1;
    XgSummary s;
    xg_summary(g, &s);
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "{\n");
    fprintf(f, "  \"ticks\": %" PRIu64 ",\n", s.ticks);
    fprintf(f, "  \"nodes\": %" PRIu64 ",\n", s.nodes);
    fprintf(f, "  \"edges\": %" PRIu64 ",\n", s.edges);
    fprintf(f, "  \"carries\": %" PRIu64 ",\n", s.carries);
    fprintf(f, "  \"edge_births\": %" PRIu64 ",\n", s.edge_births);
    fprintf(f, "  \"edge_reinforcements\": %" PRIu64 ",\n", s.edge_reinforcements);
    fprintf(f, "  \"node_births\": %" PRIu64 ",\n", s.node_births);
    fprintf(f, "  \"unresolved_carries\": %" PRIu64 ",\n", s.unresolved_carries);
    fprintf(f, "  \"max_degree\": %" PRIu64 ",\n", s.max_degree);
    fprintf(f, "  \"average_degree\": %.6f\n", s.average_degree);
    fprintf(f, "}\n");
    fclose(f);
    return 0;
}


int xg_write_edges_csv(const XiGraph *g, const char *path) {
    if (!g || !path) return -1;
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "edge_id,src,dst,flags,strength,phase_offset,last_transport_step,absorbed_debt\n");
    for (uint64_t i = 0; i < g->edge_count; ++i) {
        const XgEdge *e = &g->edges[i];
        fprintf(f,
                "%" PRIu64 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
                i, e->src, e->dst, e->flags, e->strength, e->phase_offset,
                e->last_transport_step, e->absorbed_debt);
    }
    fclose(f);
    return 0;
}

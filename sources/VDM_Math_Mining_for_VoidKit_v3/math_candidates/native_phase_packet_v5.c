#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

/* Minimal MPFR declarations to avoid requiring mpfr headers at compile-time. */
typedef long mpfr_prec_t;
typedef long mpfr_exp_t;
typedef unsigned long mp_limb_t;
typedef struct {
    mpfr_prec_t _mpfr_prec;
    int _mpfr_sign;
    mpfr_exp_t _mpfr_exp;
    mp_limb_t *_mpfr_d;
} __mpfr_struct;
typedef __mpfr_struct mpfr_t[1];
typedef __mpfr_struct * mpfr_ptr;
typedef const __mpfr_struct * mpfr_srcptr;
typedef int mpfr_rnd_t;

#define MPFR_RNDN 0
#define MPFR_RNDD 3
#define EXPORT __attribute__((visibility("default")))

void mpfr_init2(mpfr_ptr x, mpfr_prec_t prec);
void mpfr_clear(mpfr_ptr x);
int mpfr_set_d(mpfr_ptr rop, double op, mpfr_rnd_t rnd);
int mpfr_set_ui(mpfr_ptr rop, unsigned long op, mpfr_rnd_t rnd);
int mpfr_set(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_add(mpfr_ptr rop, mpfr_srcptr op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_sub(mpfr_ptr rop, mpfr_srcptr op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_mul(mpfr_ptr rop, mpfr_srcptr op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_mul_ui(mpfr_ptr rop, mpfr_srcptr op1, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_div(mpfr_ptr rop, mpfr_srcptr op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_div_ui(mpfr_ptr rop, mpfr_srcptr op1, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_sub_ui(mpfr_ptr rop, mpfr_srcptr op1, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_ui_div(mpfr_ptr rop, unsigned long op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_exp(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_log(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_neg(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_abs(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_cmp(mpfr_srcptr op1, mpfr_srcptr op2);
int mpfr_sqr(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
double mpfr_get_d(mpfr_srcptr op, mpfr_rnd_t rnd);
unsigned long mpfr_get_ui(mpfr_srcptr op, mpfr_rnd_t rnd);
size_t mpfr_snprintf(char *str, size_t size, const char *format, ...);

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef struct {
    unsigned stage_dps;
    unsigned q_terms;
    unsigned r_terms;
    unsigned safe_digits;
    double stage_seconds;
    char abs_update[128];
    char abs_residual[128];
    char error_bound[128];
} IterRow;

typedef struct {
    unsigned target_digits;
    unsigned final_safe_digits;
    unsigned stages;
    unsigned q_terms_final;
    unsigned r_terms_final;
    double total_seconds;
} Summary;

typedef struct {
    mpfr_t q, q2, r;
    mpfr_t P, S1, S2;
    mpfr_t tminus, tplus, p1, p2;
    mpfr_t tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
    mpfr_t F, Fp, Fpp;
    mpfr_t tol, ln2half, slope_floor, ln10;
} Ctx;

typedef struct {
    unsigned dps;
    mpfr_prec_t bits;
    Ctx ctx;
} Stage;

typedef struct {
    int initialized;
    unsigned target_digits;
    unsigned nstages;
    unsigned sched[16];
    Stage stages[16];
    mpfr_prec_t final_bits;
    mpfr_t x_master;
    mpfr_t abs_update;
    mpfr_t bound;
    double x0;
} Pipeline;

static Pipeline g_pipe = {0};

static mpfr_prec_t dps_to_bits(unsigned dps) {
    return (mpfr_prec_t)ceil(((double)dps + 8.0) * 3.32192809488736234787) + 32;
}

static void ctx_init(Ctx *c, mpfr_prec_t prec, unsigned tol_digits) {
#define INIT(x) mpfr_init2(c->x, prec)
    INIT(q); INIT(q2); INIT(r);
    INIT(P); INIT(S1); INIT(S2);
    INIT(tminus); INIT(tplus); INIT(p1); INIT(p2);
    INIT(tmp1); INIT(tmp2); INIT(tmp3); INIT(tmp4); INIT(tmp5); INIT(tmp6); INIT(tmp7); INIT(tmp8);
    INIT(F); INIT(Fp); INIT(Fpp);
    INIT(tol); INIT(ln2half); INIT(slope_floor); INIT(ln10);
#undef INIT
    mpfr_set_ui(c->tmp1, 2, MPFR_RNDN);
    mpfr_log(c->ln2half, c->tmp1, MPFR_RNDN);
    mpfr_div_ui(c->ln2half, c->ln2half, 2, MPFR_RNDN);
    mpfr_set_d(c->slope_floor, 0.125, MPFR_RNDN);
    mpfr_set_ui(c->ln10, 10, MPFR_RNDN);
    mpfr_log(c->ln10, c->ln10, MPFR_RNDN);
    mpfr_set_ui(c->tol, 1, MPFR_RNDN);
    for (unsigned i = 0; i < tol_digits; ++i) {
        mpfr_div_ui(c->tol, c->tol, 10, MPFR_RNDN);
    }
}

static void ctx_clear(Ctx *c) {
#define CLR(x) mpfr_clear(c->x)
    CLR(q); CLR(q2); CLR(r);
    CLR(P); CLR(S1); CLR(S2);
    CLR(tminus); CLR(tplus); CLR(p1); CLR(p2);
    CLR(tmp1); CLR(tmp2); CLR(tmp3); CLR(tmp4); CLR(tmp5); CLR(tmp6); CLR(tmp7); CLR(tmp8);
    CLR(F); CLR(Fp); CLR(Fpp);
    CLR(tol); CLR(ln2half); CLR(slope_floor); CLR(ln10);
#undef CLR
}

static unsigned pentagonal_eval(mpfr_ptr q, mpfr_ptr P, mpfr_ptr S1, mpfr_ptr S2, mpfr_ptr tol,
                                mpfr_ptr tminus, mpfr_ptr tplus, mpfr_ptr p1, mpfr_ptr p2,
                                mpfr_ptr q2, mpfr_ptr tmp1, mpfr_ptr tmp2, mpfr_ptr tmp3, mpfr_ptr tmp4) {
    mpfr_set_ui(P, 1, MPFR_RNDN);
    mpfr_set_ui(S1, 0, MPFR_RNDN);
    mpfr_set_ui(S2, 0, MPFR_RNDN);
    mpfr_set(tminus, q, MPFR_RNDN);
    mpfr_mul(q2, q, q, MPFR_RNDN);
    mpfr_set(tplus, q2, MPFR_RNDN);
    mpfr_mul(p1, q2, q, MPFR_RNDN);
    mpfr_set(p2, q2, MPFR_RNDN);

    unsigned k = 1;
    while (1) {
        mpfr_abs(tmp1, tminus, MPFR_RNDN);
        mpfr_abs(tmp2, tplus, MPFR_RNDN);
        if (mpfr_cmp(tmp1, tol) < 0 && mpfr_cmp(tmp2, tol) < 0) break;
        unsigned long kk = (unsigned long)k;
        unsigned long a1 = kk * (3ul * kk - 1ul) / 2ul;
        unsigned long a2 = a1 + kk;
        int sign = (k & 1u) ? -1 : 1;

        mpfr_add(tmp1, tminus, tplus, MPFR_RNDN);
        if (sign > 0) mpfr_add(P, P, tmp1, MPFR_RNDN);
        else mpfr_sub(P, P, tmp1, MPFR_RNDN);

        mpfr_mul_ui(tmp1, tminus, a1, MPFR_RNDN);
        mpfr_mul_ui(tmp2, tplus, a2, MPFR_RNDN);
        mpfr_add(tmp3, tmp1, tmp2, MPFR_RNDN);
        if (sign > 0) mpfr_add(S1, S1, tmp3, MPFR_RNDN);
        else mpfr_sub(S1, S1, tmp3, MPFR_RNDN);

        mpfr_mul_ui(tmp1, tminus, a1 * a1, MPFR_RNDN);
        mpfr_mul_ui(tmp2, tplus, a2 * a2, MPFR_RNDN);
        mpfr_add(tmp3, tmp1, tmp2, MPFR_RNDN);
        if (sign > 0) mpfr_add(S2, S2, tmp3, MPFR_RNDN);
        else mpfr_sub(S2, S2, tmp3, MPFR_RNDN);

        mpfr_mul(tmp3, tplus, p1, MPFR_RNDN);
        mpfr_mul(tmp4, tmp3, p2, MPFR_RNDN);
        mpfr_set(tminus, tmp3, MPFR_RNDN);
        mpfr_set(tplus, tmp4, MPFR_RNDN);
        mpfr_mul(p1, p1, q2, MPFR_RNDN);
        mpfr_mul(p2, p2, q, MPFR_RNDN);
        ++k;
        if (k > 1000000u) return 0u;
    }
    return k - 1u;
}

static void packet_eval_N2(Ctx *c, mpfr_ptr x, unsigned *q_terms, unsigned *r_terms) {
    mpfr_neg(c->q, x, MPFR_RNDN);
    mpfr_exp(c->q, c->q, MPFR_RNDN);

    mpfr_mul_ui(c->tmp1, x, 4, MPFR_RNDN);
    mpfr_neg(c->r, c->tmp1, MPFR_RNDN);
    mpfr_exp(c->r, c->r, MPFR_RNDN);

    *q_terms = pentagonal_eval(c->q, c->P, c->S1, c->S2, c->tol,
                               c->tminus, c->tplus, c->p1, c->p2,
                               c->q2, c->tmp1, c->tmp2, c->tmp3, c->tmp4);
    mpfr_set(c->tmp5, c->P, MPFR_RNDN);
    mpfr_set(c->tmp6, c->S1, MPFR_RNDN);
    mpfr_set(c->tmp7, c->S2, MPFR_RNDN);

    *r_terms = pentagonal_eval(c->r, c->P, c->S1, c->S2, c->tol,
                               c->tminus, c->tplus, c->p1, c->p2,
                               c->q2, c->tmp1, c->tmp2, c->tmp3, c->tmp4);

    mpfr_log(c->F, c->tmp5, MPFR_RNDN);
    mpfr_neg(c->F, c->F, MPFR_RNDN);
    mpfr_add(c->F, c->F, c->ln2half, MPFR_RNDN);
    mpfr_div_ui(c->tmp1, x, 8, MPFR_RNDN);
    mpfr_sub(c->F, c->F, c->tmp1, MPFR_RNDN);
    mpfr_log(c->tmp1, c->P, MPFR_RNDN);
    mpfr_add(c->F, c->F, c->tmp1, MPFR_RNDN);

    mpfr_div(c->Fp, c->tmp6, c->tmp5, MPFR_RNDN);
    mpfr_sub(c->Fp, c->Fp, c->slope_floor, MPFR_RNDN);
    mpfr_div(c->tmp1, c->S1, c->P, MPFR_RNDN);
    mpfr_mul_ui(c->tmp1, c->tmp1, 4, MPFR_RNDN);
    mpfr_sub(c->Fp, c->Fp, c->tmp1, MPFR_RNDN);

    mpfr_div(c->Fpp, c->tmp7, c->tmp5, MPFR_RNDN);
    mpfr_div(c->tmp1, c->tmp6, c->tmp5, MPFR_RNDN);
    mpfr_sqr(c->tmp1, c->tmp1, MPFR_RNDN);
    mpfr_sub(c->Fpp, c->Fpp, c->tmp1, MPFR_RNDN);
    mpfr_neg(c->Fpp, c->Fpp, MPFR_RNDN);
    mpfr_div(c->tmp2, c->S2, c->P, MPFR_RNDN);
    mpfr_div(c->tmp3, c->S1, c->P, MPFR_RNDN);
    mpfr_sqr(c->tmp3, c->tmp3, MPFR_RNDN);
    mpfr_sub(c->tmp2, c->tmp2, c->tmp3, MPFR_RNDN);
    mpfr_mul_ui(c->tmp2, c->tmp2, 16, MPFR_RNDN);
    mpfr_add(c->Fpp, c->Fpp, c->tmp2, MPFR_RNDN);
}

static void halley_update(Ctx *c, mpfr_ptr x, mpfr_ptr abs_update_out) {
    mpfr_mul(c->tmp1, c->F, c->Fp, MPFR_RNDN);
    mpfr_mul_ui(c->tmp1, c->tmp1, 2, MPFR_RNDN);
    mpfr_sqr(c->tmp2, c->Fp, MPFR_RNDN);
    mpfr_mul_ui(c->tmp2, c->tmp2, 2, MPFR_RNDN);
    mpfr_mul(c->tmp3, c->F, c->Fpp, MPFR_RNDN);
    mpfr_sub(c->tmp2, c->tmp2, c->tmp3, MPFR_RNDN);
    mpfr_div(c->tmp1, c->tmp1, c->tmp2, MPFR_RNDN);
    mpfr_sub(x, x, c->tmp1, MPFR_RNDN);
    if (abs_update_out) mpfr_abs(abs_update_out, c->tmp1, MPFR_RNDN);
}

static unsigned safe_digits_from_bound(Ctx *c, mpfr_ptr bound) {
    if (bound->_mpfr_sign == 0) return 1000000000u;
    mpfr_mul_ui(c->tmp1, bound, 2, MPFR_RNDN);
    mpfr_log(c->tmp1, c->tmp1, MPFR_RNDN);
    mpfr_neg(c->tmp1, c->tmp1, MPFR_RNDN);
    mpfr_div(c->tmp1, c->tmp1, c->ln10, MPFR_RNDN);
    double v = mpfr_get_d(c->tmp1, MPFR_RNDN);
    if (v < 0.0) return 0u;
    return (unsigned)floor(v);
}

static void bound_from_residual(Ctx *c, mpfr_ptr residual_abs, mpfr_ptr bound_out) {
    mpfr_div(bound_out, residual_abs, c->slope_floor, MPFR_RNDN);
}

static void pentagonal_double(double q, double *P, double *S1, double *S2) {
    double p = 1.0, s1 = 0.0, s2 = 0.0;
    double tminus = q;
    double q2 = q * q;
    double tplus = q2;
    double p1 = q2 * q;
    double p2 = q2;
    unsigned k = 1;
    const double tol = 1e-18;
    while (fabs(tminus) >= tol || fabs(tplus) >= tol) {
        unsigned a1 = k * (3u * k - 1u) / 2u;
        unsigned a2 = a1 + k;
        double sign = (k & 1u) ? -1.0 : 1.0;
        p += sign * (tminus + tplus);
        s1 += sign * (a1 * tminus + a2 * tplus);
        s2 += sign * ((double)a1 * (double)a1 * tminus + (double)a2 * (double)a2 * tplus);
        double next_minus = tplus * p1;
        double next_plus = next_minus * p2;
        tminus = next_minus;
        tplus = next_plus;
        p1 *= q2;
        p2 *= q;
        ++k;
        if (k > 1000000u) break;
    }
    *P = p;
    *S1 = s1;
    *S2 = s2;
}

static double starter_double(void) {
    double x = 3.0;
    for (int i = 0; i < 3; ++i) {
        double q = exp(-x), r = exp(-4.0 * x);
        double Pq, Sq1, Sq2, Pr, Sr1, Sr2;
        pentagonal_double(q, &Pq, &Sq1, &Sq2);
        pentagonal_double(r, &Pr, &Sr1, &Sr2);
        double F = 0.5 * log(2.0) - log(Pq) - x / 8.0 + log(Pr);
        double Fp = (Sq1 / Pq) - 0.125 - 4.0 * (Sr1 / Pr);
        double Fpp = -(Sq2 / Pq - (Sq1 / Pq) * (Sq1 / Pq)) + 16.0 * (Sr2 / Pr - (Sr1 / Pr) * (Sr1 / Pr));
        double dx = 2.0 * F * Fp / (2.0 * Fp * Fp - F * Fpp);
        x -= dx;
    }
    return x;
}

static unsigned build_schedule(unsigned target_digits, unsigned *sched, unsigned max_sched) {
    unsigned final_dps = target_digits + 20u;
    unsigned n = 0;
    unsigned d = 20u;
    while (d < final_dps && n < max_sched - 1u) {
        sched[n++] = d;
        d *= 3u;
    }
    sched[n++] = final_dps;
    return n;
}

static int mpfr_to_trunc_decimal(mpfr_ptr x, unsigned digits, char *outbuf, size_t outlen) {
    if (outlen < (size_t)digits + 4u) return 5;
    mpfr_prec_t bits = x->_mpfr_prec;
    mpfr_t frac;
    mpfr_init2(frac, bits);
    unsigned long intpart = mpfr_get_ui(x, MPFR_RNDD);
    mpfr_set(frac, x, MPFR_RNDN);
    mpfr_sub_ui(frac, frac, intpart, MPFR_RNDN);
    size_t pos = 0;
    if (intpart >= 10u) {
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%lu", intpart);
        size_t n = strlen(tmp);
        memcpy(outbuf + pos, tmp, n);
        pos += n;
    } else {
        outbuf[pos++] = (char)('0' + intpart);
    }
    outbuf[pos++] = '.';
    for (unsigned i = 0; i < digits; ++i) {
        mpfr_mul_ui(frac, frac, 10u, MPFR_RNDN);
        unsigned long d = mpfr_get_ui(frac, MPFR_RNDD);
        outbuf[pos++] = (char)('0' + d);
        mpfr_sub_ui(frac, frac, d, MPFR_RNDN);
    }
    outbuf[pos] = '\0';
    mpfr_clear(frac);
    return 0;
}

static void pipeline_clear(Pipeline *p) {
    if (!p->initialized) return;
    for (unsigned i = 0; i < p->nstages; ++i) {
        ctx_clear(&p->stages[i].ctx);
    }
    mpfr_clear(p->x_master);
    mpfr_clear(p->abs_update);
    mpfr_clear(p->bound);
    memset(p, 0, sizeof(*p));
}

static int pipeline_prepare(Pipeline *p, unsigned target_digits) {
    if (p->initialized && p->target_digits == target_digits) return 0;
    pipeline_clear(p);
    p->target_digits = target_digits;
    p->nstages = build_schedule(target_digits, p->sched, 16);
    p->final_bits = dps_to_bits(p->sched[p->nstages - 1u]);
    mpfr_init2(p->x_master, p->final_bits);
    mpfr_init2(p->abs_update, p->final_bits);
    mpfr_init2(p->bound, p->final_bits);
    p->x0 = starter_double();
    for (unsigned i = 0; i < p->nstages; ++i) {
        p->stages[i].dps = p->sched[i];
        p->stages[i].bits = dps_to_bits(p->sched[i]);
        ctx_init(&p->stages[i].ctx, p->stages[i].bits, p->sched[i] + 8u);
    }
    p->initialized = 1;
    return 0;
}

static int compute_fast_pipeline(Pipeline *p, char *outbuf, size_t outlen, double *seconds_out,
                                 unsigned *safe_digits_out, unsigned *q_terms_out, unsigned *r_terms_out) {
    double t0 = now_sec();
    mpfr_set_d(p->x_master, p->x0, MPFR_RNDN);

    unsigned q_terms = 0u, r_terms = 0u;
    for (unsigned i = 0; i < p->nstages; ++i) {
        Ctx *ctx = &p->stages[i].ctx;
        mpfr_set(ctx->tmp8, p->x_master, MPFR_RNDN);
        packet_eval_N2(ctx, ctx->tmp8, &q_terms, &r_terms);
        halley_update(ctx, ctx->tmp8, NULL);
        mpfr_set(p->x_master, ctx->tmp8, MPFR_RNDN);
        if (i == p->nstages - 1u) {
            packet_eval_N2(ctx, ctx->tmp8, &q_terms, &r_terms);
            mpfr_abs(ctx->tmp1, ctx->F, MPFR_RNDN);
            bound_from_residual(ctx, ctx->tmp1, p->bound);
            if (safe_digits_out) *safe_digits_out = safe_digits_from_bound(ctx, p->bound);
            if (q_terms_out) *q_terms_out = q_terms;
            if (r_terms_out) *r_terms_out = r_terms;
            mpfr_to_trunc_decimal(ctx->tmp8, p->target_digits, outbuf, outlen);
        }
    }
    if (seconds_out) *seconds_out = now_sec() - t0;
    return 0;
}

static int compute_trace_pipeline(Pipeline *p, char *outbuf, size_t outlen, IterRow *rows, unsigned max_rows, Summary *summary) {
    if (max_rows < p->nstages) return 2;
    double total_t0 = now_sec();
    mpfr_set_d(p->x_master, p->x0, MPFR_RNDN);

    unsigned final_safe = 0u, q_terms_final = 0u, r_terms_final = 0u;
    for (unsigned i = 0; i < p->nstages; ++i) {
        Ctx *ctx = &p->stages[i].ctx;
        mpfr_set(ctx->tmp8, p->x_master, MPFR_RNDN);
        double st0 = now_sec();
        unsigned tq = 0u, tr = 0u;
        packet_eval_N2(ctx, ctx->tmp8, &tq, &tr);
        halley_update(ctx, ctx->tmp8, p->abs_update);
        packet_eval_N2(ctx, ctx->tmp8, &tq, &tr);
        mpfr_abs(ctx->tmp1, ctx->F, MPFR_RNDN);
        mpfr_set(ctx->tmp4, ctx->tmp1, MPFR_RNDN);
        bound_from_residual(ctx, ctx->tmp1, p->bound);
        mpfr_set(ctx->tmp5, p->bound, MPFR_RNDN);
        unsigned safe = safe_digits_from_bound(ctx, p->bound);
        double st1 = now_sec();

        rows[i].stage_dps = p->stages[i].dps;
        rows[i].q_terms = tq;
        rows[i].r_terms = tr;
        rows[i].safe_digits = safe;
        rows[i].stage_seconds = st1 - st0;
        mpfr_snprintf(rows[i].abs_update, sizeof(rows[i].abs_update), "%.18RNe", p->abs_update);
        mpfr_snprintf(rows[i].abs_residual, sizeof(rows[i].abs_residual), "%.18RNe", ctx->tmp4);
        mpfr_snprintf(rows[i].error_bound, sizeof(rows[i].error_bound), "%.18RNe", ctx->tmp5);

        mpfr_set(p->x_master, ctx->tmp8, MPFR_RNDN);
        if (i == p->nstages - 1u) {
            final_safe = safe;
            q_terms_final = tq;
            r_terms_final = tr;
            mpfr_to_trunc_decimal(ctx->tmp8, p->target_digits, outbuf, outlen);
        }
    }

    if (summary) {
        summary->target_digits = p->target_digits;
        summary->final_safe_digits = final_safe;
        summary->stages = p->nstages;
        summary->q_terms_final = q_terms_final;
        summary->r_terms_final = r_terms_final;
        summary->total_seconds = now_sec() - total_t0;
    }
    return 0;
}

EXPORT int phase_native_pi_fast_v5(unsigned target_digits, char *outbuf, size_t outlen,
                                   double *seconds_out, unsigned *safe_digits_out,
                                   unsigned *q_terms_out, unsigned *r_terms_out) {
    pipeline_prepare(&g_pipe, target_digits);
    return compute_fast_pipeline(&g_pipe, outbuf, outlen, seconds_out, safe_digits_out, q_terms_out, r_terms_out);
}

EXPORT int phase_native_pi_trace_v5(unsigned target_digits, char *outbuf, size_t outlen,
                                    IterRow *rows, unsigned max_rows, Summary *summary) {
    pipeline_prepare(&g_pipe, target_digits);
    return compute_trace_pipeline(&g_pipe, outbuf, outlen, rows, max_rows, summary);
}

EXPORT int phase_native_pi_benchmark_v5(unsigned target_digits, unsigned reps,
                                        double *min_seconds, double *mean_seconds) {
    if (reps == 0u) return 3;
    pipeline_prepare(&g_pipe, target_digits);
    char *buf = (char *)malloc((size_t)target_digits + 32u);
    if (!buf) return 4;
    double best = 1e300, sum = 0.0;
    unsigned safe = 0u, q_terms = 0u, r_terms = 0u;
    for (unsigned i = 0; i < reps; ++i) {
        double secs = 0.0;
        compute_fast_pipeline(&g_pipe, buf, (size_t)target_digits + 32u, &secs, &safe, &q_terms, &r_terms);
        if (secs < best) best = secs;
        sum += secs;
    }
    free(buf);
    if (min_seconds) *min_seconds = best;
    if (mean_seconds) *mean_seconds = sum / (double)reps;
    return 0;
}

EXPORT void phase_native_pi_pipeline_reset_v5(void) {
    pipeline_clear(&g_pipe);
}

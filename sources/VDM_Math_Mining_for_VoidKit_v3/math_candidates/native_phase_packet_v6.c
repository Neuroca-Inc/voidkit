#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

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
int mpfr_sqrt(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
unsigned long mpfr_get_ui(mpfr_srcptr op, mpfr_rnd_t rnd);
size_t mpfr_snprintf(char *str, size_t size, const char *format, ...);

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static mpfr_prec_t dps_to_bits(unsigned dps) {
    return (mpfr_prec_t)ceil(((double)dps + 8.0) * 3.32192809488736234787) + 48;
}

static unsigned agm_iterations_for_digits(unsigned digits) {
    if (digits <= 520u) return 11u;
    if (digits <= 1000u) return 12u;
    if (digits <= 10000u) return 16u;
    return 18u;
}

/* -------- Hot native collapse path: packet collapse -> AGM -------- */
typedef struct {
    int initialized;
    unsigned target_digits;
    unsigned iters;
    mpfr_prec_t bits;
    mpfr_t sqrt2, a, b, t, p, an, bn, tn, tmp1, tmp2, frac, x;
} AgmCache;

static AgmCache g_agm = {0};

static void agm_clear(AgmCache *c) {
    if (!c->initialized) return;
    mpfr_clear(c->sqrt2); mpfr_clear(c->a); mpfr_clear(c->b); mpfr_clear(c->t); mpfr_clear(c->p);
    mpfr_clear(c->an); mpfr_clear(c->bn); mpfr_clear(c->tn); mpfr_clear(c->tmp1); mpfr_clear(c->tmp2);
    mpfr_clear(c->frac); mpfr_clear(c->x);
    memset(c, 0, sizeof(*c));
}

static void agm_prepare(AgmCache *c, unsigned target_digits) {
    if (c->initialized && c->target_digits == target_digits) return;
    agm_clear(c);
    c->target_digits = target_digits;
    c->iters = agm_iterations_for_digits(target_digits);
    c->bits = dps_to_bits(target_digits + 24u);
    mpfr_init2(c->sqrt2, c->bits); mpfr_init2(c->a, c->bits); mpfr_init2(c->b, c->bits);
    mpfr_init2(c->t, c->bits); mpfr_init2(c->p, c->bits); mpfr_init2(c->an, c->bits);
    mpfr_init2(c->bn, c->bits); mpfr_init2(c->tn, c->bits); mpfr_init2(c->tmp1, c->bits);
    mpfr_init2(c->tmp2, c->bits); mpfr_init2(c->frac, c->bits); mpfr_init2(c->x, c->bits);
    mpfr_set_ui(c->tmp1, 2u, MPFR_RNDN);
    mpfr_sqrt(c->sqrt2, c->tmp1, MPFR_RNDN);
    c->initialized = 1;
}

static void agm_compute_pi(AgmCache *c) {
    mpfr_set_ui(c->a, 1u, MPFR_RNDN);
    mpfr_ui_div(c->b, 1u, c->sqrt2, MPFR_RNDN);
    mpfr_set_d(c->t, 0.25, MPFR_RNDN);
    mpfr_set_ui(c->p, 1u, MPFR_RNDN);
    for (unsigned i = 0; i < c->iters; ++i) {
        mpfr_add(c->an, c->a, c->b, MPFR_RNDN);
        mpfr_div_ui(c->an, c->an, 2u, MPFR_RNDN);
        mpfr_mul(c->bn, c->a, c->b, MPFR_RNDN);
        mpfr_sqrt(c->bn, c->bn, MPFR_RNDN);
        mpfr_sub(c->tmp1, c->a, c->an, MPFR_RNDN);
        mpfr_sqr(c->tmp1, c->tmp1, MPFR_RNDN);
        mpfr_mul(c->tmp1, c->p, c->tmp1, MPFR_RNDN);
        mpfr_sub(c->tn, c->t, c->tmp1, MPFR_RNDN);
        mpfr_mul_ui(c->p, c->p, 2u, MPFR_RNDN);
        mpfr_set(c->a, c->an, MPFR_RNDN);
        mpfr_set(c->b, c->bn, MPFR_RNDN);
        mpfr_set(c->t, c->tn, MPFR_RNDN);
    }
    mpfr_add(c->tmp1, c->a, c->b, MPFR_RNDN);
    mpfr_sqr(c->tmp1, c->tmp1, MPFR_RNDN);
    mpfr_mul_ui(c->tmp2, c->t, 4u, MPFR_RNDN);
    mpfr_div(c->x, c->tmp1, c->tmp2, MPFR_RNDN);
}

static int mpfr_to_trunc_decimal(mpfr_ptr x, unsigned digits, mpfr_ptr frac, char *outbuf, size_t outlen) {
    if (outlen < (size_t)digits + 4u) return 5;
    unsigned long intpart = mpfr_get_ui(x, MPFR_RNDD);
    mpfr_set(frac, x, MPFR_RNDN);
    mpfr_sub_ui(frac, frac, intpart, MPFR_RNDN);
    size_t pos = 0u;
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
    for (unsigned i = 0u; i < digits; ++i) {
        mpfr_mul_ui(frac, frac, 10u, MPFR_RNDN);
        unsigned long d = mpfr_get_ui(frac, MPFR_RNDD);
        outbuf[pos++] = (char)('0' + d);
        mpfr_sub_ui(frac, frac, d, MPFR_RNDN);
    }
    outbuf[pos] = '\0';
    return 0;
}

EXPORT int phase_native_pi_hot_v6(unsigned target_digits, char *outbuf, size_t outlen,
                                  double *seconds_out, unsigned *iters_out) {
    agm_prepare(&g_agm, target_digits);
    double t0 = now_sec();
    agm_compute_pi(&g_agm);
    int rc = mpfr_to_trunc_decimal(g_agm.x, target_digits, g_agm.frac, outbuf, outlen);
    if (seconds_out) *seconds_out = now_sec() - t0;
    if (iters_out) *iters_out = g_agm.iters;
    return rc;
}

EXPORT int phase_native_pi_hot_benchmark_v6(unsigned target_digits, unsigned reps,
                                            double *min_seconds, double *mean_seconds) {
    if (reps == 0u) return 3;
    agm_prepare(&g_agm, target_digits);
    char *buf = (char *)malloc((size_t)target_digits + 32u);
    if (!buf) return 4;
    /* warm */
    phase_native_pi_hot_v6(target_digits, buf, (size_t)target_digits + 32u, NULL, NULL);
    double best = 1e300, sum = 0.0;
    for (unsigned i = 0u; i < reps; ++i) {
        double secs = 0.0;
        phase_native_pi_hot_v6(target_digits, buf, (size_t)target_digits + 32u, &secs, NULL);
        if (secs < best) best = secs;
        sum += secs;
    }
    free(buf);
    if (min_seconds) *min_seconds = best;
    if (mean_seconds) *mean_seconds = sum / (double)reps;
    return 0;
}

/* -------- Legacy current-law certificate path (N = 2 packet law) -------- */
typedef struct {
    mpfr_t q, q2, r, P, S1, S2, tminus, tplus, p1, p2;
    mpfr_t tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, F, Fp, Fpp, tol, ln2half, slope_floor, ln10;
} PacketCtx;

typedef struct {
    int initialized;
    unsigned target_digits;
    mpfr_prec_t bits;
    PacketCtx ctx;
    mpfr_t x_cache, bound;
} CertCache;

static CertCache g_cert = {0};

static void packet_ctx_init(PacketCtx *c, mpfr_prec_t bits, unsigned tol_digits) {
#define INIT(x) mpfr_init2(c->x, bits)
    INIT(q); INIT(q2); INIT(r); INIT(P); INIT(S1); INIT(S2); INIT(tminus); INIT(tplus); INIT(p1); INIT(p2);
    INIT(tmp1); INIT(tmp2); INIT(tmp3); INIT(tmp4); INIT(tmp5); INIT(tmp6); INIT(F); INIT(Fp); INIT(Fpp); INIT(tol); INIT(ln2half); INIT(slope_floor); INIT(ln10);
#undef INIT
    mpfr_set_ui(c->tmp1, 2u, MPFR_RNDN);
    mpfr_log(c->ln2half, c->tmp1, MPFR_RNDN);
    mpfr_div_ui(c->ln2half, c->ln2half, 2u, MPFR_RNDN);
    mpfr_set_d(c->slope_floor, 0.125, MPFR_RNDN);
    mpfr_set_ui(c->ln10, 10u, MPFR_RNDN);
    mpfr_log(c->ln10, c->ln10, MPFR_RNDN);
    mpfr_set_ui(c->tol, 1u, MPFR_RNDN);
    for (unsigned i = 0u; i < tol_digits; ++i) mpfr_div_ui(c->tol, c->tol, 10u, MPFR_RNDN);
}

static void packet_ctx_clear(PacketCtx *c) {
#define CLR(x) mpfr_clear(c->x)
    CLR(q); CLR(q2); CLR(r); CLR(P); CLR(S1); CLR(S2); CLR(tminus); CLR(tplus); CLR(p1); CLR(p2);
    CLR(tmp1); CLR(tmp2); CLR(tmp3); CLR(tmp4); CLR(tmp5); CLR(tmp6); CLR(F); CLR(Fp); CLR(Fpp); CLR(tol); CLR(ln2half); CLR(slope_floor); CLR(ln10);
#undef CLR
}

static void cert_clear(CertCache *c) {
    if (!c->initialized) return;
    packet_ctx_clear(&c->ctx);
    mpfr_clear(c->x_cache);
    mpfr_clear(c->bound);
    memset(c, 0, sizeof(*c));
}

static void cert_prepare(CertCache *c, unsigned target_digits) {
    if (c->initialized && c->target_digits == target_digits) return;
    cert_clear(c);
    c->target_digits = target_digits;
    c->bits = dps_to_bits(target_digits + 20u);
    packet_ctx_init(&c->ctx, c->bits, target_digits + 8u);
    mpfr_init2(c->x_cache, c->bits);
    mpfr_init2(c->bound, c->bits);
    c->initialized = 1;
}

static unsigned pentagonal_eval(PacketCtx *c, mpfr_ptr q, mpfr_ptr P, mpfr_ptr S1, mpfr_ptr S2) {
    mpfr_set_ui(P, 1u, MPFR_RNDN);
    mpfr_set_ui(S1, 0u, MPFR_RNDN);
    mpfr_set_ui(S2, 0u, MPFR_RNDN);
    mpfr_set(c->tminus, q, MPFR_RNDN);
    mpfr_mul(c->q2, q, q, MPFR_RNDN);
    mpfr_set(c->tplus, c->q2, MPFR_RNDN);
    mpfr_mul(c->p1, c->q2, q, MPFR_RNDN);
    mpfr_set(c->p2, c->q2, MPFR_RNDN);
    unsigned k = 1u;
    while (1) {
        mpfr_abs(c->tmp1, c->tminus, MPFR_RNDN);
        mpfr_abs(c->tmp2, c->tplus, MPFR_RNDN);
        if (mpfr_cmp(c->tmp1, c->tol) < 0 && mpfr_cmp(c->tmp2, c->tol) < 0) break;
        unsigned long kk = (unsigned long)k;
        unsigned long a1 = kk * (3ul * kk - 1ul) / 2ul;
        unsigned long a2 = a1 + kk;
        int sign = (k & 1u) ? -1 : 1;

        mpfr_add(c->tmp1, c->tminus, c->tplus, MPFR_RNDN);
        if (sign > 0) mpfr_add(P, P, c->tmp1, MPFR_RNDN); else mpfr_sub(P, P, c->tmp1, MPFR_RNDN);

        mpfr_mul_ui(c->tmp1, c->tminus, a1, MPFR_RNDN);
        mpfr_mul_ui(c->tmp2, c->tplus, a2, MPFR_RNDN);
        mpfr_add(c->tmp3, c->tmp1, c->tmp2, MPFR_RNDN);
        if (sign > 0) mpfr_add(S1, S1, c->tmp3, MPFR_RNDN); else mpfr_sub(S1, S1, c->tmp3, MPFR_RNDN);

        mpfr_mul_ui(c->tmp1, c->tminus, a1 * a1, MPFR_RNDN);
        mpfr_mul_ui(c->tmp2, c->tplus, a2 * a2, MPFR_RNDN);
        mpfr_add(c->tmp3, c->tmp1, c->tmp2, MPFR_RNDN);
        if (sign > 0) mpfr_add(S2, S2, c->tmp3, MPFR_RNDN); else mpfr_sub(S2, S2, c->tmp3, MPFR_RNDN);

        mpfr_mul(c->tmp3, c->tplus, c->p1, MPFR_RNDN);
        mpfr_mul(c->tmp4, c->tmp3, c->p2, MPFR_RNDN);
        mpfr_set(c->tminus, c->tmp3, MPFR_RNDN);
        mpfr_set(c->tplus, c->tmp4, MPFR_RNDN);
        mpfr_mul(c->p1, c->p1, c->q2, MPFR_RNDN);
        mpfr_mul(c->p2, c->p2, q, MPFR_RNDN);
        ++k;
        if (k > 1000000u) return 0u;
    }
    return k - 1u;
}

static void packet_eval_N2(PacketCtx *c, mpfr_ptr x, unsigned *q_terms, unsigned *r_terms) {
    mpfr_neg(c->q, x, MPFR_RNDN);
    mpfr_exp(c->q, c->q, MPFR_RNDN);
    mpfr_mul_ui(c->tmp1, x, 4u, MPFR_RNDN);
    mpfr_neg(c->r, c->tmp1, MPFR_RNDN);
    mpfr_exp(c->r, c->r, MPFR_RNDN);

    *q_terms = pentagonal_eval(c, c->q, c->P, c->S1, c->S2);
    mpfr_set(c->tmp5, c->P, MPFR_RNDN);
    mpfr_set(c->tmp6, c->S1, MPFR_RNDN);

    *r_terms = pentagonal_eval(c, c->r, c->P, c->S1, c->S2);

    mpfr_log(c->F, c->tmp5, MPFR_RNDN);
    mpfr_neg(c->F, c->F, MPFR_RNDN);
    mpfr_add(c->F, c->F, c->ln2half, MPFR_RNDN);
    mpfr_div_ui(c->tmp1, x, 8u, MPFR_RNDN);
    mpfr_sub(c->F, c->F, c->tmp1, MPFR_RNDN);
    mpfr_log(c->tmp1, c->P, MPFR_RNDN);
    mpfr_add(c->F, c->F, c->tmp1, MPFR_RNDN);

    mpfr_abs(c->tmp1, c->F, MPFR_RNDN);
    mpfr_div(c->tmp2, c->tmp1, c->slope_floor, MPFR_RNDN);
}

static unsigned safe_digits_from_bound(PacketCtx *c, mpfr_ptr bound) {
    if (bound->_mpfr_sign == 0) return 1000000000u;
    mpfr_mul_ui(c->tmp3, bound, 2u, MPFR_RNDN);
    mpfr_log(c->tmp3, c->tmp3, MPFR_RNDN);
    mpfr_neg(c->tmp3, c->tmp3, MPFR_RNDN);
    mpfr_div(c->tmp3, c->tmp3, c->ln10, MPFR_RNDN);
    double v = 0.0; /* mpfr_get_d not declared/needed for integer-ish floor */
    char buf[64];
    mpfr_snprintf(buf, sizeof(buf), "%.18Rf", c->tmp3);
    v = atof(buf);
    if (v < 0.0) return 0u;
    return (unsigned)floor(v);
}

EXPORT int phase_current_law_cert_v6(unsigned target_digits, unsigned *safe_digits_out,
                                     double *seconds_out, unsigned *q_terms_out, unsigned *r_terms_out) {
    agm_prepare(&g_agm, target_digits);
    cert_prepare(&g_cert, target_digits);
    agm_compute_pi(&g_agm);
    mpfr_set(g_cert.x_cache, g_agm.x, MPFR_RNDN);
    double t0 = now_sec();
    unsigned tq = 0u, tr = 0u;
    packet_eval_N2(&g_cert.ctx, g_cert.x_cache, &tq, &tr);
    mpfr_abs(g_cert.ctx.tmp1, g_cert.ctx.F, MPFR_RNDN);
    mpfr_div(g_cert.bound, g_cert.ctx.tmp1, g_cert.ctx.slope_floor, MPFR_RNDN);
    unsigned safe = safe_digits_from_bound(&g_cert.ctx, g_cert.bound);
    if (seconds_out) *seconds_out = now_sec() - t0;
    if (safe_digits_out) *safe_digits_out = safe;
    if (q_terms_out) *q_terms_out = tq;
    if (r_terms_out) *r_terms_out = tr;
    return 0;
}

EXPORT int phase_current_law_cert_benchmark_v6(unsigned target_digits, unsigned reps,
                                               double *min_seconds, double *mean_seconds) {
    if (reps == 0u) return 3;
    agm_prepare(&g_agm, target_digits);
    cert_prepare(&g_cert, target_digits);
    agm_compute_pi(&g_agm);
    mpfr_set(g_cert.x_cache, g_agm.x, MPFR_RNDN);
    /* warm */
    unsigned tq = 0u, tr = 0u;
    packet_eval_N2(&g_cert.ctx, g_cert.x_cache, &tq, &tr);
    double best = 1e300, sum = 0.0;
    for (unsigned i = 0u; i < reps; ++i) {
        double t0 = now_sec();
        packet_eval_N2(&g_cert.ctx, g_cert.x_cache, &tq, &tr);
        mpfr_abs(g_cert.ctx.tmp1, g_cert.ctx.F, MPFR_RNDN);
        mpfr_div(g_cert.bound, g_cert.ctx.tmp1, g_cert.ctx.slope_floor, MPFR_RNDN);
        (void)safe_digits_from_bound(&g_cert.ctx, g_cert.bound);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
        sum += dt;
    }
    if (min_seconds) *min_seconds = best;
    if (mean_seconds) *mean_seconds = sum / (double)reps;
    return 0;
}

EXPORT int phase_native_pi_hot_with_cert_benchmark_v6(unsigned target_digits, unsigned reps,
                                                       double *min_seconds, double *mean_seconds) {
    if (reps == 0u) return 3;
    agm_prepare(&g_agm, target_digits);
    cert_prepare(&g_cert, target_digits);
    char *buf = (char *)malloc((size_t)target_digits + 32u);
    if (!buf) return 4;
    phase_native_pi_hot_v6(target_digits, buf, (size_t)target_digits + 32u, NULL, NULL);
    phase_current_law_cert_v6(target_digits, NULL, NULL, NULL, NULL);
    double best = 1e300, sum = 0.0;
    for (unsigned i = 0u; i < reps; ++i) {
        double t0 = now_sec();
        phase_native_pi_hot_v6(target_digits, buf, (size_t)target_digits + 32u, NULL, NULL);
        mpfr_set(g_cert.x_cache, g_agm.x, MPFR_RNDN);
        unsigned tq = 0u, tr = 0u;
        packet_eval_N2(&g_cert.ctx, g_cert.x_cache, &tq, &tr);
        mpfr_abs(g_cert.ctx.tmp1, g_cert.ctx.F, MPFR_RNDN);
        mpfr_div(g_cert.bound, g_cert.ctx.tmp1, g_cert.ctx.slope_floor, MPFR_RNDN);
        (void)safe_digits_from_bound(&g_cert.ctx, g_cert.bound);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
        sum += dt;
    }
    free(buf);
    if (min_seconds) *min_seconds = best;
    if (mean_seconds) *mean_seconds = sum / (double)reps;
    return 0;
}

EXPORT void phase_native_reset_v6(void) {
    agm_clear(&g_agm);
    cert_clear(&g_cert);
}

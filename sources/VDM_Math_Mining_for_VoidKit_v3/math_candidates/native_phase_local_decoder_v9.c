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
int mpfr_mul_2ui(mpfr_ptr rop, mpfr_srcptr op1, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_div(mpfr_ptr rop, mpfr_srcptr op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_div_ui(mpfr_ptr rop, mpfr_srcptr op1, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_ui_div(mpfr_ptr rop, unsigned long op1, mpfr_srcptr op2, mpfr_rnd_t rnd);
int mpfr_sub_ui(mpfr_ptr rop, mpfr_srcptr op1, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_sqr(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_sqrt(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_pow_ui(mpfr_ptr rop, mpfr_srcptr op, unsigned long op2, mpfr_rnd_t rnd);
int mpfr_frac(mpfr_ptr rop, mpfr_srcptr op, mpfr_rnd_t rnd);
int mpfr_const_pi(mpfr_ptr rop, mpfr_rnd_t rnd);
unsigned long mpfr_get_ui(mpfr_srcptr op, mpfr_rnd_t rnd);

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static mpfr_prec_t decimal_digits_to_bits(unsigned digits) {
    return (mpfr_prec_t)ceil(((double)digits + 16.0) * 3.32192809488736234787) + 64;
}

static double coarse_bound_log10(unsigned iters) {
    const double LOG10_2 = 0.30102999566398119521;
    const double INV_LN10 = 0.43429448190325182765;
    return ((double)iters + 8.0) * LOG10_2 - 3.0 * (double)(1u << (iters + 1u)) * INV_LN10;
}

static unsigned agm_iterations_for_decimal_digits(unsigned target_digits) {
    const unsigned required = target_digits + 8u;
    for (unsigned iters = 1u; iters < 32u; ++iters) {
        if (-coarse_bound_log10(iters) > (double)required) return iters;
    }
    return 31u;
}

typedef struct {
    int initialized;
    unsigned required_decimal_digits;
    unsigned probe_digits;
    unsigned iters;
    mpfr_prec_t bits;
    double bound_log10;
    mpfr_t sqrt2, a, b, t, p, an, bn, tn, tmp1, tmp2, pi;
} LocalState;

static LocalState g_state = {0};

static void clear_state(LocalState *s) {
    if (!s->initialized) return;
    mpfr_clear(s->sqrt2);
    mpfr_clear(s->a);
    mpfr_clear(s->b);
    mpfr_clear(s->t);
    mpfr_clear(s->p);
    mpfr_clear(s->an);
    mpfr_clear(s->bn);
    mpfr_clear(s->tn);
    mpfr_clear(s->tmp1);
    mpfr_clear(s->tmp2);
    mpfr_clear(s->pi);
    memset(s, 0, sizeof(*s));
}

static void prepare_state(LocalState *s, unsigned required_decimal_digits, unsigned probe_digits) {
    if (s->initialized && s->required_decimal_digits == required_decimal_digits && s->probe_digits == probe_digits) return;
    clear_state(s);
    s->required_decimal_digits = required_decimal_digits;
    s->probe_digits = probe_digits;
    s->iters = agm_iterations_for_decimal_digits(required_decimal_digits + probe_digits + 8u);
    s->bits = decimal_digits_to_bits(required_decimal_digits);
    s->bound_log10 = coarse_bound_log10(s->iters);
    mpfr_init2(s->sqrt2, s->bits);
    mpfr_init2(s->a, s->bits);
    mpfr_init2(s->b, s->bits);
    mpfr_init2(s->t, s->bits);
    mpfr_init2(s->p, s->bits);
    mpfr_init2(s->an, s->bits);
    mpfr_init2(s->bn, s->bits);
    mpfr_init2(s->tn, s->bits);
    mpfr_init2(s->tmp1, s->bits);
    mpfr_init2(s->tmp2, s->bits);
    mpfr_init2(s->pi, s->bits);
    mpfr_set_ui(s->tmp1, 2u, MPFR_RNDN);
    mpfr_sqrt(s->sqrt2, s->tmp1, MPFR_RNDN);
    s->initialized = 1;
}

static void compute_pi(LocalState *s) {
    mpfr_set_ui(s->a, 1u, MPFR_RNDN);
    mpfr_ui_div(s->b, 1u, s->sqrt2, MPFR_RNDN);
    mpfr_set_d(s->t, 0.25, MPFR_RNDN);
    mpfr_set_ui(s->p, 1u, MPFR_RNDN);
    for (unsigned i = 0; i < s->iters; ++i) {
        mpfr_add(s->an, s->a, s->b, MPFR_RNDN);
        mpfr_div_ui(s->an, s->an, 2u, MPFR_RNDN);
        mpfr_mul(s->bn, s->a, s->b, MPFR_RNDN);
        mpfr_sqrt(s->bn, s->bn, MPFR_RNDN);
        mpfr_sub(s->tmp1, s->a, s->an, MPFR_RNDN);
        mpfr_sqr(s->tmp1, s->tmp1, MPFR_RNDN);
        mpfr_mul(s->tmp1, s->p, s->tmp1, MPFR_RNDN);
        mpfr_sub(s->tn, s->t, s->tmp1, MPFR_RNDN);
        mpfr_mul_ui(s->p, s->p, 2u, MPFR_RNDN);
        mpfr_set(s->a, s->an, MPFR_RNDN);
        mpfr_set(s->b, s->bn, MPFR_RNDN);
        mpfr_set(s->t, s->tn, MPFR_RNDN);
    }
    mpfr_add(s->tmp1, s->a, s->b, MPFR_RNDN);
    mpfr_sqr(s->tmp1, s->tmp1, MPFR_RNDN);
    mpfr_mul_ui(s->tmp2, s->t, 4u, MPFR_RNDN);
    mpfr_div(s->pi, s->tmp1, s->tmp2, MPFR_RNDN);
}

static int is_power_of_two_base(unsigned base, unsigned *shift_per_digit) {
    if (base < 2u) return 0;
    unsigned x = base;
    unsigned shift = 0u;
    while ((x & 1u) == 0u) {
        x >>= 1u;
        ++shift;
    }
    if (x != 1u) return 0;
    *shift_per_digit = shift;
    return 1;
}

static char digit_char(unsigned d) {
    return (d < 10u) ? (char)('0' + d) : (char)('A' + (d - 10u));
}

static int orient_fraction(mpfr_srcptr pi_value,
                           mpfr_prec_t bits,
                           unsigned base,
                           unsigned start,
                           mpfr_ptr frac_out) {
    mpfr_t scaled, factor;
    mpfr_init2(scaled, bits);
    if (start <= 1u) {
        mpfr_frac(frac_out, pi_value, MPFR_RNDD);
        mpfr_clear(scaled);
        return 0;
    }
    unsigned shift_per_digit = 0u;
    if (is_power_of_two_base(base, &shift_per_digit)) {
        unsigned long shift = (unsigned long)shift_per_digit * (unsigned long)(start - 1u);
        mpfr_mul_2ui(scaled, pi_value, shift, MPFR_RNDD);
        mpfr_frac(frac_out, scaled, MPFR_RNDD);
        mpfr_clear(scaled);
        return 0;
    }
    mpfr_init2(factor, bits);
    mpfr_set_ui(factor, (unsigned long)base, MPFR_RNDN);
    mpfr_pow_ui(factor, factor, (unsigned long)(start - 1u), MPFR_RNDN);
    mpfr_mul(scaled, pi_value, factor, MPFR_RNDD);
    mpfr_frac(frac_out, scaled, MPFR_RNDD);
    mpfr_clear(factor);
    mpfr_clear(scaled);
    return 0;
}

static int emit_block(mpfr_ptr frac,
                      unsigned base,
                      unsigned length,
                      unsigned probe_digits,
                      char *outbuf,
                      size_t outlen,
                      char *probebuf,
                      size_t probelen,
                      unsigned *first_nonmax_pos_out) {
    if (outlen < (size_t)length + 1u) return 10;
    if (probe_digits > 0u && probelen < (size_t)probe_digits + 1u) return 11;
    unsigned first_nonmax = 0u;
    for (unsigned i = 0; i < length + probe_digits; ++i) {
        mpfr_mul_ui(frac, frac, (unsigned long)base, MPFR_RNDD);
        unsigned long d = mpfr_get_ui(frac, MPFR_RNDD);
        if (d >= (unsigned long)base) d = (unsigned long)base - 1u;
        mpfr_sub_ui(frac, frac, d, MPFR_RNDD);
        char ch = digit_char((unsigned)d);
        if (i < length) {
            outbuf[i] = ch;
        } else {
            unsigned j = i - length;
            probebuf[j] = ch;
            if (first_nonmax == 0u && d < (unsigned long)(base - 1u)) first_nonmax = j + 1u;
        }
    }
    outbuf[length] = '\0';
    if (probe_digits > 0u) probebuf[probe_digits] = '\0';
    if (first_nonmax_pos_out) *first_nonmax_pos_out = first_nonmax;
    return 0;
}

static int query_from_value(mpfr_srcptr pi_value,
                            mpfr_prec_t bits,
                            unsigned base,
                            unsigned start,
                            unsigned length,
                            unsigned probe_digits,
                            char *outbuf,
                            size_t outlen,
                            char *probebuf,
                            size_t probelen,
                            unsigned *first_nonmax_pos_out) {
    mpfr_t frac;
    mpfr_init2(frac, bits);
    orient_fraction(pi_value, bits, base, start, frac);
    int rc = emit_block(frac, base, length, probe_digits, outbuf, outlen, probebuf, probelen, first_nonmax_pos_out);
    mpfr_clear(frac);
    return rc;
}

EXPORT int phase_local_prepare_v9(unsigned required_decimal_digits,
                                  unsigned probe_digits,
                                  double *seconds_out,
                                  unsigned *iters_out,
                                  double *bound_log10_out) {
    double t0 = now_sec();
    prepare_state(&g_state, required_decimal_digits, probe_digits);
    compute_pi(&g_state);
    if (seconds_out) *seconds_out = now_sec() - t0;
    if (iters_out) *iters_out = g_state.iters;
    if (bound_log10_out) *bound_log10_out = g_state.bound_log10;
    return 0;
}

EXPORT int phase_local_query_v9(unsigned base,
                                unsigned start,
                                unsigned length,
                                unsigned probe_digits,
                                char *outbuf,
                                size_t outlen,
                                char *probebuf,
                                size_t probelen,
                                double *seconds_out,
                                unsigned *first_nonmax_pos_out,
                                int *cert_ok_out,
                                unsigned *safe_length_lower_bound_out) {
    if (!g_state.initialized) return 20;
    double t0 = now_sec();
    unsigned first_nonmax = 0u;
    int rc = query_from_value(g_state.pi, g_state.bits, base, start, length, probe_digits,
                              outbuf, outlen, probebuf, probelen, &first_nonmax);
    if (rc != 0) return rc;
    if (seconds_out) *seconds_out = now_sec() - t0;
    if (first_nonmax_pos_out) *first_nonmax_pos_out = first_nonmax;
    double log10b = log10((double)base);
    double oriented_bound_log10 = g_state.bound_log10 + ((double)(start - 1u)) * log10b;
    unsigned safe_len = 0u;
    int ok = 0;
    if (first_nonmax > 0u) {
        double safe = (-oriented_bound_log10 / log10b) - (double)first_nonmax;
        safe_len = (safe > 0.0) ? (unsigned)floor(safe) : 0u;
        ok = (safe_len >= length) ? 1 : 0;
    }
    if (cert_ok_out) *cert_ok_out = ok;
    if (safe_length_lower_bound_out) *safe_length_lower_bound_out = safe_len;
    return 0;
}

EXPORT int phase_reference_query_v9(unsigned base,
                                    unsigned start,
                                    unsigned length,
                                    unsigned probe_digits,
                                    char *outbuf,
                                    size_t outlen,
                                    char *probebuf,
                                    size_t probelen) {
    double need = ((double)(start - 1u + length + probe_digits + 12u)) * log10((double)base);
    unsigned required_decimal_digits = (unsigned)ceil(need) + 8u;
    mpfr_prec_t bits = decimal_digits_to_bits(required_decimal_digits);
    mpfr_t pi;
    mpfr_init2(pi, bits);
    mpfr_const_pi(pi, MPFR_RNDN);
    int rc = query_from_value(pi, bits, base, start, length, probe_digits,
                              outbuf, outlen, probebuf, probelen, NULL);
    mpfr_clear(pi);
    return rc;
}

EXPORT int phase_local_query_benchmark_v9(unsigned base,
                                          unsigned start,
                                          unsigned length,
                                          unsigned probe_digits,
                                          unsigned reps,
                                          double *min_seconds_out,
                                          double *mean_seconds_out) {
    if (!g_state.initialized || reps == 0u) return 30;
    char *out = (char *)malloc((size_t)length + 2u);
    char *probe = (char *)malloc((size_t)probe_digits + 2u);
    if (!out || !probe) {
        free(out); free(probe);
        return 31;
    }
    double best = 1e300;
    double sum = 0.0;
    for (unsigned i = 0u; i < reps; ++i) {
        double secs = 0.0;
        phase_local_query_v9(base, start, length, probe_digits, out, (size_t)length + 2u,
                             probe, (size_t)probe_digits + 2u, &secs, NULL, NULL, NULL);
        if (secs < best) best = secs;
        sum += secs;
    }
    free(out); free(probe);
    if (min_seconds_out) *min_seconds_out = best;
    if (mean_seconds_out) *mean_seconds_out = sum / (double)reps;
    return 0;
}

EXPORT void phase_local_reset_v9(void) {
    clear_state(&g_state);
}

/*
 * mfcc_extract.c  --  MFCC feature extraction for TMS320F28379D
 *
 * Same features as the baseline version (original/mfcc_extract.c): mean
 * removal, Hann window, power spectrum, mel filterbank, 10*log10 with a
 * global top_db floor, DCT, z-score, clip. What changed is how the work is
 * done, because the baseline spent most of its 203 ms on overhead around the
 * arithmetic rather than on the arithmetic:
 *
 *   1. Real-input FFT at half size. The N real samples of a frame are packed
 *      as N/2 complex values z[n] = x[2n] + j*x[2n+1]; one N/2-point complex
 *      FFT and a short unpacking step give the N/2+1 bins of the real
 *      spectrum. The baseline ran an N-point complex FFT with the imaginary
 *      half set to zero.
 *   2. Radix-4. Two radix-2 stages are done in one pass over the buffer, so
 *      a value is loaded and stored once where radix-2 does it twice, and the
 *      four outputs of a butterfly need 3 complex multiplies instead of 4.
 *      With optimisation off, loads and stores are most of the cost.
 *   3. The first pass does window, complex packing, bit reversal AND the
 *      first two FFT stages together: it reads eight ADC codes from their
 *      bit-reversed places, windows them, and writes four finished values in
 *      order. Those two stages have twiddles 1 and -j, so no multiplies.
 *   4. Twiddle factors come from tables, laid out per stage in the order
 *      they are used. The baseline derived each one from the previous with
 *      four multiplies per butterfly, and called cosf/sinf once per stage
 *      per frame.
 *   5. Every loop walks pointers. No index is computed per butterfly.
 *   6. The float copy of the whole slice (x[], 16896 floats = 33792 words of
 *      RAM, and a pass to fill it) is gone. The ADC code is converted and
 *      the mean removed while windowing.
 *   7. The power spectrum overwrites the FFT buffer, and the unpacking step
 *      produces bins k and N/2-k together from one pair of FFT outputs.
 *
 * RAM (16-bit words, N = 512): z 1028, twiddle tables 1024 + 1024,
 * bit-reversal 64, log_mel 5200. The baseline used 33792 + 2048 + 5200.
 *
 * The tables are built on the FIRST call (about a thousand cosf/sinf), so
 * the first call is much slower than the rest. Time the second call onwards.
 *
 * Numerically this is the same transform computed in a different order, so
 * results agree with the baseline to rounding, not bit for bit. Two details
 * are deliberately more exact than the baseline: the mean is an integer sum
 * (the baseline's float sum of 16896 codes rounds), and the twiddles are
 * read from a table rather than accumulated. See pc_test/run_mfcc.sh.
 *
 *   8. 10*log10() is computed here instead of calling the library's
 *      log10f(), which is by far the most expensive single operation left
 *      (FEAT_N_FRAMES * FEAT_N_MELS calls per slice). See MFCC_FAST_LOG.
 *
 * int is 16 bits on the C28x: every index here stays below 32768
 * (FEAT_SLICE_LEN = 16896), and sums that would not are uint32_t.
 */

#include <math.h>
#include <stdint.h>
#include "feature_tables.h"

#define FFT_N   FEAT_N_FFT              /* real frame length              */
#define FFT_M   (FEAT_N_FFT / 2)        /* size of the complex FFT        */

#if (FFT_N < 32) || (FFT_N & (FFT_N - 1))
#error "FEAT_N_FFT must be a power of two, 32 or larger"
#endif

/*
 * MFCC_FAST_LOG  1 (default)  10*log10(x) from the float's own exponent plus
 *                             a short series for the mantissa (power_db()).
 *                0            call log10f(), as the baseline did.
 * Build with both and time them to see what the library call was costing.
 */
#ifndef MFCC_FAST_LOG
#define MFCC_FAST_LOG 1
#endif

static float log_mel[FEAT_N_FRAMES][FEAT_N_MELS];

/* FFT work buffer: FFT_M complex values, interleaved (re, im). After
 * unpacking, the power of bin k sits in z[2*k], k = 0 .. FFT_M (hence +2). */
static float z[2 * FFT_M + 2];

/* tw[2k] = cos(2*pi*k/N), tw[2k+1] = sin(2*pi*k/N), k = 0 .. FFT_M-1.
 * Used by the unpacking step, and by the single radix-2 stage that is left
 * over when FFT_M is not a power of 4. */
static float tw[2 * FFT_M];

/* Radix-4 twiddles, one block per stage, in the order the stage reads them.
 * For the stage that builds groups of 4*h from groups of h, butterfly k
 * (k = 0 .. h-1) uses six floats: cos,sin of a, 2a, 3a with a = 2*pi*k/(4h).
 * Stages are h = 4, 16, 64, ...; the blocks total less than 2*FFT_M floats. */
static float tw4[2 * FFT_M];

/* rev4[i] = 2 * bitreverse(4*i): the sample index, within the frame, of the
 * real part of the first of the four complex inputs that end up at
 * z[8*i ..]. The other three sit FFT_M, FFT_M/2 and 3*FFT_M/2 samples on. */
static uint16_t rev4[FFT_M / 4];

static int tables_ready = 0;

static void build_tables(void)
{
    const float two_pi = 2.0f * 3.14159265358979f;
    float *t;
    uint16_t k, n, bit, r, h;

    for (k = 0; k < FFT_M; k++) {
        float ang = two_pi * (float)k / (float)FFT_N;
        tw[2 * k]     = cosf(ang);
        tw[2 * k + 1] = sinf(ang);
    }

    t = tw4;
    for (h = 4; 4 * h <= FFT_M; h <<= 2) {
        for (k = 0; k < h; k++) {
            float ang = two_pi * (float)k / (float)(4 * h);
            *t++ = cosf(ang);         *t++ = sinf(ang);
            *t++ = cosf(2.0f * ang);  *t++ = sinf(2.0f * ang);
            *t++ = cosf(3.0f * ang);  *t++ = sinf(3.0f * ang);
        }
    }

    for (n = 0; n < FFT_M / 4; n++) {
        r = 0;
        k = 4 * n;
        for (bit = FFT_M >> 1; bit != 0; bit >>= 1) {
            if (k & 1) r |= bit;
            k >>= 1;
        }
        rev4[n] = 2 * r;
    }
    tables_ready = 1;
}

#if MFCC_FAST_LOG
/*
 * 10*log10(x) for a normal, positive float (the caller has already clamped
 * x to FEAT_AMIN, so it is never zero, negative or denormal).
 *
 *   x = m * 2^e, m in [1, 2)        read straight from the IEEE-754 bits
 *   m folded into [0.707, 1.414)    so that t below stays small
 *   t = (m - 1) / (m + 1),  |t| <= 0.1716
 *   ln(m) = 2*(t + t^3/3 + t^5/5 + t^7/7)
 *
 * The first dropped term is 2*t^9/9 < 3e-8, i.e. below 1.5e-7 dB -- under
 * the rounding of a float holding a value near 100 dB (about 8e-6 dB), so
 * this is as exact as the type allows. One division, six multiplies.
 */
static float power_db(float x)
{
    union { float f; uint32_t u; } b;
    float m, t, t2, ln_m;
    int16_t e;

    b.f = x;
    e = (int16_t)((b.u >> 23) & 0xFFu) - 127;
    b.u = (b.u & 0x007FFFFFuL) | 0x3F800000uL;      /* exponent := 0 -> [1, 2) */
    m = b.f;
    if (m > 1.41421356f) {
        m *= 0.5f;
        e += 1;
    }
    t  = (m - 1.0f) / (m + 1.0f);
    t2 = t * t;
    ln_m = 2.0f * t * (1.0f + t2 * (0.333333333f + t2 * (0.2f + t2 * 0.142857143f)));
    /* 10*log10(2) * e  +  10/ln(10) * ln(m) */
    return 3.01029995664f * (float)e + 4.34294481903f * ln_m;
}
#define POWER_DB(x)   power_db(x)
#else
#define POWER_DB(x)   (10.0f * log10f(x))
#endif

/*
 * One frame: s points at FFT_N ADC codes. On return z[2*k] holds
 *     4 * FEAT_ADC_SCALE^2 * |X[k]|^2,   k = 0 .. FFT_M
 * where X is the spectrum the baseline computed. The constant factor is a
 * power of two and is taken out once per mel band by the caller, which is
 * exact.
 */
static void frame_power(const unsigned int *s, float mean)
{
    register float *p;
    register float *q;
    register const float *t;
    float *p2, *p3;
    const unsigned int *sp;
    const float *hp;
    const uint16_t *r = rev4;
    float ar, ai, br, bi, cr, ci, dr, di;
    float ur, ui, vr, vi, wr, wi;
    uint16_t n, g, k, h, h2;

    /* windowed sample i of the group being read */
#define WS(i)   (((float)sp[i] - mean) * hp[i])

    /*
     * Pass 1: window + pack + bit-reverse + radix-2 stages 1 and 2.
     * a, b, c, d are the complex inputs that bit reversal puts at positions
     * 4i, 4i+1, 4i+2, 4i+3. Stage 1 pairs (a,b) and (c,d) with twiddle 1;
     * stage 2 pairs the results with twiddles 1 and -j.
     */
    p = z;
    for (n = FFT_M / 4; n != 0; n--) {
        sp = s + *r;
        hp = feat_hann_window + *r;
        r++;
        ar = WS(0);                 ai = WS(1);
        br = WS(FFT_M);             bi = WS(FFT_M + 1);
        cr = WS(FFT_M / 2);         ci = WS(FFT_M / 2 + 1);
        dr = WS(3 * FFT_M / 2);     di = WS(3 * FFT_M / 2 + 1);
        ur = ar + br;  ui = ai + bi;                /* a + b                  */
        vr = cr + dr;  vi = ci + di;                /* c + d                  */
        p[0] = ur + vr;  p[1] = ui + vi;
        p[4] = ur - vr;  p[5] = ui - vi;
        ur = ar - br;  ui = ai - bi;                /* a - b                  */
        vr = cr - dr;  vi = ci - di;                /* c - d                  */
        p[2] = ur + vi;  p[3] = ui - vr;            /* (a-b) - j*(c-d)        */
        p[6] = ur - vi;  p[7] = ui + vr;            /* (a-b) + j*(c-d)        */
        p += 8;
    }
#undef WS

    /*
     * Radix-4 passes. Each builds groups of 4*h from groups of h. In a
     * group, butterfly k takes the four values h apart (x0..x3) and, with
     * w = exp(-j*2*pi*k/(4h)):
     *
     *   y1 = w^2 * x1      y2 = w * x2      y3 = w^3 * x3
     *   x0' = (x0 + y1) + (y2 + y3)       x2' = (x0 + y1) - (y2 + y3)
     *   x1' = (x0 - y1) - j*(y2 - y3)     x3' = (x0 - y1) + j*(y2 - y3)
     *
     * which is exactly the two radix-2 stages of lengths 2h and 4h.
     * tw4 holds (cos, sin) of w, w^2, w^3 for k = 0..h-1, stage after stage.
     */
    t = tw4;
    for (h = 4; 4 * h <= FFT_M; h <<= 2) {
        const float *t0 = t;                        /* this stage's block     */
        h2 = 2 * h;                                 /* floats per quarter     */
        p = z;
        for (g = FFT_M / (4 * h); g != 0; g--) {
            q  = p + h2;                            /* x1                     */
            p2 = q + h2;                            /* x2                     */
            p3 = p2 + h2;                           /* x3                     */
            t = t0;
            for (k = h; k != 0; k--) {
                ar = q[0]  * t[2] + q[1]  * t[3];   /* y1 = x1 * w^2          */
                ai = q[1]  * t[2] - q[0]  * t[3];
                br = p2[0] * t[0] + p2[1] * t[1];   /* y2 = x2 * w            */
                bi = p2[1] * t[0] - p2[0] * t[1];
                cr = p3[0] * t[4] + p3[1] * t[5];   /* y3 = x3 * w^3          */
                ci = p3[1] * t[4] - p3[0] * t[5];
                t += 6;
                ur = p[0] + ar;  ui = p[1] + ai;    /* x0 + y1                */
                vr = br + cr;    vi = bi + ci;      /* y2 + y3                */
                wr = p[0] - ar;  wi = p[1] - ai;    /* x0 - y1                */
                p[0]  = ur + vr;  p[1]  = ui + vi;
                p2[0] = ur - vr;  p2[1] = ui - vi;
                vr = br - cr;    vi = bi - ci;      /* y2 - y3                */
                q[0]  = wr + vi;  q[1]  = wi - vr;
                p3[0] = wr - vi;  p3[1] = wi + vr;
                p += 2;  q += 2;  p2 += 2;  p3 += 2;
            }
            p = p3;                                 /* next group             */
        }
    }

    /* FFT_M not a power of 4: one radix-2 stage of length FFT_M is left.
     * (Not the case for FEAT_N_FFT = 512; kept so other sizes still work.) */
    if (h < FFT_M) {
        p = z;
        q = z + 2 * h;
        t = tw;
        for (k = h; k != 0; k--) {
            vr = q[0] * t[0] + q[1] * t[1];
            vi = q[1] * t[0] - q[0] * t[1];
            t += 4;                                 /* exp(-j*2*pi*k/FFT_M)   */
            ur = p[0]; ui = p[1];
            p[0] = ur + vr;  p[1] = ui + vi;
            q[0] = ur - vr;  q[1] = ui - vi;
            p += 2;
            q += 2;
        }
    }

    /*
     * Unpack to the real spectrum and take the power. With Z the FFT above,
     * a = Z[k], b = Z[M-k], c + j*s = exp(j*2*pi*k/N):
     *
     *   e = a + conj(b)        d = a - conj(b)
     *   2*X[k]   = e - j*(c - j*s)*d
     *   2*X[M-k] = conj(e) + j*(c + j*s)*conj(d)   (same products, other signs)
     *
     * so one pair (a, b) and four multiplies give both bins.
     */
    ur = z[0]; ui = z[1];                           /* k = 0 and k = M        */
    vr = ur + ui;
    vi = ur - ui;
    z[0]         = 4.0f * vr * vr;
    z[2 * FFT_M] = 4.0f * vi * vi;

    p = z + 2;                                      /* a, walks up            */
    q = z + 2 * FFT_M - 2;                          /* b, walks down          */
    t = tw + 2;
    for (k = FFT_M / 2 - 1; k != 0; k--) {
        float er = p[0] + q[0];
        float ei = p[1] - q[1];
        float dr = p[0] - q[0];
        float di = p[1] + q[1];
        wr = t[0]; wi = t[1];
        t += 2;
        vr = wr * di - wi * dr;
        vi = wr * dr + wi * di;
        ur = er + vr;  ui = ei - vi;
        p[0] = ur * ur + ui * ui;                   /* bin k                  */
        ur = er - vr;  ui = ei + vi;
        q[0] = ur * ur + ui * ui;                   /* bin M-k                */
        p += 2;
        q -= 2;
    }
    ur = p[0]; ui = p[1];                           /* k = M/2: a and b coincide */
    p[0] = 4.0f * (ur * ur + ui * ui);
}

/*
 * adc_buf : FEAT_SLICE_LEN raw ADC codes (0..4095)
 * out     : [FEAT_N_FRAMES][FEAT_N_OUT] normalised MFCC
 */
void mfcc_extract(const unsigned int *adc_buf, float out[FEAT_N_FRAMES][FEAT_N_OUT])
{
    register const float *a;
    register const float *w;
    register float acc;
    const unsigned int *s;
    const uint16_t *ms, *ml;
    const float *mu, *sg;
    float *lm, *o;
    float mean, gmax, floor_v, v;
    uint32_t sum;
    uint16_t i, f, m, k;

    if (!tables_ready) build_tables();

    /* --- 1: mean of the slice (exact integer sum) ---------------------- */
    sum = 0;
    s = adc_buf;
    for (i = FEAT_SLICE_LEN; i != 0; i--) sum += *s++;
    mean = (float)sum / (float)FEAT_SLICE_LEN;

    /* --- 2-4: per frame: window -> FFT -> power -> mel -> log ---------- */
    lm = &log_mel[0][0];
    gmax = -1.0e30f;
    s = adc_buf;
    for (f = FEAT_N_FRAMES; f != 0; f--) {
        frame_power(s, mean);
        s += FEAT_HOP;

        w  = feat_mel_weights;
        ms = feat_mel_start;
        ml = feat_mel_len;
        for (m = FEAT_N_MELS; m != 0; m--) {
            a = z + 2 * (*ms++);
            acc = 0.0f;
            for (k = *ml++; k != 0; k--) {
                acc += (*a) * (*w++);
                a += 2;
            }
            /* undo the 4 * FEAT_ADC_SCALE^2 carried in z[] */
            acc *= 0.25f / (FEAT_ADC_SCALE * FEAT_ADC_SCALE);
            if (acc < FEAT_AMIN) acc = FEAT_AMIN;
            v = POWER_DB(acc);
            *lm++ = v;
            if (v > gmax) gmax = v;
        }
    }

    /* --- 5: GLOBAL top_db floor ----------------------------------------
     * Cannot be applied until every frame of the slice is done: training
     * (librosa) clips against the slice maximum. */
    floor_v = gmax - FEAT_TOP_DB;
    lm = &log_mel[0][0];
    for (i = FEAT_N_FRAMES * FEAT_N_MELS; i != 0; i--) {
        if (*lm < floor_v) *lm = floor_v;
        lm++;
    }

    /* --- 6-7: DCT -> z-score -> clip ----------------------------------- */
    lm = &log_mel[0][0];
    o  = &out[0][0];
    for (f = FEAT_N_FRAMES; f != 0; f--) {
        w  = &feat_dct_matrix[0][0];
        mu = feat_mu;
        sg = feat_sigma;
        for (i = FEAT_N_OUT; i != 0; i--) {
            a = lm;
            acc = 0.0f;
            for (m = FEAT_N_MELS; m != 0; m--) acc += (*w++) * (*a++);
            v = (acc - *mu++) / *sg++;
            if (v >  FEAT_CLIP) v =  FEAT_CLIP;
            if (v < -FEAT_CLIP) v = -FEAT_CLIP;
            *o++ = v;
        }
        lm += FEAT_N_MELS;
    }
}

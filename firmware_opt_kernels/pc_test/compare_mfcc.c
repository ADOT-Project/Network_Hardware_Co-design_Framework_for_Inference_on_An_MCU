/*
 * PC check for drop_in/mfcc_extract.c.
 *
 *   ref : original/mfcc_extract.c (the baseline)
 *   new : drop_in/mfcc_extract.c
 *   exp : golden_expected_* in the golden vector headers (the training-side
 *         reference the board test already uses)
 *
 * For every golden slice: how far the features are from each other and from
 * the reference, how many int16 network inputs change, and whether the
 * network's logits and decision move (indexed reference kernels, so only the
 * features differ).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "definitions.h"
#include "weights.h"
#include "feature_tables.h"
#if NUM_CLASSES == 2
#include "golden_vector_normal.h"
#include "golden_vector_abnormal.h"
#else
#include "golden_vector_healthy.h"
#include "golden_vector_bearing.h"
#include "golden_vector_propeller.h"
#endif

void mfcc_ref(const unsigned int*, float out[FEAT_N_FRAMES][FEAT_N_OUT]);
void mfcc_new(const unsigned int*, float out[FEAT_N_FRAMES][FEAT_N_OUT]);
void ref_conv1(const int16_t*, int16_t*, const int16_t*, const int16_t*);
void ref_conv2(const int16_t*, int16_t*, const int16_t*, const int16_t*);
void ref_dense(const int16_t*, const int16_t*, const int16_t*, float*);

static unsigned int adc[FEAT_SLICE_LEN];           /* int is 32-bit here */
static float f_ref[FEAT_N_FRAMES][FEAT_N_OUT], f_new[FEAT_N_FRAMES][FEAT_N_OUT];
static int16_t in_ref[IMAGE_H * IMAGE_W], in_new[IMAGE_H * IMAGE_W];
static int16_t ws[MAX_ELEMENTS];

static void to_input(float f[FEAT_N_FRAMES][FEAT_N_OUT], int16_t *dst)   /* as main.c does */
{
    int i, o, idx = 0;
    for (i = 0; i < FEAT_N_FRAMES; i++)
        for (o = 0; o < FEAT_N_OUT; o++) {
            float s = f[i][o] * INPUT_SCALE;
            if (s > 32767.0f) s = 32767.0f;
            if (s < -32768.0f) s = -32768.0f;
            dst[idx++] = (int16_t)s;
        }
}

static int logits(const int16_t *in, float *lg)
{
    int k, best = 0;
    ref_conv1(in, ws, conv1_weights, conv1_bias);
    ref_conv2(ws, &ws[POOL1_SIZE], conv2_weights, conv2_bias);
    ref_dense(&ws[POOL1_SIZE], fc_weights, fc_bias, lg);
    for (k = 1; k < NUM_CLASSES; k++) if (lg[k] > lg[best]) best = k;
    return best;
}

static int one(const char *name, const uint16_t *g, const float *expected, int shift)
{
    float lr[NUM_CLASSES], ln[NUM_CLASSES];
    double d_rn = 0, d_re = 0, d_ne = 0, lrel = 0;
    int i, o, k, changed = 0, maxlsb = 0, cr, cn;

    /* shift > 0: a second look at the same recording, started `shift`
     * samples in and wrapped, so the frames are not the golden ones */
    for (i = 0; i < FEAT_SLICE_LEN; i++) adc[i] = g[(i + shift) % FEAT_SLICE_LEN];
    memset(f_ref, 0, sizeof f_ref); memset(f_new, 0, sizeof f_new);
    mfcc_ref(adc, f_ref);
    mfcc_new(adc, f_new);

    for (i = 0; i < FEAT_N_FRAMES; i++)
        for (o = 0; o < FEAT_N_OUT; o++) {
            double a = fabs((double)f_new[i][o] - f_ref[i][o]);
            if (a > d_rn) d_rn = a;
            if (!shift) {
                double e = expected[i * FEAT_N_OUT + o];
                a = fabs(f_ref[i][o] - e); if (a > d_re) d_re = a;
                a = fabs(f_new[i][o] - e); if (a > d_ne) d_ne = a;
            }
        }
    to_input(f_ref, in_ref); to_input(f_new, in_new);
    for (i = 0; i < IMAGE_H * IMAGE_W; i++) {
        int d = abs(in_ref[i] - in_new[i]);
        if (d) changed++;
        if (d > maxlsb) maxlsb = d;
    }
    cr = logits(in_ref, lr); cn = logits(in_new, ln);
    for (k = 0; k < NUM_CLASSES; k++) {
        double r = fabs((double)ln[k] - lr[k]) / (fabs((double)lr[k]) + 1e-30);
        if (r > lrel) lrel = r;
    }
    printf("%-10s shift %5d | new-ref %.2e", name, shift, d_rn);
    if (!shift) printf(" | ref-exp %.2e  new-exp %.2e", d_re, d_ne);
    printf(" | inputs changed %d/%d (max %d LSB) | logits max rel %.3f%% | class %d -> %d%s\n",
           changed, IMAGE_H * IMAGE_W, maxlsb, 100.0 * lrel, cr, cn, cr == cn ? "" : "  <-- FLIP");
    return cr != cn;
}

int main(void)
{
    int bad = 0, s;
    static const int shifts[] = { 0, 1, 97, 4001, 9999 };
    for (s = 0; s < 5; s++) {
#if NUM_CLASSES == 2
        bad += one("normal",   golden_adc_normal,   &golden_expected_normal[0][0],   shifts[s]);
        bad += one("abnormal", golden_adc_abnormal, &golden_expected_abnormal[0][0], shifts[s]);
#else
        bad += one("healthy",   golden_adc_healthy,   &golden_expected_healthy[0][0],   shifts[s]);
        bad += one("bearing",   golden_adc_bearing,   &golden_expected_bearing[0][0],   shifts[s]);
        bad += one("propeller", golden_adc_propeller, &golden_expected_propeller[0][0], shifts[s]);
#endif
    }
    /* the tables are built on the first call; run one slice again to make
     * sure later calls give the same answer */
#if NUM_CLASSES == 2
    bad += one("repeat", golden_adc_normal, &golden_expected_normal[0][0], 0);
#else
    bad += one("repeat", golden_adc_healthy, &golden_expected_healthy[0][0], 0);
#endif
    printf("\nRESULT: %s\n", bad ? "decision changed on at least one slice" : "same decision on every slice");
    return bad ? 1 : 0;
}

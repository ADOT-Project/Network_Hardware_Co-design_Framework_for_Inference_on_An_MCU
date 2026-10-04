/*
 * PC check for firmware_opt_kernels/drop_in.
 *
 * Runs the same inputs through
 *   ref : the indexed kernels in the project's own convolution.c
 *   new : drop_in/convolution.c (CPU) and drop_in/dense_layer.cla (CLA),
 *         with each stage on the core cla_pipeline_config.h assigns it to,
 *         in the order main.c runs them
 * and compares every workspace word and every logit.
 *
 * Built once per stage split by run.sh, which supplies a
 * cla_pipeline_config.h for that split.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define __interrupt
#define fc_weights weight          /* main.c's CLA-side name for the dense weights */
#include "definitions.h"
#include "weights.h"               /* input_image, conv*, weight, fc_bias */
#include "cla_pipeline_config.h"
#include "F2837xD_device.h"

struct pc_stub_gpio_data GpioDataRegs;
int16_t workspace[MAX_ELEMENTS];
float   fResult[NUM_CLASSES];

/* reference: project's convolution.c, compiled with these names */
void ref_conv1(const int16_t*, int16_t*, const int16_t*, const int16_t*);
void ref_conv2(const int16_t*, int16_t*, const int16_t*, const int16_t*);
void ref_dense(const int16_t*, const int16_t*, const int16_t*, float*);

/* new, CPU */
void conv1_pool_fused_cpu(const int16_t*, int16_t*, const int16_t*, const int16_t*);
void conv2_pool_fused_cpu(const int16_t*, int16_t*, const int16_t*, const int16_t*);
void dense_fc_cpu(const int16_t*, const int16_t*, const int16_t*, float*);

/* new, CLA tasks (only the ones this split compiles) */
#if CLA_ANY_STAGE
void Cla1Task1(void);
#endif
#if !CLA_SINGLE_TASK && CLA_STAGE_CONV2_POOL2
void Cla1Task2(void);
#endif
#if !CLA_SINGLE_TASK && CLA_STAGE_DENSE
void Cla1Task3(void);
#endif

static void run_new(void)
{
#if CLA_SINGLE_TASK
    Cla1Task1();
#else
  #if CLA_STAGE_CONV1_POOL1
    Cla1Task1();
  #else
    conv1_pool_fused_cpu(input_image, workspace, conv1_weights, conv1_bias);
  #endif
  #if CLA_STAGE_CONV2_POOL2
    Cla1Task2();
  #else
    conv2_pool_fused_cpu(workspace, &workspace[POOL1_SIZE], conv2_weights, conv2_bias);
  #endif
  #if CLA_STAGE_DENSE
    Cla1Task3();
  #else
    dense_fc_cpu(&workspace[POOL1_SIZE], weight, fc_bias, fResult);
  #endif
#endif
}

static uint32_t rng = 0x1234567u;
static int32_t rnd(int32_t span)            /* uniform in [-span, span] */
{
    rng = rng * 1664525u + 1013904223u;
    return (int32_t)((rng >> 8) % (uint32_t)(2 * span + 1)) - span;
}

static int16_t base[IMAGE_H * IMAGE_W];

/*
 * set 0, "runtime": what main.c really feeds the network -- normalised MFCC
 *        features times INPUT_SCALE. Bell-shaped noise with a per-coefficient
 *        offset, sigma swept from 0.5 to 3 feature units.
 * set 1, "overflow": the sample stored in weights.h, frames shuffled, noise
 *        added. That sample is peak-scaled to +-32767, about 40x larger than
 *        a runtime input, and drives activations past int16. The indexed
 *        kernels are already wrong there; this set only shows whether the new
 *        ones are wrong in the same way.
 */
static void make_input(int set, int t)
{
    int i, k;
    if (set == 0) {
        int32_t sigma = (int32_t)(INPUT_SCALE * (0.5f + 0.25f * (t % 11)));
        for (i = 0; i < IMAGE_H; i++)
            for (k = 0; k < IMAGE_W; k++) {
                int32_t v = (rnd(sigma) + rnd(sigma) + rnd(sigma)) + (k - 6) * (t % 7) * 16;
                input_image[i * IMAGE_W + k] = (int16_t)v;
            }
    } else if (t == 0) {
        memcpy(input_image, base, sizeof base);
    } else {
        int noise = 200 * (1 + t % 20);
        for (i = 0; i < IMAGE_H; i++) {
            int src = (int)((rnd(1 << 20) + (1 << 20)) % IMAGE_H);
            for (k = 0; k < IMAGE_W; k++) {
                int32_t v = base[src * IMAGE_W + k] + rnd(noise);
                if (v > 32767) v = 32767;
                if (v < -32768) v = -32768;
                input_image[i * IMAGE_W + k] = (int16_t)v;
            }
        }
    }
}

/* returns 1 when anything differed */
static int run_set(int set, int n_img)
{
    static int16_t ws_ref[MAX_ELEMENTS];
    float lg_ref[NUM_CLASSES];
    long ws_diff = 0, ws_total = 0, lg_bitdiff = 0, argmax_diff = 0, wrapped = 0;
    int ws_maxabs = 0, peak = 0;
    double lg_maxrel = 0.0;
    int t, i, k;

    for (t = 0; t < n_img; t++) {
        make_input(set, t);

        ref_conv1(input_image, ws_ref, conv1_weights, conv1_bias);
        ref_conv2(ws_ref, &ws_ref[POOL1_SIZE], conv2_weights, conv2_bias);
        ref_dense(&ws_ref[POOL1_SIZE], weight, fc_bias, lg_ref);

        memset(workspace, 0x55, sizeof workspace);
        run_new();

        for (i = 0; i < MAX_ELEMENTS; i++) {
            int d = abs((int)workspace[i] - (int)ws_ref[i]);
            ws_total++;
            if (d) { ws_diff++; if (d > ws_maxabs) ws_maxabs = d; }
            if (ws_ref[i] < 0) wrapped++;      /* relu output cannot be negative */
            if (ws_ref[i] > peak) peak = ws_ref[i];
        }
        {
            int a_ref = 0, a_new = 0;
            for (k = 0; k < NUM_CLASSES; k++) {
                double rel;
                if (memcmp(&fResult[k], &lg_ref[k], sizeof(float)) != 0) lg_bitdiff++;
                rel = fabs((double)fResult[k] - lg_ref[k]) / (fabs((double)lg_ref[k]) + 1e-30);
                if (rel > lg_maxrel) lg_maxrel = rel;
                if (lg_ref[k] > lg_ref[a_ref]) a_ref = k;
                if (fResult[k] > fResult[a_new]) a_new = k;
            }
            if (a_ref != a_new) argmax_diff++;
        }
    }

    printf("  %-8s activations %ld/%ld differ (max |d| %d) | logits %ld/%d not bit-equal, "
           "max rel %.2g%% | argmax flips %ld/%d | ref peak %d, ref wrapped %ld\n",
           set == 0 ? "runtime" : "overflow", ws_diff, ws_total, ws_maxabs,
           lg_bitdiff, n_img * NUM_CLASSES, 100.0 * lg_maxrel, argmax_diff, n_img, peak, wrapped);
    return (ws_diff || lg_bitdiff) ? 1 : 0;
}

/* exit code: bit 0 = runtime set differed, bit 1 = overflow set differed */
int main(int argc, char **argv)
{
    int n_img = (argc > 1) ? atoi(argv[1]) : 300;
    int rc;
    memcpy(base, input_image, sizeof base);
    printf("S1S2S3=%d%d%d  (%d inputs per set)\n",
           CLA_STAGE_CONV1_POOL1, CLA_STAGE_CONV2_POOL2, CLA_STAGE_DENSE, n_img);
    rc  = run_set(0, n_img);
    rc |= run_set(1, n_img) << 1;
    return rc;
}

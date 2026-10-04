/*
 * fused_kernels_opt.h
 *
 * Pointer-walking, 3x3-unrolled versions of the three inference stages:
 *
 *     conv1 + relu + pool1      conv2 + relu + pool2      dense
 *
 * ONE source for both cores. convolution.c includes this file to build the
 * C28x functions, dense_layer.cla includes it to build the CLA ones, so the
 * arithmetic cannot drift apart between the two -- "move a stage to the other
 * core and the result does not change" holds by construction.
 *
 * What changed against the indexed kernels, and why each one matters at -O0
 * (where the compiler removes nothing for you):
 *
 *   1. No index arithmetic per MAC. The old inner statement
 *          input[row_ptr + (c + fj)] * filters[filter_offset + fi*3 + fj]
 *      computed two 32-bit indices for every multiply. Here both operands are
 *      read through a pointer that is simply advanced.
 *   2. The 3x3 window is written out: 9 MACs in a row, no fi/fj loops and no
 *      branches inside a window. A 3-trip loop is nearly all loop overhead,
 *      and the CLA has no zero-overhead repeat block to hide it.
 *   3. The bias term is scaled once per filter instead of once per window.
 *   4. relu and the 2x1 max-pool are done on the float sums, and the result
 *      is descaled and converted to int16 ONCE per output instead of once per
 *      window: half the divisions. Truncation is monotonic, so
 *          (int16)(max(s0, s1, 0) / S)  ==  max((int16)(relu(s0) / S),
 *                                               (int16)(relu(s1) / S))
 *      as long as both quotients fit in int16. See KERNEL_POOL_FLOAT for the
 *      case where they do not.
 *   5. Output is written through a walking pointer (the indexed form was
 *      already sequential, it just recomputed the index every time).
 *
 * With the default switches the results are BIT-IDENTICAL to the indexed
 * kernels for every input whose activations fit in int16: same products,
 * same order of accumulation, same division. See pc_test/ for the check.
 *
 * Switches (define before including, or with a compiler -D):
 *
 *   KERNEL_ADDR_WALK       1 (default)  operands read as *a++ / *w++, the
 *                                       only indirect mode the CLA has
 *                                       (post-increment on MAR0/MAR1).
 *                          0            constant offsets a[k] / w[k] from a
 *                                       pointer held still. May suit the
 *                                       C28x better. Same numbers either way
 *                                       -- measure both and keep the faster.
 *
 *   KERNEL_USE_RECIPROCAL  0 (default)  descale by dividing, as before.
 *                          1            descale by multiplying with a
 *                                       compile-time 1/SCALE. Removes the
 *                                       software division on the CLA. NOT
 *                                       bit-identical: an output can differ
 *                                       by one LSB. Regenerate
 *                                       expected_logits.txt if you turn it on.
 *
 *   KERNEL_POOL_FLOAT      1 (default)  item 4 above.
 *                          0            convert each window to int16 first
 *                                       and pool the integers, exactly as the
 *                                       indexed kernels did. Bit-identical
 *                                       for ALL inputs, including ones whose
 *                                       activations overflow int16 -- there
 *                                       the two orders pick different wrong
 *                                       values. Costs one more descale and
 *                                       conversion per output.
 *
 * Which functions are emitted is chosen by the includer:
 *
 *   #define KFN_CONV1  <name>     -> conv1 + relu + pool1
 *   #define KFN_CONV2  <name>     -> conv2 + relu + pool2
 *   #define KFN_DENSE  <name>     -> dense, as a function
 *   K_DENSE_BODY(in, w, bias, out) is always available as a statement, for
 *   the CLA tasks that run the dense layer inline.
 *
 * int is 16 bits on the C28x and 32 on the CLA; every counter and step below
 * is an explicit uint16_t or a small constant, so neither width matters.
 */

#ifndef FUSED_KERNELS_OPT_H_
#define FUSED_KERNELS_OPT_H_

#include <stdint.h>
#include "definitions.h"

#ifndef KERNEL_ADDR_WALK
#define KERNEL_ADDR_WALK 1
#endif
#ifndef KERNEL_USE_RECIPROCAL
#define KERNEL_USE_RECIPROCAL 0
#endif
#ifndef KERNEL_POOL_FLOAT
#define KERNEL_POOL_FLOAT 1
#endif

/* 2x1 pooling keeps the width, and the kernels below rely on it to write the
 * output sequentially. */
#if (FILTER_SIZE != 3)
#error "fused_kernels_opt.h is written for 3x3 filters"
#endif

#if KERNEL_USE_RECIPROCAL
#define K_DESCALE(x, scale)   ((x) * (1.0f / (scale)))
#else
#define K_DESCALE(x, scale)   ((x) / (scale))
#endif

#define K_MAC(x, y)           ((float)(x) * (float)(y))

/*
 * relu + 2x1 max-pool over the two windows of one output.
 *   K_POOL_DECL            declares the running maximum
 *   K_POOL_INIT            before the two windows
 *   K_POOL_ACC(sum, S)     after each window's sum (bias already added)
 *   K_POOL_OUT(S)          the int16 to store
 */
#if KERNEL_POOL_FLOAT
#define K_POOL_DECL           float k_m;
#define K_POOL_INIT           k_m = 0.0f;                  /* 0 = relu floor */
#define K_POOL_ACC(sum, S)    if ((sum) > k_m) k_m = (sum);
#define K_POOL_OUT(S)         ((int16_t)K_DESCALE(k_m, S))
#else
#define K_POOL_DECL           int16_t k_m; int16_t k_v;
#define K_POOL_INIT           k_m = -32768;
#define K_POOL_ACC(sum, S)    if ((sum) < 0.0f) (sum) = 0.0f;              \
                              k_v = (int16_t)K_DESCALE(sum, S);            \
                              if (k_v > k_m) k_m = k_v;
#define K_POOL_OUT(S)         (k_m)
#endif

/*
 * K_TAPS9(sum, a, w, ROW_W)
 *   Adds the 9 products of one 3x3 window to sum, in the same order as the
 *   indexed kernels (row by row, left to right).
 *   On entry a points at the window's top-left input, w at its first weight.
 *   On exit  w has advanced by 9. Where a ends up depends on the mode, which
 *   is why moving on is done with K_NEXT_PLANE and never by hand.
 *
 * K_NEXT_PLANE(a, ROW_W, PLANE)
 *   After K_TAPS9, moves a to the same window position in the next input
 *   channel (PLANE = rows * ROW_W inputs further on).
 */
#if KERNEL_ADDR_WALK

#define K_TAPS9(sum, a, w, ROW_W)                               \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a,   *w++);   a += (ROW_W) - 2;               \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a,   *w++);   a += (ROW_W) - 2;               \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a++, *w++);                                   \
    sum += K_MAC(*a,   *w++);

/* a sits on the window's bottom-right input: 2 rows and 2 columns in */
#define K_NEXT_PLANE(a, ROW_W, PLANE)                           \
    a += (PLANE) - 2 * (ROW_W) - 2;

#else  /* constant offsets */

#define K_TAPS9(sum, a, w, ROW_W)                               \
    sum += K_MAC(a[0],               w[0]);                     \
    sum += K_MAC(a[1],               w[1]);                     \
    sum += K_MAC(a[2],               w[2]);                     \
    sum += K_MAC(a[(ROW_W)],         w[3]);                     \
    sum += K_MAC(a[(ROW_W) + 1],     w[4]);                     \
    sum += K_MAC(a[(ROW_W) + 2],     w[5]);                     \
    sum += K_MAC(a[2 * (ROW_W)],     w[6]);                     \
    sum += K_MAC(a[2 * (ROW_W) + 1], w[7]);                     \
    sum += K_MAC(a[2 * (ROW_W) + 2], w[8]);                     \
    w += 9;

#define K_NEXT_PLANE(a, ROW_W, PLANE)                           \
    a += (PLANE);

#endif /* KERNEL_ADDR_WALK */


/*
 * Dense layer as a statement. IN: DENSE_LAYER_INPUT int16 activations,
 * W: NUM_CLASSES rows of DENSE_LAYER_INPUT weights, B: NUM_CLASSES biases,
 * OUT: NUM_CLASSES float logits. The W*X sum is descaled first and the bias
 * is added in real units, exactly as before.
 */
#define K_DENSE_BODY(IN, W, B, OUT)                                         \
    {                                                                       \
        register const int16_t *kd_x;                                       \
        register const int16_t *kd_w = (W);                                 \
        register float kd_sum;                                              \
        const int16_t *kd_b = (B);                                          \
        float *kd_o = (OUT);                                                \
        uint16_t kd_k, kd_i;                                                \
        for (kd_k = NUM_CLASSES; kd_k != 0; kd_k--) {                       \
            kd_x = (IN);                                                    \
            kd_sum = 0.0f;                                                  \
            for (kd_i = DENSE_LAYER_INPUT; kd_i != 0; kd_i--) {             \
                kd_sum += K_MAC(*kd_x++, *kd_w++);                          \
            }                                                               \
            *kd_o++ = K_DESCALE(kd_sum, FC_WEIGHT_SCALE)                    \
                    + (float)(*kd_b++) * (INPUT_SCALE / FC_BIAS_SCALE);     \
        }                                                                   \
    }

#endif /* FUSED_KERNELS_OPT_H_ */


/* ------------------------------------------------------------------------
 * Function bodies. Outside the include guard on purpose: each is emitted
 * once, under the name the includer asked for.
 * --------------------------------------------------------------------- */

#ifdef KFN_CONV1
/*
 * conv1 + relu + pool1. input: IMAGE_H x IMAGE_W (one channel).
 * pooled_out: NUM_FILTERS x POOL1_OUT_H x POOL1_OUT_W, written in order.
 */
void KFN_CONV1(const int16_t* input, int16_t* pooled_out,
               const int16_t* filters, const int16_t* bias)
{
    register const int16_t *a;      /* walks the input window            */
    register const int16_t *w;      /* walks this filter's 9 weights     */
    register float sum;
    const int16_t *wf = filters;    /* first weight of the current filter */
    const int16_t *row;             /* input row 2*pr, column 0           */
    const int16_t *col;             /* input row 2*pr, column c           */
    const int16_t *q;               /* window top-left for this sub-row   */
    float b;                        /* bias, in the weight-scale domain   */
    K_POOL_DECL
    uint16_t f, pr, c, sub;

    for (f = NUM_FILTERS; f != 0; f--) {
        b = (float)(*bias++) * ((CONV1_WEIGHT_SCALE / CONV1_BIAS_SCALE) * INPUT_SCALE);
        row = input;
        for (pr = POOL1_OUT_H; pr != 0; pr--) {
            col = row;
            for (c = CONV1_OUT_W; c != 0; c--) {
                K_POOL_INIT
                q = col;
                for (sub = 2; sub != 0; sub--) {
                    a = q;
                    w = wf;
                    sum = 0.0f;
                    K_TAPS9(sum, a, w, IMAGE_W)
                    sum += b;
                    K_POOL_ACC(sum, CONV1_WEIGHT_SCALE)
                    q += IMAGE_W;
                }
                *pooled_out++ = K_POOL_OUT(CONV1_WEIGHT_SCALE);
                col++;
            }
            row += 2 * IMAGE_W;
        }
        wf += FILTER_SIZE * FILTER_SIZE;
    }
}
#undef KFN_CONV1
#endif /* KFN_CONV1 */


#ifdef KFN_CONV2
/*
 * conv2 + relu + pool2. input: NUM_FILTERS x POOL1_OUT_H x POOL1_OUT_W.
 * pooled_out: NUM_FILTERS x POOL2_OUT_H x POOL2_OUT_W, written in order.
 * Weights are laid out [out filter][in channel][3][3], so one pointer walks
 * all NUM_FILTERS*9 of an output filter's weights without ever being reset
 * inside a window.
 */
void KFN_CONV2(const int16_t* input, int16_t* pooled_out,
               const int16_t* filters, const int16_t* bias)
{
    register const int16_t *a;
    register const int16_t *w;
    register float sum;
    const int16_t *wf = filters;    /* first weight of the current out filter */
    const int16_t *row;             /* channel 0, row 2*pr, column 0          */
    const int16_t *col;             /* channel 0, row 2*pr, column c          */
    const int16_t *q;               /* channel 0 window top-left, this sub-row */
    float b;
    K_POOL_DECL
    uint16_t f, pr, c, sub, ic;

    for (f = NUM_FILTERS; f != 0; f--) {
        b = (float)(*bias++) * ((CONV2_WEIGHT_SCALE / CONV2_BIAS_SCALE) * INPUT_SCALE);
        row = input;
        for (pr = POOL2_OUT_H; pr != 0; pr--) {
            col = row;
            for (c = CONV2_OUT_W; c != 0; c--) {
                K_POOL_INIT
                q = col;
                for (sub = 2; sub != 0; sub--) {
                    a = q;
                    w = wf;
                    sum = 0.0f;
                    for (ic = NUM_FILTERS; ic != 0; ic--) {
                        K_TAPS9(sum, a, w, POOL1_OUT_W)
                        K_NEXT_PLANE(a, POOL1_OUT_W, POOL1_OUT_H * POOL1_OUT_W)
                    }
                    sum += b;
                    K_POOL_ACC(sum, CONV2_WEIGHT_SCALE)
                    q += POOL1_OUT_W;
                }
                *pooled_out++ = K_POOL_OUT(CONV2_WEIGHT_SCALE);
                col++;
            }
            row += 2 * POOL1_OUT_W;
        }
        wf += NUM_FILTERS * FILTER_SIZE * FILTER_SIZE;
    }
}
#undef KFN_CONV2
#endif /* KFN_CONV2 */


#ifdef KFN_DENSE
void KFN_DENSE(const int16_t* input, const int16_t* weights,
               const int16_t* bias, float* output)
{
    K_DENSE_BODY(input, weights, bias, output)
}
#undef KFN_DENSE
#endif /* KFN_DENSE */

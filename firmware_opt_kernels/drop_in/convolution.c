/*
 * convolution.c
 *
 *  Created on: Feb 11, 2026
 *      Author: canberk
 */
#include "convolution.h"



void convolution(const float* input, float* output, const float* filters, const float* bias, int in_h, int in_w)
{
    uint16_t f, r, c, fi, fj;

    // Calculate output dimensions based on "Valid" convolution rules
    int out_h = in_h - FILTER_SIZE + 1;
    int out_w = in_w - FILTER_SIZE + 1;

    for (f = 0; f < NUM_FILTERS; f++) {

        uint32_t filter_offset = f * (FILTER_SIZE * FILTER_SIZE);
        uint32_t output_offset = f * (out_h * out_w);

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {
                float sum = 0.0f;

                for (fi = 0; fi < FILTER_SIZE; fi++) {
                    // Stride across the input is always the full input width
                    uint32_t input_row_ptr = (r + fi) * in_w;

                    for (fj = 0; fj < FILTER_SIZE; fj++) {
                        sum += input[input_row_ptr + (c + fj)] * filters[filter_offset + (fi * FILTER_SIZE) + fj];
                    }
                }

                // Stride across the output is the calculated output width
                output[output_offset + (r * out_w) + c] = sum + bias[f];
            }
        }
    }
}


void relu_activation(float* data, uint32_t size)
{
    uint32_t i;

    for (i = 0; i < size; i++) {

        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

void max_pooling(float* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters) {
    uint16_t f, r, c, i, j;
    uint32_t write_idx = 0;

    // Calculate output dimensions (downsampling by factor of 2)

    for (f = 0; f < num_filters; f++) {
        // Offset to the start of the current feature map in the input data
        // Use the original dimensions (in_h * in_w) to find each filter's start
        uint32_t filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < in_h - 1; r += 2) {     // Step by 2 rows
            for (c = 0; c < in_w - 1; c += 2) { // Step by 2 columns
                float max_val = -1e37f; // Smallest possible float

                // Look at the 2x2 window
                for (i = 0; i < 2; i++) {
                    uint32_t row_ptr = filter_offset + (r + i) * in_w;
                    for (j = 0; j < 2; j++) {
                        float val = data[row_ptr + (c + j)];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }

                // Write the winner to the front of the array
                data[write_idx++] = max_val;
            }
        }
    }
}


void convolution_2(const float* input, float* output, const float* filters, const float* bias, uint16_t in_h, uint16_t in_w)
{
    uint16_t out_f, in_c, r, c, fi, fj;

    // Calculate output dimensions (Valid convolution, 3x3 filter)
    uint16_t out_h = in_h - 3 + 1;
    uint16_t out_w = in_w - 3 + 1;

    for (out_f = 0; out_f < NUM_FILTERS; out_f++) {
        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {
                float sum = 0.0f;

                for (in_c = 0; in_c < NUM_FILTERS; in_c++) {
                    // Use in_w for the row jump in the input
                    uint32_t in_offset = (uint32_t)in_c * (in_h * in_w);

                    // Filter memory layout: (Output Filter index * Total Input Channels * 9)
                    uint32_t filter_offset = (out_f * NUM_FILTERS * 9) + (in_c * 9);

                    for (fi = 0; fi < 3; fi++) {
                        // Point to the correct row using the input width (in_w)
                        uint32_t input_row_ptr = in_offset + (r + fi) * in_w;

                        for (fj = 0; fj < 3; fj++) {
                            sum += input[input_row_ptr + (c + fj)] * filters[filter_offset + fi * 3 + fj];
                        }
                    }
                }

                // Use out_w for the row jump in the output
                output[out_f * (out_h * out_w) + r * out_w + c] = sum + bias[out_f];
            }
        }
    }
}


void max_pooling_2(float* input, float* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters)
{
    uint16_t f, r, c, i, j;

    // Calculate separate output dimensions
    uint16_t out_h = in_h / 2;
    uint16_t out_w = in_w / 2;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        // Use full input area (h * w) to jump to the next filter
        uint32_t input_filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {
                float max_val = -1e37f;

                for (i = 0; i < 2; i++) {
                    // Critical: Jump by the input width (in_w)
                    uint32_t row_offset = input_filter_offset + ((r * 2) + i) * in_w;

                    for (j = 0; j < 2; j++) {
                        // Horizontal position: column index * 2 + window offset
                        float val = input[row_offset + (c * 2) + j];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                // Write to the output array sequentially
                output[write_idx++] = max_val;
            }
        }
    }
}



// Note: You must define SCALE_FACTOR based on the comment in your generated .h file
// Example: #define CONV1_SCALE 128.5f

void convolution_int16(const int16_t* input, int16_t* output, const int16_t* filters, const int16_t* bias, int in_h, int in_w, float scale_factor)
{
    uint16_t f, r, c, fi, fj;

    int out_h = in_h - FILTER_SIZE + 1;
    int out_w = in_w - FILTER_SIZE + 1;

    for (f = 0; f < NUM_FILTERS; f++) {
        uint32_t filter_offset = f * (FILTER_SIZE * FILTER_SIZE);
        uint32_t output_offset = f * (out_h * out_w);

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {

                // 1. USE INT32 FOR THE ACCUMULATOR TO PREVENT OVERFLOW
                float sum = 0;

                for (fi = 0; fi < FILTER_SIZE; fi++) {
                    uint32_t input_row_ptr = (r + fi) * in_w;

                    for (fj = 0; fj < FILTER_SIZE; fj++) {
                        // 2. Cast to int32_t before multiplying!
                        int32_t in_val = input[input_row_ptr + (c + fj)];
                        int32_t weight_val = filters[filter_offset + (fi * FILTER_SIZE) + fj];

                        sum += (in_val * weight_val);
                    }
                }

                // Add bias
                sum += bias[f];

                // 3. SCALE IT BACK DOWN AND CAST TO INT16
                // We divide by the scale factor to reverse the quantization scaling
                output[output_offset + (r * out_w) + c] = (int16_t)(sum / scale_factor);
            }
        }
    }
}


void relu_activation_int16(int16_t* data, uint32_t size)
{
    uint32_t i;
    for (i = 0; i < size; i++) {
        // Changed 0.0f to integer 0
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}


void max_pooling_int16(int16_t* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters) {
    uint16_t f, r, c, i, j;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        uint32_t filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < in_h - 1; r += 2) {
            for (c = 0; c < in_w - 1; c += 2) {

                // Changed -1e37f to the lowest possible int16 value
                int16_t max_val = -32768;

                for (i = 0; i < 2; i++) {
                    uint32_t row_ptr = filter_offset + (r + i) * in_w;
                    for (j = 0; j < 2; j++) {
                        int16_t val = data[row_ptr + (c + j)];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                data[write_idx++] = max_val;
            }
        }
    }
}

void max_pooling_2x1_int16(const int16_t* input, int16_t* output,
                                 uint16_t in_h, uint16_t in_w, uint16_t num_filters)
{
    uint16_t f, r, c;
    uint16_t out_h = in_h / 2;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        uint32_t off = (uint32_t)f * in_h * in_w;
        for (r = 0; r < out_h; r++) {
            const int16_t* row0 = &input[off + (uint32_t)(r * 2)     * in_w];
            const int16_t* row1 = &input[off + (uint32_t)(r * 2 + 1) * in_w];
            for (c = 0; c < in_w; c++) {
                int16_t a = row0[c], b = row1[c];
                output[write_idx++] = (a > b) ? a : b;
            }
        }
    }
}


void convolution_2_int16(const int16_t* input, int16_t* output, const int16_t* filters, const int16_t* bias, uint16_t in_h, uint16_t in_w, float scale_factor)
{
    uint16_t out_f, in_c, r, c, fi, fj;

    uint16_t out_h = in_h - 3 + 1;
    uint16_t out_w = in_w - 3 + 1;

    for (out_f = 0; out_f < NUM_FILTERS; out_f++) {
        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {

                float sum = 0; // Prevent overflow

                for (in_c = 0; in_c < NUM_FILTERS; in_c++) {
                    uint32_t in_offset = (uint32_t)in_c * (in_h * in_w);
                    uint32_t filter_offset = (out_f * NUM_FILTERS * 9) + (in_c * 9);

                    for (fi = 0; fi < 3; fi++) {
                        uint32_t input_row_ptr = in_offset + (r + fi) * in_w;

                        for (fj = 0; fj < 3; fj++) {
                            int32_t in_val = input[input_row_ptr + (c + fj)];
                            int32_t w_val = filters[filter_offset + fi * 3 + fj];
                            sum += (in_val * w_val);
                        }
                    }
                }

                sum += bias[out_f];
                output[out_f * (out_h * out_w) + r * out_w + c] = (int16_t)(sum / scale_factor);
            }
        }
    }
}

void max_pooling_2_int16(int16_t* input, int16_t* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters)
{
    uint16_t f, r, c, i, j;

    // Calculate separate output dimensions
    uint16_t out_h = in_h / 2;
    uint16_t out_w = in_w / 2;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        // Use full input area (h * w) to jump to the next filter
        uint32_t input_filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {

                // Set to the minimum possible int16 value instead of -1e37f
                int16_t max_val = -32768;

                for (i = 0; i < 2; i++) {
                    // Critical: Jump by the input width (in_w)
                    uint32_t row_offset = input_filter_offset + ((r * 2) + i) * in_w;

                    for (j = 0; j < 2; j++) {
                        // Horizontal position: column index * 2 + window offset
                        int16_t val = input[row_offset + (c * 2) + j];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                // Write to the output array sequentially
                output[write_idx++] = max_val;
            }
        }
    }
}




// ==========================================================================
// CPU twins of the CLA stages (conv1+pool1, conv2+pool2, dense).
//
// The bodies are no longer written here: they come from fused_kernels_opt.h,
// the same file dense_layer.cla includes for the CLA versions. One source for
// both cores, so moving a stage in cla_pipeline_config.h changes where it is
// computed and nothing else. To change the arithmetic, change it there.
//
// The indexed originals these replace are in git history and in
// firmware_opt_kernels/original/.
// ==========================================================================
#define KFN_CONV1  conv1_pool_fused_cpu
#define KFN_CONV2  conv2_pool_fused_cpu
#define KFN_DENSE  dense_fc_cpu
#include "fused_kernels_opt.h"

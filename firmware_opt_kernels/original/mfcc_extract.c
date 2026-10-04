/*
 * mfcc_extract.c  --  minimal MFCC feature extraction for TMS320F28379D
 *
 * TEMEL SURUM. Optimizasyon yok, sadece dogruluk. Once bunu calistirin,
 * golden vector testini gecin, sonra hizlandirirsiniz.
 *
 * Kullanim:
 *   1) 8448 ADC ornegini (0..4095) bir tampona toplayin  -> 16 kHz'de 528 ms
 *   2) mfcc_extract(adc_buf, out) cagirin
 *   3) out[32][12] matrisi CNN'e girer
 *
 * Bellek: ~24 KB stack/statik (asagidaki tabloya bakin). F28379D'de bol.
 * Sure : 200 MHz'de kabaca birkac ms. Hop suresi 16 ms, sorun degil.
 */

#include <math.h>
#include "feature_tables.h"

/* --- calisma tamponlari (statik: stack tasmasin) -----------------------
 *   x        8448 * 4 = 33.0 KB   <-- en buyuk kalem
 *   log_mel  32*40 * 4 =  5.0 KB
 *   fft_re/im 512 * 4 * 2 = 4.0 KB
 * Toplam ~42 KB. F28379D'de 204 KB RAM var.
 *
 * NOT: x[] tamponunu ayirmak istemezseniz, adc_buf'i yerinde
 * donusturebilirsiniz (uint16 -> float ayni yere sigmaz, dikkat).
 */
static float x[FEAT_SLICE_LEN];
static float log_mel[FEAT_N_FRAMES][FEAT_N_MELS];
static float fft_re[FEAT_N_FFT];
static float fft_im[FEAT_N_FFT];

/* --- basit radix-2 karmasik FFT (yerinde) -------------------------------
 * C2000Ware'deki RFFT_f32 bunun ~10 kati hizli ve reel-ozel.
 * Once BU calissin, sonra degistirin.
 */
static void fft512(float *re, float *im, int n)
{
    /* bit-reversal */
    int i, j = 0, k, len;
    for (i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t;
            t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    /* butterfly */
    for (len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * 3.14159265358979f / (float)len;
        float wr = cosf(ang), wi = sinf(ang);
        for (i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (k = 0; k < len / 2; k++) {
                float ur = re[i + k],           ui = im[i + k];
                float vr = re[i + k + len/2] * cr - im[i + k + len/2] * ci;
                float vi = re[i + k + len/2] * ci + im[i + k + len/2] * cr;
                re[i + k]           = ur + vr;
                im[i + k]           = ui + vi;
                re[i + k + len/2]   = ur - vr;
                im[i + k + len/2]   = ui - vi;
                float ncr = cr * wr - ci * wi;
                ci        = cr * wi + ci * wr;
                cr        = ncr;
            }
        }
    }
}

/*
 * adc_buf : FEAT_SLICE_LEN adet ham ADC kodu (0..4095)
 * out     : [FEAT_N_FRAMES][FEAT_N_OUT] normalize edilmis MFCC
 */
void mfcc_extract(const unsigned int *adc_buf, float out[FEAT_N_FRAMES][FEAT_N_OUT])
{
    int i, f, m, o, k, w;

    /* --- ADIM 1: ortalamayi cikar, olcekle ---------------------------- */
    float mean = 0.0f;
    for (i = 0; i < FEAT_SLICE_LEN; i++) mean += (float)adc_buf[i];
    mean /= (float)FEAT_SLICE_LEN;
    for (i = 0; i < FEAT_SLICE_LEN; i++)
        x[i] = ((float)adc_buf[i] - mean) / FEAT_ADC_SCALE;

    /* --- ADIM 2-4: her cerceve: pencere -> FFT -> guc -> mel -> log --- */
    for (f = 0; f < FEAT_N_FRAMES; f++) {
        for (i = 0; i < FEAT_N_FFT; i++) {
            fft_re[i] = x[f * FEAT_HOP + i] * feat_hann_window[i];
            fft_im[i] = 0.0f;
        }
        fft512(fft_re, fft_im, FEAT_N_FFT);

        w = 0;
        for (m = 0; m < FEAT_N_MELS; m++) {
            float acc = 0.0f;
            for (k = 0; k < feat_mel_len[m]; k++) {
                int b = feat_mel_start[m] + k;
                float p = fft_re[b] * fft_re[b] + fft_im[b] * fft_im[b];
                acc += p * feat_mel_weights[w++];
            }
            if (acc < FEAT_AMIN) acc = FEAT_AMIN;
            log_mel[f][m] = 10.0f * log10f(acc);
        }
    }

    /* --- ADIM 5: GLOBAL top_db kirpmasi ------------------------------
     * DIKKAT: bu adim 32 cercevenin TAMAMI bitmeden yapilamaz.
     * Egitimde librosa boyle yapiyor; atlarsaniz sessiz sapma olusur.
     */
    float gmax = log_mel[0][0];
    for (f = 0; f < FEAT_N_FRAMES; f++)
        for (m = 0; m < FEAT_N_MELS; m++)
            if (log_mel[f][m] > gmax) gmax = log_mel[f][m];
    float floor_v = gmax - FEAT_TOP_DB;
    for (f = 0; f < FEAT_N_FRAMES; f++)
        for (m = 0; m < FEAT_N_MELS; m++)
            if (log_mel[f][m] < floor_v) log_mel[f][m] = floor_v;

    /* --- ADIM 6-7: DCT -> z-score -> kirp ----------------------------- */
    for (f = 0; f < FEAT_N_FRAMES; f++) {
        for (o = 0; o < FEAT_N_OUT; o++) {
            float acc = 0.0f;
            for (m = 0; m < FEAT_N_MELS; m++)
                acc += feat_dct_matrix[o][m] * log_mel[f][m];
            float z = (acc - feat_mu[o]) / feat_sigma[o];
            if (z >  FEAT_CLIP) z =  FEAT_CLIP;
            if (z < -FEAT_CLIP) z = -FEAT_CLIP;
            out[f][o] = z;
        }
    }
}

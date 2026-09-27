#include "mfcc_compute.h"
#include "config.h"
#include "mel_dct_tables.h"
#include <math.h>
#include <string.h>

#define FRAME_LEN_SAMPLES (SAMPLE_RATE * FRAME_MS / 1000)
#define HOP_LEN_SAMPLES   (SAMPLE_RATE * HOP_MS / 1000)
#define N_FFT_HALF        (N_FFT / 2 + 1)
#define PREEMPH_COEFF     0.97f

static float hamming_window[FRAME_LEN_SAMPLES];

static float ring_buffer[FRAME_LEN_SAMPLES + 1];
static int ring_pos = 0;
static int ring_count = 0;
static float prev_sample = 0.0f;

static float mfcc_results[N_FRAMES][N_MFCC];
static int frame_count = 0;
static float window_energy_sum = 0.0f;
static int window_sample_count = 0;

static float fft_real[N_FFT];
static float fft_imag[N_FFT];

static int bit_reverse(int x, int bits) {
    int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

static void compute_mfcc(float *windowed_frame, float *mfcc_out) {
    memset(fft_real, 0, sizeof(fft_real));
    memset(fft_imag, 0, sizeof(fft_imag));
    for (int i = 0; i < FRAME_LEN_SAMPLES; i++) {
        fft_real[i] = windowed_frame[i];
    }

    int log_n = 0;
    {
        int temp = N_FFT;
        while (temp > 1) { log_n++; temp >>= 1; }
    }

    for (int i = 0; i < N_FFT; i++) {
        int j = bit_reverse(i, log_n);
        if (i < j) {
            float t = fft_real[i]; fft_real[i] = fft_real[j]; fft_real[j] = t;
            t = fft_imag[i]; fft_imag[i] = fft_imag[j]; fft_imag[j] = t;
        }
    }

    for (int len = 2; len <= N_FFT; len *= 2) {
        float angle = -2.0f * 3.14159265f / len;
        float w_real = cosf(angle);
        float w_imag = sinf(angle);
        for (int i = 0; i < N_FFT; i += len) {
            float cur_w_real = 1.0f;
            float cur_w_imag = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int u = i + j;
                int v = i + j + len / 2;
                float t_real = cur_w_real * fft_real[v] - cur_w_imag * fft_imag[v];
                float t_imag = cur_w_real * fft_imag[v] + cur_w_imag * fft_real[v];
                fft_real[v] = fft_real[u] - t_real;
                fft_imag[v] = fft_imag[u] - t_imag;
                fft_real[u] += t_real;
                fft_imag[u] += t_imag;
                float new_w_real = cur_w_real * w_real - cur_w_imag * w_imag;
                float new_w_imag = cur_w_real * w_imag + cur_w_imag * w_real;
                cur_w_real = new_w_real;
                cur_w_imag = new_w_imag;
            }
        }
    }

    float power[N_FFT_HALF];
    for (int k = 0; k < N_FFT_HALF; k++) {
        power[k] = (fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k]) / N_FFT;
    }

    float mel_energies[N_MELS];
    for (int m = 0; m < N_MELS; m++) {
        float sum = 0.0f;
        for (int k = 0; k < N_FFT_HALF; k++) {
            sum += power[k] * MEL_FB_TABLE[m][k];
        }
        if (sum < 1e-20f) sum = 1e-20f;
        mel_energies[m] = logf(sum);
    }

    for (int k = 0; k < N_MFCC; k++) {
        float sum = 0.0f;
        for (int m = 0; m < N_MELS; m++) {
            sum += mel_energies[m] * DCT_TABLE[k][m];
        }
        mfcc_out[k] = sum;
    }
}

void mfcc_init(void) {
    for (int i = 0; i < FRAME_LEN_SAMPLES; i++) {
        hamming_window[i] = 0.54f - 0.46f * cosf(2.0f * 3.14159265f * i / (FRAME_LEN_SAMPLES - 1));
    }
    mfcc_reset();
}

void mfcc_reset(void) {
    ring_pos = 0;
    ring_count = 0;
    frame_count = 0;
    prev_sample = 0.0f;
    window_energy_sum = 0.0f;
    window_sample_count = 0;
    memset(ring_buffer, 0, sizeof(ring_buffer));
}

int mfcc_add_samples(const int16_t *samples, int n) {
    int got_frame = 0;
    for (int i = 0; i < n; i++) {
        float s = (float)samples[i] / 32768.0f;
        window_energy_sum += s * s;
        window_sample_count++;
        float filtered = s - PREEMPH_COEFF * prev_sample;
        prev_sample = s;

        ring_buffer[ring_pos] = filtered;
        ring_pos = (ring_pos + 1) % FRAME_LEN_SAMPLES;
        ring_count++;

        if (ring_count >= FRAME_LEN_SAMPLES && (ring_count - FRAME_LEN_SAMPLES) % HOP_LEN_SAMPLES == 0) {
            if (frame_count < N_FRAMES) {
                float windowed[FRAME_LEN_SAMPLES];
                for (int j = 0; j < FRAME_LEN_SAMPLES; j++) {
                    int idx = (ring_pos + j) % FRAME_LEN_SAMPLES;
                    windowed[j] = ring_buffer[idx] * hamming_window[j];
                }
                compute_mfcc(windowed, mfcc_results[frame_count]);
                frame_count++;
                got_frame = 1;
            }
        }
    }
    return got_frame;
}

int mfcc_is_ready(void) {
    return frame_count >= N_FRAMES;
}

float *mfcc_get_frame(int frame_idx) {
    if (frame_idx < 0 || frame_idx >= frame_count) return NULL;
    return mfcc_results[frame_idx];
}

float mfcc_get_window_rms(void) {
    if (window_sample_count == 0) return 0.0f;
    return sqrtf(window_energy_sum / window_sample_count) * 32768.0f;
}

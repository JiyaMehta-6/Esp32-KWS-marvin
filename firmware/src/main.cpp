#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "config.h"
#include "model_data.h"
#include "norm_stats.h"
#include "audio_i2s.h"
#include "mfcc_compute.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
static int8_t input_buffer[N_FRAMES * N_MFCC];

enum DetectState { STATE_IDLE, STATE_COOLDOWN };
static DetectState detect_state = STATE_IDLE;
static int64_t state_start_ms = 0;
static int inference_count = 0;
static int64_t total_inference_us = 0;

static void fill_input_buffer(int8_t *buf) {
    for (int f = 0; f < N_FRAMES; f++) {
        float *frame = mfcc_get_frame(f);
        for (int c = 0; c < N_MFCC; c++) {
            float val = (frame[c] - MFCC_MEAN[c]) / MFCC_STD[c];
            if (val > CLIP_MAX) val = CLIP_MAX;
            if (val < -CLIP_MAX) val = -CLIP_MAX;
            float qval = val / INPUT_SCALE + INPUT_ZERO_POINT;
            if (qval > 127.0f) qval = 127.0f;
            if (qval < -128.0f) qval = -128.0f;
            buf[f * N_MFCC + c] = (int8_t)(qval + (qval >= 0 ? 0.5f : -0.5f));
        }
    }
}

extern "C" void app_main(void) {
    printf("\n=== KWS Voice Activator (ESP-IDF) ===\n");
    printf("Free heap: %ld bytes\n", (long)esp_get_free_heap_size());
    printf("Model size: %d bytes (%.1f KB)\n", g_model_data_len, g_model_data_len / 1024.0f);
    printf("Threshold: %.2f | Energy gate: %.0f\n", DETECT_THRESHOLD, ENERGY_GATE);

    printf("[1] Loading model...\n");
    const tflite::Model *model = tflite::GetModel(g_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: schema version mismatch\n");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }
    printf("  Model OK\n");

    printf("[2] Building interpreter...\n");
    static tflite::MicroMutableOpResolver<10> resolver;
    if (resolver.AddConv2D() != kTfLiteOk) { printf("ERROR: AddConv2D\n"); while(1); }
    if (resolver.AddDepthwiseConv2D() != kTfLiteOk) { printf("ERROR: AddDepthwiseConv2D\n"); while(1); }
    if (resolver.AddMean() != kTfLiteOk) { printf("ERROR: AddMean\n"); while(1); }
    if (resolver.AddFullyConnected() != kTfLiteOk) { printf("ERROR: AddFullyConnected\n"); while(1); }
    if (resolver.AddReshape() != kTfLiteOk) { printf("ERROR: AddReshape\n"); while(1); }
    if (resolver.AddRelu() != kTfLiteOk) { printf("ERROR: AddRelu\n"); while(1); }
    if (resolver.AddLogistic() != kTfLiteOk) { printf("ERROR: AddLogistic\n"); while(1); }
    if (resolver.AddQuantize() != kTfLiteOk) { printf("ERROR: AddQuantize\n"); while(1); }
    if (resolver.AddDequantize() != kTfLiteOk) { printf("ERROR: AddDequantize\n"); while(1); }
    if (resolver.AddBatchMatMul() != kTfLiteOk) { printf("ERROR: AddBatchMatMul\n"); while(1); }
    static tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("ERROR: AllocateTensors failed\n");
        while (1) { vTaskDelay(pdMS_TO_TICKS(1000)); }
    }

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);
    printf("  Input: [%d,%d,%d,%d] type=%d\n",
           input->dims->data[0], input->dims->data[1],
           input->dims->data[2], input->dims->data[3], input->type);
    printf("  Output: [%d,%d] type=%d\n",
           output->dims->data[0], output->dims->data[1], output->type);
    printf("Free heap after alloc: %ld bytes\n", (long)esp_get_free_heap_size());

    printf("[3] Initializing audio pipeline...\n");
    audio_i2s_init();
    mfcc_init();
    printf("  Audio pipeline OK\n");

    printf("[4] Starting listen loop...\n\n");
    detect_state = STATE_IDLE;

    int16_t hop_buf[HOP_LEN];

    while (1) {
        int n_read = audio_i2s_read(hop_buf, HOP_LEN);
        if (n_read <= 0) continue;

        mfcc_add_samples(hop_buf, n_read);

        if (!mfcc_is_ready()) continue;

        float rms = mfcc_get_window_rms();
        if (rms < ENERGY_GATE) {
            mfcc_reset();
            continue;
        }

        int64_t t0 = esp_timer_get_time();
        fill_input_buffer(input_buffer);
        int64_t t1 = esp_timer_get_time();

        TfLiteTensor *in_tensor = interpreter.input(0);
        TfLiteTensor *out_tensor = interpreter.output(0);
        memcpy(in_tensor->data.int8, input_buffer, N_FRAMES * N_MFCC * sizeof(int8_t));

        if (interpreter.Invoke() != kTfLiteOk) {
            printf("ERROR: Invoke failed\n");
            continue;
        }

        int64_t t2 = esp_timer_get_time();
        int64_t elapsed = t2 - t0;
        int64_t quant_us = t1 - t0;
        int64_t infer_us = t2 - t1;
        total_inference_us += elapsed;
        inference_count++;

        int drained = 0;
        while (drained < 8) {
            int n = audio_i2s_read(hop_buf, HOP_LEN);
            if (n <= 0) break;
            drained++;
        }
        mfcc_reset();

        int8_t raw_out = out_tensor->data.int8[0];
        float sigmoid_score = (raw_out - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;

        float marvin_score = sigmoid_score;
        bool marvin_detected = marvin_score >= DETECT_THRESHOLD;

        int64_t now_ms = esp_timer_get_time() / 1000;

        static int8_t score_ring[DETECT_WINDOW_N] = {0};
        static int ring_idx = 0;

        score_ring[ring_idx] = marvin_detected ? 1 : 0;
        ring_idx = (ring_idx + 1) % DETECT_WINDOW_N;

        int window_sum = 0;
        for (int i = 0; i < DETECT_WINDOW_N; i++) window_sum += score_ring[i];

        switch (detect_state) {
            case STATE_IDLE:
                if (window_sum >= DETECT_CONFIRM_K) {
                    detect_state = STATE_COOLDOWN;
                    state_start_ms = now_ms;
                    printf("*** DETECTED *** marvin=%.3f score=%.3f window=%d/%d infer=%lldus ***\n",
                           marvin_score, sigmoid_score, window_sum, DETECT_WINDOW_N, (long long)elapsed);
                }
                break;

            case STATE_COOLDOWN:
                if (now_ms - state_start_ms > COOLDOWN_MS) {
                    detect_state = STATE_IDLE;
                    for (int i = 0; i < DETECT_WINDOW_N; i++) score_ring[i] = 0;
                    ring_idx = 0;
                }
                break;

            default:
                break;
        }

        if (inference_count % 10 == 0) {
            float avg_us = (float)total_inference_us / inference_count;
            printf("  m=%.3f rms=%.0f | quant=%lldus infer=%lldus total=%lldus avg=%.0fus drained=%d | state=%d win=%d\n",
                   sigmoid_score, rms, (long long)quant_us, (long long)infer_us,
                   (long long)elapsed, avg_us, drained, detect_state, window_sum);
        }
    }
}

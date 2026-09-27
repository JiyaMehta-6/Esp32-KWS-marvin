#ifndef CONFIG_H
#define CONFIG_H

#define SAMPLE_RATE       16000
#define FRAME_MS          25
#define HOP_MS            10
#define FRAME_LEN         (SAMPLE_RATE * FRAME_MS / 1000)
#define HOP_LEN           (SAMPLE_RATE * HOP_MS / 1000)
#define N_FFT             512
#define N_MELS            26
#define N_MFCC            12
#define N_FRAMES          49
#define N_CLASSES         3

#define I2S_PORT          I2S_NUM_0
#define I2S_BCLK_PIN      26
#define I2S_WS_PIN        25
#define I2S_SD_PIN        33

#define LED_PIN           2
#define LED_BRIGHTNESS    50

#define DETECT_THRESHOLD  0.900f
#define ENERGY_GATE        150.0f
#define DETECT_CONFIRM_K  3
#define DETECT_WINDOW_N   5
#define COOLDOWN_MS       3000
#define TENSOR_ARENA_SIZE (32 * 1024)

#define CLASS_MARVIN      0
#define CLASS_NOT_KW      1
#define CLASS_UNKNOWN     2

#define INPUT_SCALE       0.031373f
#define INPUT_ZERO_POINT  -1
#define OUTPUT_SCALE      0.003906f
#define OUTPUT_ZERO_POINT -128
#define TEMPERATURE       1.0f

#endif

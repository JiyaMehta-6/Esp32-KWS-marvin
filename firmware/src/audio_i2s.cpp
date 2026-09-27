#include "audio_i2s.h"
#include "driver/i2s_std.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

static i2s_chan_handle_t rx_handle = NULL;

void audio_i2s_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = 240;
    i2s_new_channel(&chan_cfg, NULL, &rx_handle);

    i2s_std_slot_config_t slot_cfg =
        I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
    slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    slot_cfg.ws_width = I2S_SLOT_BIT_WIDTH_16BIT;
    slot_cfg.ws_pol = false;
    slot_cfg.bit_shift = true;

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = slot_cfg,
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)I2S_BCLK_PIN,
            .ws = (gpio_num_t)I2S_WS_PIN,
            .dout = I2S_GPIO_UNUSED,
            .din = (gpio_num_t)I2S_SD_PIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    i2s_channel_init_std_mode(rx_handle, &std_cfg);
    i2s_channel_enable(rx_handle);
}

int audio_i2s_read(int16_t *buf, int max_samples) {
    size_t bytes_read = 0;
    i2s_channel_read(rx_handle, buf, max_samples * sizeof(int16_t), &bytes_read, pdMS_TO_TICKS(500));
    return bytes_read / sizeof(int16_t);
}

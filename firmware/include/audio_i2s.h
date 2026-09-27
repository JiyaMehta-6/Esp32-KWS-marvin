#ifndef AUDIO_I2S_H
#define AUDIO_I2S_H

#include <stdint.h>
#include "config.h"

void audio_i2s_init(void);
int audio_i2s_read(int16_t *buf, int max_samples);

#endif

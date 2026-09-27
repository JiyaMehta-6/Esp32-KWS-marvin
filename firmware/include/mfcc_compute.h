#ifndef MFCC_COMPUTE_H
#define MFCC_COMPUTE_H

#include <stdint.h>

void mfcc_init(void);
void mfcc_reset(void);
int mfcc_add_samples(const int16_t *samples, int n);
int mfcc_is_ready(void);
float *mfcc_get_frame(int frame_idx);
float mfcc_get_window_rms(void);

#endif

# Architecture Details

## DS-CNN Model (v18)

### Input
- Shape: `(49, 12, 1)` — 49 MFCC frames × 12 coefficients × 1 channel
- Sample rate: 16kHz, 25ms frames, 10ms hop

### Layers
| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Input | (49, 12, 1) | 0 |
| Conv2D(32, 3×3, stride 2×2) | (25, 6, 32) | 320 |
| BatchNorm | (25, 6, 32) | 128 |
| ReLU | (25, 6, 32) | 0 |
| DepthwiseConv2D(32, 3×3, stride 2×2) | (13, 3, 32) | 320 |
| BatchNorm | (13, 3, 32) | 128 |
| ReLU | (13, 3, 32) | 0 |
| DepthwiseConv2D(32, 3×3, stride 2×2) | (7, 2, 32) | 320 |
| BatchNorm | (7, 2, 32) | 128 |
| ReLU | (7, 2, 32) | 0 |
| DepthwiseConv2D(32, 3×3, stride 2×2) | (4, 1, 32) | 320 |
| BatchNorm | (4, 1, 32) | 128 |
| ReLU | (4, 1, 32) | 0 |
| GlobalAvgPool | (32) | 0 |
| Dense(128) + ReLU | (128) | 4,224 |
| Dropout(0.3) | (128) | 0 |
| Dense(1) + Sigmoid | (1) | 129 |

**Total**: 6,145 parameters

### Quantization
- INT8 post-training quantization
- Input scale: 0.031373, zero point: -1
- Output scale: 0.003906, zero point: -128
- Model size: 18,344 bytes (17.9 KB)

## MFCC Pipeline

1. **Pre-emphasis**: y[n] = x[n] - 0.97 × x[n-1]
2. **Framing**: 25ms Hamming-windowed frames
3. **FFT**: 512-point FFT
4. **Mel Filterbank**: 26 triangular filters
5. **Log Energy**: log(mel_energies)
6. **DCT**: 12 MFCCs (k=1..12, skip k=0)
7. **Normalization**: (mfcc - mean) / std, clip to [-4, 4]

## Detection Pipeline

1. I2S DMA captures 16kHz mono audio (10ms chunks)
2. MFCC extracts features over 490ms window (49 frames)
3. Energy gate skips silent windows (RMS < 150)
4. DS-CNN predicts keyword probability
5. Sliding window (3/5) confirms detection
6. Cooldown (3s) prevents re-triggering

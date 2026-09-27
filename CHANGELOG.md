# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v18.0.0] - 2024-09-27

### Added
- **DS-CNN v18 model** — 32× marvin oversampling, focal loss (α=0.70, γ=2.0)
- **Rich MFCC augmentation** — Gaussian noise, time/freq masking, mixup, coefficient dropout
- **INT8 post-training quantization** — 18 KB model (18,344 bytes)
- **Energy gate** — Skip inference when RMS < 150 (filters electrical noise)
- **Sliding window voting** — 3-of-5 confirmation reduces false triggers
- **Cooldown state machine** — 3s cooldown prevents re-triggering
- **Precomputed mel/DCT tables** — Zero-runtime FFT/mel/DCT on ESP32
- **Golden reference test** — Embedded marvin sample validates pipeline at boot

### Model Performance
| Metric | Value |
|--------|-------|
| Accuracy (test) | 98.3% |
| Recall @ FAR≤1% | 66.7% (th=0.96) / 82.1% (th=0.90) |
| Inference time | ~19ms @ 240MHz |
| Model size | 18,344 bytes (INT8) |
| Parameters | 6,145 |

### Build Constraints Met
- ✅ Accuracy >95% (98.3%)
- ✅ Inference <20ms (~19ms)
- ✅ Flash <350KB (~321KB)
- ✅ RAM idle <256KB (~150KB)
- ✅ FAR <1% (0.31% at th=0.90)
- ✅ End-to-end <300ms (~40ms)

### Fixed
- GPIO2 LED interference resolved (serial-only output)
- MFCC normalization bug (k=1..12 vs k=0..11) fixed
- Capture mode UART blocking issue fixed (non-blocking stdin)

---

## [v17.0.0] - 2024-09-20

### Added
- 16× marvin oversampling
- Basic MFCC augmentation (Gaussian noise, time shift)
- Focal loss with α=0.65
- SpecAugment (time/freq masking)
- Early stopping with cosine LR decay

### Changed
- Migration from Arduino to ESP-IDF 6.1.0
- TFLite Micro ops resolver updated

---

## [v16.0.0] - 2024-09-15

### Added
- QAT (Quantization-Aware Training) pipeline
- INT8 quantization with representative dataset
- Model export to C header (`model_data.h`)
- PlatformIO build system

---

## [v15.0.0] - 2024-09-10

### Added
- Binary classification (marvin vs not_keyword)
- DS-CNN architecture with depthwise separable convolutions
- Google Speech Commands V2 data preparation
- Stratified train/val/test splits

---

## [v1.0.0] - 2024-08-01

### Added
- Initial keyword spotting prototype
- Basic CNN model
- INMP441 I2S driver
- MFCC feature extraction on ESP32

---

## Upgrade Guide

### From v17 to v18
1. **Model**: Replace `model_data.h` with v18 INT8 model
2. **Config**: Update `DETECT_THRESHOLD=0.900f`, `ENERGY_GATE=150.0f`
3. **Threshold**: Recommended th=0.90 (was 0.96 for v17)
4. **Retrain**: Run `python train_v18.py` for custom data

### Breaking Changes
- v18 uses different normalization stats (regenerate `norm_stats.h`)
- v18 MFCC pipeline expects energy gate; disable if not wanted

---

## Migration Notes

### v16 → v17
- QAT training now produces smaller INT8 models
- PlatformIO required for firmware build

### v15 → v16
- Switched from multiclass to binary classification
- Model architecture simplified (DS-CNN only)

---

## Links

- [v18 Release](https://github.com/your-repo/releases/tag/v18.0.0)
- [Training Script](training/train_v18.py)
- [Export Tool](tools/export_model.py)
- [Architecture Docs](docs/architecture.md)
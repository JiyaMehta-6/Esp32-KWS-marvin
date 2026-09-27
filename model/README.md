# Pre-trained Models

Place your `.tflite` model files here.

## Recommended Model

- **model_binary_int8_v18.tflite** — Best balance of accuracy and size

## How to Generate

1. Train: `cd ../training && python train_v18.py`
2. Quantize: `python quantize_int8.py`
3. Export: `python export_model.py`
4. Copy `model_data.h` to `../firmware/include/`

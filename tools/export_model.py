#!/usr/bin/env python3
"""
Export trained Keras model to INT8 TFLite and C header.

Usage:
    python export_model.py --model checkpoints/best_binary_float32_v18.keras --output firmware/include/model_data.h
"""
import os
import argparse
import numpy as np
import tensorflow as tf

os.environ["TF_USE_LEGACY_KERAS"] = "1"


def representative_dataset(n=100):
    for _ in range(n):
        yield [np.random.randn(1, 49, 12, 1).astype(np.float32)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .keras model')
    parser.add_argument('--output', type=str, required=True, help='Output path for model_data.h')
    parser.add_argument('--tflite_out', type=str, default=None, help='Also save .tflite file')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print("Converting to INT8 TFLite...")
    tflite_model = converter.convert()

    print(f"Model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)")

    if args.tflite_out:
        with open(args.tflite_out, 'wb') as f:
            f.write(tflite_model)
        print(f"Saved TFLite: {args.tflite_out}")

    # Generate C header
    var_name = "g_model_data"
    hex_array = ', '.join(f'0x{b:02x}' for b in tflite_model)
    header = f"""// Auto-generated model data. Do not edit.
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

#define {var_name}_len {len(tflite_model)}

alignas(16) const unsigned char {var_name}[] = {{
{hex_array}
}};

#endif
"""

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(header)
    print(f"Saved C header: {args.output}")
    print(f"  #define {var_name}_len {len(tflite_model)}")


if __name__ == '__main__':
    main()

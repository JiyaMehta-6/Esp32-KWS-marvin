#!/usr/bin/env python3
"""
Generate mel filterbank and DCT tables for ESP32 firmware.

Usage:
    python generate_tables.py --output ../firmware/include/mel_dct_tables.h
"""
import os
import argparse
import numpy as np
import scipy.fftpack

N_MELS = 26
N_MFCC = 12
N_FFT = 512
SAMPLE_RATE = 16000


def create_mel_filterbank():
    high_mel = 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700)
    mel_points = np.linspace(0, high_mel, N_MELS + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((N_FFT + 1) * hz_points / SAMPLE_RATE).astype(int)
    fbank = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fbank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    fbank = create_mel_filterbank()
    dct_full = scipy.fftpack.dct(np.eye(N_MELS), type=2, axis=1, norm='ortho')
    dct_table = dct_full.T[1:N_MFCC + 1, :]

    nfft_half = N_FFT // 2 + 1

    lines = [
        "#ifndef MEL_DCT_TABLES_H",
        "#define MEL_DCT_TABLES_H",
        "",
        f"static const float MEL_FB_TABLE[{N_MELS}][{nfft_half}] = {{",
    ]
    for m in range(N_MELS):
        vals = ', '.join(f'{v:.8e}' for v in fbank[m])
        comma = ',' if m < N_MELS - 1 else ''
        lines.append(f"    {{{vals}}}{comma}")
    lines.append("};")
    lines.append("")

    lines.append(f"static const float DCT_TABLE[{N_MFCC}][{N_MELS}] = {{")
    for k in range(N_MFCC):
        vals = ', '.join(f'{v:.8e}' for v in dct_table[k])
        comma = ',' if k < N_MFCC - 1 else ''
        lines.append(f"    {{{vals}}}{comma}")
    lines.append("};")
    lines.append("")
    lines.append("#endif")
    lines.append("")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Generated: {args.output}")
    print(f"  MEL_FB_TABLE: [{N_MELS}][{nfft_half}]")
    print(f"  DCT_TABLE: [{N_MFCC}][{N_MELS}]")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
KWS v18: DS-CNN Keyword Spotting for "marvin" detection.

Train on Google Speech Commands V2 with 32x oversampling,
focal loss, and MFCC augmentation.

Usage:
    python train_v18.py --data_root /path/to/speech_commands
"""
import os
import argparse
import json
import random

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, losses
from pathlib import Path
import scipy.signal
import scipy.io.wavfile as wavfile
import scipy.fftpack

SAMPLE_RATE = 16000
N_MFCC_TOTAL = 13
N_MFCC = 12
N_FFT = 512
N_MELS = 26
N_FRAMES = 49
BATCH_SIZE = 64
EPOCHS = 200
LR = 5e-4
CLIP_MAX = 4.0
LABEL_SMOOTHING = 0.05

WORDS_SIMILAR = [
    'follow', 'forward', 'visual', 'bird', 'bed', 'cat', 'dog',
    'boy', 'girl', 'happy', 'house', 'sheila', 'tree',
    'wow', 'yes', 'no', 'on', 'off', 'up', 'down', 'left', 'right',
    'go', 'stop'
]

_mel_fb_cache = None


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def create_mel_filterbank():
    global _mel_fb_cache
    if _mel_fb_cache is not None:
        return _mel_fb_cache
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, N_MELS + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin_points = np.floor((N_FFT + 1) * hz_points / SAMPLE_RATE).astype(int)
    fbank = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus) if f_m != f_m_minus else 0
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m) if f_m_plus != f_m else 0
    _mel_fb_cache = fbank
    return fbank


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def mfcc_comput(wav, sr=SAMPLE_RATE):
    wav = preemphasis(wav)
    frame_len = int(sr * 25 / 1000)
    hop_len = int(sr * 10 / 1000)
    n_frames = 1 + max(0, (len(wav) - frame_len) // hop_len)
    if n_frames == 0:
        n_frames = 1
    pad_len = (n_frames - 1) * hop_len + frame_len
    if len(wav) < pad_len:
        wav = np.pad(wav, (0, pad_len - len(wav)), mode='constant')
    frames = np.zeros((n_frames, frame_len))
    for i in range(n_frames):
        frames[i] = wav[i * hop_len:i * hop_len + frame_len]
    frames *= np.hamming(frame_len)
    mag_frames = np.abs(np.fft.rfft(frames, N_FFT))
    pow_frames = (mag_frames ** 2) / N_FFT
    mel_energies = np.dot(pow_frames, create_mel_filterbank().T)
    mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
    log_mel = np.log(mel_energies)
    mfcc = scipy.fftpack.dct(log_mel, type=2, axis=1, norm='ortho')[:, :N_MFCC_TOTAL]
    return mfcc[:, 1:].astype(np.float32)


def load_wav_mfcc(item, data_root, speed_perturb=False):
    wav_path = data_root / item["path"]
    if not wav_path.exists():
        return np.zeros((N_FRAMES, N_MFCC), dtype=np.float32)
    sr, wav = wavfile.read(wav_path)
    if len(wav.shape) > 1:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32) / 32768.0
    if sr != SAMPLE_RATE:
        wav = scipy.signal.resample(wav, int(len(wav) * SAMPLE_RATE / sr))
    if speed_perturb and random.random() < 0.5:
        rate = random.uniform(0.82, 1.18)
        wav = scipy.signal.resample(wav, int(len(wav) / rate))
    if len(wav) > SAMPLE_RATE:
        wav = wav[:SAMPLE_RATE]
    elif len(wav) < SAMPLE_RATE:
        wav = np.pad(wav, (0, SAMPLE_RATE - len(wav)))
    mfcc = mfcc_comput(wav)
    if mfcc.shape[0] > N_FRAMES:
        mfcc = mfcc[:N_FRAMES]
    elif mfcc.shape[0] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, N_FRAMES - mfcc.shape[0]), (0, 0)))
    return mfcc


def augment(mfcc):
    x = mfcc.copy()
    if random.random() < 0.5:
        x *= random.uniform(0.5, 1.5)
    if random.random() < 0.5:
        shift = random.randint(-7, 7)
        x = np.roll(x, shift, axis=0)
    if random.random() < 0.6:
        level = random.choice([0.05, 0.1, 0.15])
        x += np.random.randn(*x.shape).astype(np.float32) * level
    if random.random() < 0.5:
        t = random.randint(1, 10)
        t0 = random.randint(0, max(0, N_FRAMES - t))
        x[t0:t0 + t, :] = 0
    if random.random() < 0.5:
        f = random.randint(1, 6)
        f0 = random.randint(0, max(0, N_MFCC - f))
        x[:, f0:f0 + f] = 0
    return x


def mixup(x1, x2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    return (lam * x1 + (1 - lam) * x2).astype(np.float32)


def focal_loss(gamma=2.0, alpha=0.70):
    bce = losses.BinaryCrossentropy(from_logits=False, reduction='none')
    def loss(y_true, y_pred):
        y_true_s = y_true * (1.0 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        p = y_pred
        p_t = y_true * p + (1 - y_true) * (1 - p)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_t * tf.pow(1.0 - p_t, gamma) * bce(y_true_s, p)
    return loss


def build_model():
    inputs = keras.Input(shape=(N_FRAMES, N_MFCC, 1), name="input")
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', name='dwconv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', name='dwconv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', name='dwconv4')(x)
    x = layers.BatchNormalization(name='bn4')(x)
    x = layers.ReLU(name='relu4')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.3, name='dropout')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='sigmoid')(x)
    return keras.Model(inputs, outputs, name='ds_cnn_kws_v18')


def eval_recall(model, val_mfccs, val_labels):
    preds = model.predict(val_mfccs, batch_size=256, verbose=0).flatten()
    m_mask = val_labels == 1
    nm_mask = val_labels == 0
    for t in np.arange(0.005, 0.995, 0.005):
        fp = ((preds >= t) & nm_mask).sum()
        tn = ((preds < t) & nm_mask).sum()
        far = fp / max(1, fp + tn)
        if far <= 0.01:
            tp = ((preds >= t) & m_mask).sum()
            fn = ((preds < t) & m_mask).sum()
            recall = tp / max(1, tp + fn)
            return recall, t
    return 0, 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Google Speech Commands directory')
    parser.add_argument('--splits_dir', type=str, default=None,
                        help='Path to splits directory (default: data_root/../splits)')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for models')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir) if args.splits_dir else data_root.parent / 'splits'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    print("Loading splits...")
    train_items = load_jsonl(splits_dir / "train.jsonl")
    val_items = load_jsonl(splits_dir / "val.jsonl")
    test_items = load_jsonl(splits_dir / "test.jsonl")

    for item in train_items:
        item["bl"] = 1 if item["label"] == 0 else 0
    for item in val_items:
        item["bl"] = 1 if item["label"] == 0 else 0
    for item in test_items:
        item["bl"] = 1 if item["label"] == 0 else 0

    train_marvin = [i for i in train_items if i["bl"] == 1]
    train_not = [i for i in train_items if i["bl"] == 0]
    hard_neg = [i for i in train_not
                if any(w in i.get("word", "") for w in WORDS_SIMILAR)]
    other_neg = [i for i in train_not
                 if not any(w in i.get("word", "") for w in WORDS_SIMILAR)]
    random.shuffle(hard_neg)
    random.shuffle(other_neg)

    balanced = (
        train_marvin * 32 +
        hard_neg[:min(len(hard_neg), len(train_marvin) * 5)] * 2 +
        other_neg[:min(len(other_neg), len(train_marvin) * 3)]
    )
    random.shuffle(balanced)
    n_marvin = sum(1 for it in balanced if it["bl"] == 1)
    n_not = sum(1 for it in balanced if it["bl"] == 0)
    print(f"Balanced: {len(balanced)} (marvin={n_marvin}, not={n_not}, ratio=1:{n_not//max(1,n_marvin):.1f})")

    print("Preloading train MFCCs...")
    train_mfccs = np.zeros((len(balanced), N_FRAMES, N_MFCC, 1), dtype=np.float32)
    train_labels = np.zeros(len(balanced), dtype=np.float32)
    for i, item in enumerate(balanced):
        train_mfccs[i, :, :, 0] = load_wav_mfcc(item, data_root, speed_perturb=(item["bl"] == 1))
        train_labels[i] = float(item["bl"])
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(balanced)}")

    mfcc_flat = train_mfccs[:, :, :, 0].reshape(-1, N_MFCC)
    mfcc_mean = mfcc_flat.mean(axis=0)
    mfcc_std = mfcc_flat.std(axis=0)
    mfcc_std = np.where(mfcc_std < 1e-6, 1.0, mfcc_std)
    train_mfccs[:, :, :, 0] = ((train_mfccs[:, :, :, 0].reshape(-1, N_MFCC) - mfcc_mean) / mfcc_std).reshape(-1, N_FRAMES, N_MFCC)
    train_mfccs = np.clip(train_mfccs, -CLIP_MAX, CLIP_MAX)
    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump({'mean': mfcc_mean.tolist(), 'std': mfcc_std.tolist(), 'clip_max': CLIP_MAX}, f, indent=2)

    print("Preloading val MFCCs...")
    val_mfccs = np.zeros((len(val_items), N_FRAMES, N_MFCC, 1), dtype=np.float32)
    val_labels = np.zeros(len(val_items), dtype=np.float32)
    for i, item in enumerate(val_items):
        val_mfccs[i, :, :, 0] = load_wav_mfcc(item, data_root)
        val_labels[i] = float(item["bl"])
    val_mfccs[:, :, :, 0] = ((val_mfccs[:, :, :, 0].reshape(-1, N_MFCC) - mfcc_mean) / mfcc_std).reshape(-1, N_FRAMES, N_MFCC)
    val_mfccs = np.clip(val_mfccs, -CLIP_MAX, CLIP_MAX)

    print("\n=== Training v18: 32x oversampling + rich augmentation ===")
    model = build_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.70),
        metrics=['accuracy']
    )
    model.summary()

    best_recall = 0
    best_epoch = -1
    patience = 0
    steps = len(balanced) // BATCH_SIZE
    marvin_idx = np.where(train_labels == 1)[0]

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        indices = np.random.permutation(len(balanced))
        epoch_loss = 0
        for step in range(steps):
            start = step * BATCH_SIZE
            batch_idx = indices[start:start + BATCH_SIZE]
            batch_x = np.zeros((len(batch_idx), N_FRAMES, N_MFCC, 1), dtype=np.float32)
            for j, idx in enumerate(batch_idx):
                batch_x[j] = augment(train_mfccs[idx].copy())
            for j in range(len(batch_idx)):
                if random.random() < 0.3 and train_labels[batch_idx[j]] == 1:
                    mi = random.choice(marvin_idx)
                    batch_x[j] = mixup(batch_x[j], train_mfccs[mi].copy(), alpha=0.4)
            batch_y = train_labels[batch_idx]
            loss, acc = model.train_on_batch(batch_x, batch_y)
            epoch_loss += loss

        recall, thresh = eval_recall(model, val_mfccs, val_labels)
        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            patience = 0
            model.save(output_dir / "best_binary_float32_v18.keras")
            print(f"Epoch {epoch+1}/{EPOCHS} loss={epoch_loss/steps:.4f} recall={recall:.4f} th={thresh:.3f} **BEST**")
        else:
            patience += 1
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} loss={epoch_loss/steps:.4f} recall={recall:.4f}")
            if patience >= 40:
                print("Early stopping")
                break
        if epoch > 0:
            keras.backend.set_value(model.optimizer.learning_rate,
                                   LR * 0.5 * (1 + np.cos(np.pi * epoch / EPOCHS)))

    print(f"\nBest recall@FAR<=1%={best_recall:.4f} at epoch {best_epoch+1}")
    print(f"\nModel saved to: {output_dir / 'best_binary_float32_v18.keras'}")
    print("Done!")


if __name__ == "__main__":
    main()

"""Load the semi-simulated ocular-artifact benchmark (Klados & Bamidis)."""

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

from models.config import FS, N_CHANNELS, N_RECORDINGS, T, WINDOWS_PER_RECORDING


def filter_eeg(eeg, low=0.5, high=40.0, fs=FS):
    b, a = butter(5, [low / fs * 2, high / fs * 2], "bandpass")
    return filtfilt(b, a, eeg)


def standardize_pair(clean, noisy):
    clean = clean.astype(np.float64, copy=True)
    noisy = noisy.astype(np.float64, copy=True)
    for ch in range(clean.shape[0]):
        clean[ch] = clean[ch] - np.mean(clean[ch])
        noisy[ch] = noisy[ch] - np.mean(noisy[ch])
        denom = np.std(noisy[ch]) + 1.0e-8
        clean[ch] = clean[ch] / denom
        noisy[ch] = noisy[ch] / denom
    return clean.astype(np.float32), noisy.astype(np.float32)


def load_current19(clean_mat, contaminated_mat):
    """Return clean/noisy windows [540, 19, 500] and recording ids [540]."""
    pure = loadmat(str(clean_mat))
    contaminated = loadmat(str(contaminated_mat))
    x_parts, y_parts, ids = [], [], []
    for n in range(1, N_RECORDINGS + 1):
        clean = np.asarray(pure[f"sim{n}_resampled"])[:, :5000].copy()
        noisy = np.asarray(contaminated[f"sim{n}_con"])[:, :5000].copy()
        if clean.shape != (N_CHANNELS, 5000) or noisy.shape != (N_CHANNELS, 5000):
            raise ValueError(f"Unexpected shape for recording {n}")
        for ch in range(N_CHANNELS):
            clean[ch] = filter_eeg(clean[ch])
            noisy[ch] = filter_eeg(noisy[ch])
        clean, noisy = standardize_pair(clean, noisy)
        x_parts.append(clean.reshape(N_CHANNELS, WINDOWS_PER_RECORDING, T).transpose(1, 0, 2))
        y_parts.append(noisy.reshape(N_CHANNELS, WINDOWS_PER_RECORDING, T).transpose(1, 0, 2))
        ids.append(np.full(WINDOWS_PER_RECORDING, n - 1, dtype=np.int64))
    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    rid = np.concatenate(ids, axis=0)
    return x, y, rid


def rotating_splits(n_recordings=N_RECORDINGS, n_splits=10):
    """Rotating validation-development split used by the paper."""
    indices = np.arange(n_recordings, dtype=int)
    blocks = [b.astype(int) for b in np.array_split(indices, n_splits)]
    val_count = max(1, int(n_recordings * 0.1))
    all_ids = set(int(x) for x in indices)
    train_list, val_list, test_list = [], [], []
    for fold in range(n_splits):
        test_ids = [int(x) for x in blocks[fold].tolist()]
        val_ids = [int(x) for x in blocks[(fold + 1) % n_splits][:val_count].tolist()]
        train_ids = sorted(all_ids.difference(test_ids).difference(val_ids))
        train_list.append(np.asarray(train_ids, dtype=int))
        val_list.append(np.asarray(val_ids, dtype=int))
        test_list.append(np.asarray(test_ids, dtype=int))
    return train_list, val_list, test_list


def select_windows(arrays, recording_ids, recording_list):
    idx = np.concatenate([np.where(recording_ids == r)[0] for r in recording_list])
    return [a[idx] for a in arrays]

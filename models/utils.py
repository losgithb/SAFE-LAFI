"""Numeric helpers and paper evaluation metrics."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import EPS, FS, POST, VIEWS


def rms(x, axis=None, keepdims=False):
    return np.sqrt(np.mean(np.square(x), axis=axis, keepdims=keepdims) + EPS)


def rms64(x, axis=None, keepdims=False):
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(np.square(x), axis=axis, keepdims=keepdims) + EPS)


def pearson_flat(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a) * np.sum(b * b)) + EPS))


def corr_rows(a, b):
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return np.sum(a * b, axis=1) / (np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)) + EPS)


def moving_average(x, width=25):
    if width <= 1:
        return x.copy()
    pad = width // 2
    kernel = np.ones(width, dtype=np.float32) / float(width)
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="edge")
    return np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="valid"), axis=-1, arr=xp
    ).astype(np.float32)


def view_rms(arr, view_name):
    return rms64(arr[:, VIEWS[view_name], :], axis=(1, 2))


def atom_energy(atom, view_name):
    return float(rms64(atom[VIEWS[view_name], :]))


def roc_auc(y_true, score):
    y = y_true.astype(int).reshape(-1)
    s = score.reshape(-1)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = np.sum(pos[:, None] > neg[None, :])
    ties = np.sum(pos[:, None] == neg[None, :])
    return float((wins + 0.5 * ties) / (pos.size * neg.size))


def bandpass(x, lo=8.0, hi=30.0, fs=FS):
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x.astype(np.float64), axis=-1)


def paper_metrics(y, x, x_hat):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    num = np.sum(np.linalg.norm(x_hat - x, axis=(1, 2)) ** 2)
    den = np.sum(np.linalg.norm(x, axis=(1, 2)) ** 2)
    rrmse = float(np.sqrt(num / den))
    cc = float(np.mean([
        np.corrcoef(x_hat[i, c], x[i, c])[0, 1]
        for i in range(y.shape[0]) for c in range(x.shape[1])
    ]))

    def snr(a, b):
        return 10.0 * np.log10(
            np.sum(np.linalg.norm(a, axis=(1, 2)) ** 2)
            / (np.sum(np.linalg.norm(a - b, axis=(1, 2)) ** 2) + EPS)
        )

    snr_hat = float(snr(x, x_hat))
    snr_raw = float(snr(x, y))
    return {
        "rrmse": rrmse,
        "cc": cc,
        "snr": snr_hat,
        "delta_snr_db": snr_hat - snr_raw,
    }


def posterior_over(y, x, x_hat):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    r = y - x_hat
    r_true = y - x
    num = np.linalg.norm(np.maximum(np.abs(r[:, POST]) - np.abs(r_true[:, POST]), 0.0))
    den = np.linalg.norm(r_true[:, POST]) + EPS
    return float(num / den)


def mi_band_harm(y, x, x_hat, fs=200):
    """Change in 8-30 Hz residual error relative to the raw input, in points."""
    num = np.linalg.norm(bandpass(x_hat, fs=fs) - bandpass(x, fs=fs), axis=(1, 2)) - \
        np.linalg.norm(bandpass(y, fs=fs) - bandpass(x, fs=fs), axis=(1, 2))
    den = np.linalg.norm(bandpass(y, fs=fs) - bandpass(x, fs=fs), axis=(1, 2))
    with np.errstate(invalid="ignore"):
        ratio = num / np.where(den > EPS, den, np.nan)
    return float(np.nanmean(ratio)) * 100.0


def full_metrics(y, x, x_hat):
    return {
        **paper_metrics(y, x, x_hat),
        "over": posterior_over(y, x, x_hat),
        "mi_points": mi_band_harm(y, x, x_hat),
    }

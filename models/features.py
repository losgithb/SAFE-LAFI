"""y-only feature construction and ridge helpers."""

import numpy as np

from .config import VIEWS
from .basis import first_pc_waveforms
from .utils import moving_average, rms


def make_yonly_features(y, clean_priors=None):
    frontal7, high5 = VIEWS["frontal7"], VIEWS["high_frontal5"]
    fp_pair, post5 = VIEWS["fp_pair"], VIEWS["posterior5"]
    feats = []
    feats.append(y[:, frontal7, ::10].reshape(y.shape[0], -1))
    fp_mean = np.mean(y[:, fp_pair, :], axis=1)
    fp_diff = y[:, 0, :] - y[:, 1, :]
    high_mean = np.mean(y[:, high5, :], axis=1)
    feats.append(fp_mean[:, ::5])
    feats.append(fp_diff[:, ::5])
    feats.append(high_mean[:, ::5])
    smooth_fp = moving_average(fp_mean[:, None, :], width=25)[:, 0, :]
    smooth_high = moving_average(high_mean[:, None, :], width=25)[:, 0, :]
    feats.append(smooth_fp[:, ::10])
    feats.append(smooth_high[:, ::10])
    feats.append(first_pc_waveforms(y[:, frontal7, :])[:, ::10])

    ch_rms = rms(y, axis=2)
    ch_std = np.std(y, axis=2)
    ch_p2p = np.ptp(y, axis=2)
    ch_absmax = np.max(np.abs(y), axis=2)
    ch_diff_std = np.std(np.diff(y, axis=2), axis=2)
    feats.extend([ch_rms, ch_std, ch_p2p, ch_absmax, ch_diff_std])

    frontal_energy = rms(y[:, frontal7, :], axis=(1, 2))
    posterior_energy = rms(y[:, post5, :], axis=(1, 2))
    full_energy = rms(y, axis=(1, 2))
    fp_energy = rms(y[:, fp_pair, :], axis=(1, 2))
    high_energy = rms(y[:, high5, :], axis=(1, 2))
    feats.append(np.stack([
        frontal_energy / (posterior_energy + 1.0e-8),
        fp_energy / (frontal_energy + 1.0e-8),
        posterior_energy / (full_energy + 1.0e-8),
        frontal_energy / (full_energy + 1.0e-8),
        high_energy / (frontal_energy + 1.0e-8),
    ], axis=1))

    dy = np.diff(y, axis=2)
    feats.append(np.mean(np.abs(dy), axis=2))
    feats.append(np.max(np.abs(dy), axis=2))
    feats.append(rms(np.diff(smooth_high, axis=1), axis=1, keepdims=True))
    feats.append(np.max(np.abs(smooth_high), axis=1, keepdims=True))
    feats.append(rms(smooth_high, axis=1, keepdims=True))

    corr_feats = []
    for sample in y[:, frontal7, :]:
        corr = np.corrcoef(sample)
        corr_feats.append(np.nan_to_num(
            corr[np.triu_indices(len(frontal7), k=1)], nan=0.0, posinf=0.0, neginf=0.0
        ))
    feats.append(np.stack(corr_feats, axis=0))

    if clean_priors is not None:
        front_nll = clean_priors["frontal7"].nll_np(y, q=32)[:, None]
        med = clean_priors["frontal7"].train_nll_stats["median"]
        p95 = clean_priors["frontal7"].train_nll_stats["p95"]
        feats.append(front_nll)
        feats.append(np.maximum(0.0, (front_nll - med) / (p95 - med + 1.0e-8)))

    out = np.concatenate(feats, axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def fit_feature_scaler(x_train):
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = (x_train.std(axis=0, keepdims=True) + 1.0e-6).astype(np.float32)
    return mean, std


def scale_and_augment(train_feat, val_feat):
    mean, std = fit_feature_scaler(train_feat)
    xtr = np.concatenate([np.ones((train_feat.shape[0], 1), dtype=np.float32),
                          (train_feat - mean) / std], axis=1)
    xva = np.concatenate([np.ones((val_feat.shape[0], 1), dtype=np.float32),
                          (val_feat - mean) / std], axis=1)
    return xtr.astype(np.float32), xva.astype(np.float32), (mean, std)


def ridge_fit_predict(x_train, y_train, x_val, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    n = x_train.shape[0]
    folds = np.array_split(np.arange(n), 3)
    best_alpha, best_mse = float(alphas[0]), float("inf")
    for alpha in alphas:
        mses = []
        for val_idx in folds:
            tr_idx = np.setdiff1d(np.arange(n), val_idx)
            xtr, ytr = x_train[tr_idx], y_train[tr_idx]
            xva, yva = x_train[val_idx], y_train[val_idx]
            k = xtr @ xtr.T
            dual = np.linalg.solve(k + alpha * np.eye(k.shape[0], dtype=np.float32), ytr)
            w = xtr.T @ dual
            mses.append(float(np.mean((xva @ w - yva) ** 2)))
        mse = float(np.mean(mses))
        if mse < best_mse:
            best_mse, best_alpha = mse, float(alpha)
    k = x_train @ x_train.T
    dual = np.linalg.solve(k + best_alpha * np.eye(k.shape[0], dtype=np.float32), y_train)
    return (x_val @ (x_train.T @ dual)).astype(np.float32), {
        "ridge_alpha": best_alpha,
        "ridge_cv_target_mse": best_mse,
    }


def fit_scaled_ridge(x_train, target_train, x_val):
    t_mean = target_train.mean(axis=0, keepdims=True).astype(np.float32)
    t_std = (target_train.std(axis=0, keepdims=True) + 1.0e-6).astype(np.float32)
    pred, meta = ridge_fit_predict(x_train, (target_train - t_mean) / t_std, x_val)
    return (pred * t_std + t_mean).astype(np.float32), (t_mean, t_std), meta


def knn_average_beta(query_x, ref_x, ref_beta, k, exclude_self=False):
    d = np.sum((query_x[:, None, :] - ref_x[None, :, :]) ** 2, axis=2)
    if exclude_self and query_x.shape[0] == ref_x.shape[0]:
        np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    dist = np.take_along_axis(d, idx, axis=1)
    w = 1.0 / (np.sqrt(dist) + 1.0e-4)
    w = w / (np.sum(w, axis=1, keepdims=True) + 1.0e-8)
    beta = np.sum(ref_beta[idx] * w[:, :, None], axis=1)
    return beta.astype(np.float32), idx.astype(int), dist.astype(np.float32)

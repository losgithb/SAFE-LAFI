"""View-wise clean-signal priors (PCA subspace + residual variance)."""

import json

import numpy as np

from .config import VIEWS
from .utils import moving_average


class CleanPrior:
    def __init__(self, name, channels, qmax=64, ds=5, smooth_width=25):
        self.name = name
        self.channels = list(channels)
        self.qmax = qmax
        self.ds = ds
        self.smooth_width = smooth_width
        self.mu = None
        self.v = None
        self.var = None
        self.resid_var = 1.0
        self.diag_var = None
        self.train_nll_stats = {}
        self.train_y_nll_stats = {}

    def features_np(self, signals):
        x = signals[:, self.channels, :].astype(np.float32)
        smooth = moving_average(x, width=self.smooth_width)
        deriv = np.diff(smooth, prepend=smooth[:, :, :1], axis=2)
        feat = np.concatenate(
            [x[:, :, :: self.ds], smooth[:, :, :: self.ds], deriv[:, :, :: self.ds]], axis=1
        )
        return feat.reshape(signals.shape[0], -1).astype(np.float32)

    def fit(self, clean_train, noisy_train):
        z = self.features_np(clean_train).astype(np.float64)
        self.mu = z.mean(axis=0, keepdims=True)
        centered = z - self.mu
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        q = min(self.qmax, vt.shape[0])
        self.v = vt[:q].astype(np.float64)
        self.var = ((s[:q] ** 2) / max(1, z.shape[0] - 1)).astype(np.float64) + 1.0e-5
        recon = (centered @ self.v.T) @ self.v
        self.resid_var = float(np.mean((centered - recon) ** 2) + 1.0e-5)
        self.diag_var = np.var(centered, axis=0, keepdims=True).astype(np.float64) + 1.0e-5
        self.train_nll_stats = _robust_stats(self.nll_features(z, q=min(32, q)))
        self.train_y_nll_stats = _robust_stats(self.nll_np(noisy_train, q=min(32, q)))

    @property
    def dim(self):
        return int(self.mu.shape[1])

    def nll_features(self, z, q=32, diag=False):
        zc = z.astype(np.float64) - self.mu
        if diag:
            return np.mean((zc * zc) / self.diag_var, axis=1)
        qq = min(q, self.v.shape[0])
        vq, varq = self.v[:qq], self.var[:qq]
        proj = zc @ vq.T
        recon = proj @ vq
        resid = zc - recon
        nll = np.sum((proj * proj) / varq[None, :], axis=1) + np.sum(
            (resid * resid) / self.resid_var, axis=1
        )
        return nll / float(zc.shape[1])

    def nll_np(self, signals, q=32, diag=False):
        return self.nll_features(self.features_np(signals), q=q, diag=diag)

    def save_npz(self, path):
        metadata = {
            "name": self.name,
            "channels": list(self.channels),
            "qmax": self.qmax,
            "ds": self.ds,
            "smooth_width": self.smooth_width,
            "train_nll_stats": self.train_nll_stats,
            "train_y_nll_stats": self.train_y_nll_stats,
        }
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            mu=self.mu.astype(np.float64),
            v=self.v.astype(np.float64),
            var=self.var.astype(np.float64),
            resid_var=np.asarray(float(self.resid_var), dtype=np.float64),
            diag_var=self.diag_var.astype(np.float64),
        )

    @classmethod
    def load_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            prior = cls(
                metadata["name"], metadata["channels"],
                qmax=int(metadata["qmax"]),
                ds=int(metadata["ds"]),
                smooth_width=int(metadata["smooth_width"]),
            )
            prior.mu = data["mu"].astype(np.float64)
            prior.v = data["v"].astype(np.float64)
            prior.var = data["var"].astype(np.float64)
            prior.resid_var = float(data["resid_var"].item())
            prior.diag_var = data["diag_var"].astype(np.float64)
            prior.train_nll_stats = metadata["train_nll_stats"]
            prior.train_y_nll_stats = metadata["train_y_nll_stats"]
        return prior


def fit_clean_priors(x_train, y_train):
    priors = {
        name: CleanPrior(name, VIEWS[name], qmax=64)
        for name in ["frontal7", "high_frontal5", "fp_pair", "posterior5"]
    }
    for prior in priors.values():
        prior.fit(x_train, y_train)
    return priors


def _robust_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }

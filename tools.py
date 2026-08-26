"""Evaluation metrics used during training."""

from models.utils import full_metrics


def fold_metrics(y, x, x_hat):
    m = full_metrics(y, x, x_hat)
    return {
        "rrmse": m["rrmse"],
        "cc": m["cc"],
        "delta_snr_db": m["delta_snr_db"],
        "over": m["over"],
        "mi_points": m["mi_points"],
    }

"""Safe Floor candidate: extra trees, source routes and batch gate."""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .basis import (
    field_from_topo_source,
    proto_group_indices,
    synthesize_from_basis,
    topo_from_sources,
)
from .config import BASIS_SEED, VIEWS
from .features import knn_average_beta
from .utils import rms


def normalize_sources_for_y(sources, y):
    src = np.asarray(sources, dtype=np.float32).copy()
    fp_mean = np.mean(y[:, VIEWS["fp_pair"], :], axis=1)
    for i in range(src.shape[0]):
        src[i] = src[i] - float(np.mean(src[i]))
        if float(np.dot(src[i], fp_mean[i])) < 0.0:
            src[i] = -src[i]
        src[i] = src[i] / (float(rms(src[i])) + 1.0e-8)
    return src.astype(np.float32)


def source_locked_beta(y, r_reference, sources, basis, weights_proj):
    src = normalize_sources_for_y(sources, y)
    topo = topo_from_sources(r_reference.astype(np.float32), src)
    field = field_from_topo_source(topo, src)
    from .basis import project_onto_basis
    beta = project_onto_basis(field, basis, weights=weights_proj)
    r_hat = synthesize_from_basis(beta, basis)
    return beta.astype(np.float32), r_hat.astype(np.float32)


def fit_floor_estimator(x_train, beta_oracle_train):
    return ExtraTreesRegressor(
        n_estimators=180,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.55,
        random_state=BASIS_SEED,
        n_jobs=-1,
    ).fit(x_train, beta_oracle_train)


def build_floor(y, x_aug, basis, weights_proj, atom_rows, estimator, beta_map,
                source_models, training_sources, x_train_aug):
    """View-blended floor with a source-locked correction (equation 8 anchor)."""
    beta_et = estimator.predict(x_aug).astype(np.float32)
    beta_reference = (0.65 * beta_et + 0.35 * beta_map).astype(np.float32)

    source_ridge = (
        (x_aug @ source_models["weights"]) * source_models["target_std"]
        + source_models["target_mean"]
    ).astype(np.float32)
    source_ridge = normalize_sources_for_y(source_ridge, y)
    source_knn10, _, _ = knn_average_beta(x_aug, x_train_aug, training_sources, k=10)
    source_knn10 = normalize_sources_for_y(source_knn10, y)

    r_ref = synthesize_from_basis(beta_reference, basis)
    beta_lock, _ = source_locked_beta(y, r_ref, source_knn10, basis, weights_proj)
    beta_floor = (0.75 * beta_reference + 0.25 * beta_lock).astype(np.float32)
    beta_floor *= 1.05
    post_idx, _ = proto_group_indices(atom_rows)
    if post_idx.size:
        beta_floor[:, post_idx] *= 0.85
    r_floor = synthesize_from_basis(beta_floor, basis).astype(np.float32)
    return beta_floor, r_floor, source_ridge, source_knn10


def batch_gate(y, r_map, front_prior, train_nll_q80, train_proxy_q80):
    """Batch-level risk gate: fall back to the floor when MAP looks risky."""
    val_map_nll = front_prior.nll_np(y - r_map, q=32, diag=False)
    val_y_nll = front_prior.nll_np(y, q=32, diag=False)
    val_proxy = val_map_nll - val_y_nll
    nll_active = float(np.mean(val_map_nll)) > float(train_nll_q80)
    proxy_active = float(np.mean(val_proxy)) > float(train_proxy_q80)
    active = bool(nll_active or proxy_active)
    return {
        "batch_gate_active": active,
        "map_selected_rate": 0.0 if active else 1.0,
        "floor_selected_rate": 1.0 if active else 0.0,
        "val_map_nll_mean": float(np.mean(val_map_nll)),
        "risk_proxy_mean": float(np.mean(val_proxy)),
    }

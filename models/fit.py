"""Fit a per-fold model state from the semi-simulated benchmark."""

import numpy as np

from .basis import (
    atom_diagnostics,
    build_basis,
    build_topography_model,
    classify_atoms,
    estimate_source_field,
    projection_weights,
    project_onto_basis,
    temporal_atoms_from_sources,
)
from .config import FINAL_EPOCHS, CANDIDATE_EPOCHS, SEED, VIEWS
from .features import make_yonly_features, scale_and_augment
from .floor import fit_floor_estimator
from .hybrid import normalize_y, select_and_refit, train_critic
from .map import (
    apply_ratio_cap,
    fit_map_ridge,
    free_map_optimize,
    reliability_from_train,
    y_proxy_and_mask,
)
from .priors import fit_clean_priors
from .routing import beta_group_meta
from .state import save_state
from .utils import rms, view_rms


def _robust(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p85": float(np.quantile(values, 0.85)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def fit_fold(clean_mat, contaminated_mat, fold_id, out_dir, device=None):
    """Build one fold state and save it under out_dir/fold{fold}."""
    import torch
    from preprocess.data import load_current19, rotating_splits, select_windows
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    x, y, rids = load_current19(clean_mat, contaminated_mat)
    train_list, val_list, _ = rotating_splits()
    (x_tr,) = select_windows((x,), rids, train_list[fold_id])
    (y_tr,) = select_windows((y,), rids, train_list[fold_id])
    r_tr = (y_tr - x_tr).astype(np.float32)

    source_field = estimate_source_field(r_tr)
    atoms = temporal_atoms_from_sources(source_field["sources"], kt=16)
    topo_model = build_topography_model(source_field, ktop=5)
    basis = build_basis(topo_model, atoms)
    priors = fit_clean_priors(x_tr, y_tr)
    event_threshold = float(np.quantile(rms(r_tr[:, VIEWS["frontal7"], :], axis=1), 0.35))
    weights_proj = projection_weights().astype(np.float32)
    atom_rows = atom_diagnostics(basis, kt=16)

    features = make_yonly_features(y_tr, clean_priors=priors)
    x_train_aug, _, (feature_mean, feature_std) = scale_and_augment(features, features)
    beta_oracle_train = project_onto_basis(r_tr, basis, weights=weights_proj)
    map_ridge, ridge_meta = fit_map_ridge(x_train_aug, beta_oracle_train)
    beta0_train = (
        (x_train_aug @ map_ridge["weights"]) * map_ridge["target_std"]
        + map_ridge["target_mean"]
    ).astype(np.float32)

    reliability, reliability_meta = reliability_from_train(y_tr, y_tr)
    ratios = rms(r_tr[:, VIEWS["posterior5"], :], axis=(1, 2)) / (
        rms(r_tr[:, VIEWS["frontal7"], :], axis=(1, 2)) + 1.0e-8
    )
    post_prior = _robust(ratios)
    dynamic_cap_train = np.minimum(
        float(post_prior["p90"]),
        float(post_prior["median"]) + reliability * (float(post_prior["p90"]) - float(post_prior["median"])),
    ).astype(np.float32)
    proxy, mask, proxy_meta = y_proxy_and_mask(y_tr, y_tr)
    alpha_std = (np.std(beta_oracle_train, axis=0, keepdims=True) + 1.0e-3).astype(np.float32)
    r_map_train, beta_map_train, map_meta = free_map_optimize(
        y_tr, basis, beta0_train, alpha_std, mask, proxy, dynamic_cap_train, device
    )
    r_map_train, cap_meta = apply_ratio_cap(r_map_train, dynamic_cap_train)
    map_meta = {**map_meta, **cap_meta, **proxy_meta}

    groups = classify_atoms(basis)
    floor_estimator = fit_floor_estimator(x_train_aug, beta_oracle_train)
    training_sources = source_field["sources"].astype(np.float32)
    source_ridge, _ = fit_map_ridge(x_train_aug, training_sources)

    y_norm_train, y_mean, y_std = normalize_y(y_tr)
    critic, critic_meta = train_critic(x_tr, y_tr, r_tr, device)
    hybrid, selected_config, hybrid_meta = select_and_refit(
        y_tr, x_tr, r_tr, basis, beta_oracle_train, y_norm_train,
        groups, device, critic,
        final_epochs=FINAL_EPOCHS, candidate_epochs=CANDIDATE_EPOCHS,
    )
    beta_mean = beta_oracle_train.mean(axis=0, keepdims=True).astype(np.float32)
    beta_std = (beta_oracle_train.std(axis=0, keepdims=True) + 1.0e-5).astype(np.float32)

    train_map_nll = priors["frontal7"].nll_np(y_tr - r_map_train, q=32, diag=False)
    train_y_nll = priors["frontal7"].nll_np(y_tr, q=32, diag=False)
    train_proxy = train_map_nll - train_y_nll
    map_config = {
        "map_event_threshold_yonly": float(proxy_meta["map_event_threshold_yonly"]),
        "reliability_train_p10": float(reliability_meta["reliability_train_p10"]),
        "reliability_train_p90": float(reliability_meta["reliability_train_p90"]),
        "alpha_std": alpha_std.tolist(),
        "train_map_nll_q80": float(np.quantile(train_map_nll, 0.80)),
        "train_proxy_q80": float(np.quantile(train_proxy, 0.80)),
        "train_map_meta": map_meta,
        "optimizer": "Adam(lr=0.035)",
        "steps": 160,
    }
    train_pf = view_rms(r_tr, "posterior5") / (view_rms(r_tr, "frontal7") + 1.0e-8)
    train_group_meta = beta_group_meta(beta_oracle_train, groups)
    safety_config = {
        "strict": True,
        "train_beta_norm_limit": float(np.quantile(rms(beta_oracle_train, axis=1), 0.99) * 2.0),
        "train_posterior_front_q95": float(np.quantile(train_pf, 0.95)),
        "train_beta_postheavy_source_ratio": float(train_group_meta["beta_postheavy_source_ratio"]),
        "beta_groups": {k: v.astype(int).tolist() for k, v in groups.items()},
        "clean_critic_margin": -0.10,
        "post_limit_ceiling": 0.66,
        "post_limit_floor": 0.58,
        "fp_ratio_minimum": 0.60,
    }

    out_dir = out_dir / f"fold{fold_id}"
    save_state(out_dir, {
        "event_threshold": event_threshold,
        "selected_config": selected_config,
        "basis": basis.astype(np.float32),
        "projection_weights": weights_proj,
        "temporal_atoms": atoms.astype(np.float32),
        "topography_mean": topo_model["mean"],
        "topography_pcs": topo_model["pcs"],
        "atom_rows": atom_rows,
        "priors": priors,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "map_weights": map_ridge["weights"],
        "map_target_mean": map_ridge["target_mean"],
        "map_target_std": map_ridge["target_std"],
        "x_train_aug": x_train_aug,
        "beta_oracle_train": beta_oracle_train,
        "beta_map_train": beta_map_train,
        "r_map_train": r_map_train,
        "dynamic_cap_train": dynamic_cap_train,
        "map_config": map_config,
        "floor_estimator": floor_estimator,
        "training_sources": training_sources,
        "source_weights": source_ridge["weights"],
        "source_target_mean": source_ridge["target_mean"],
        "source_target_std": source_ridge["target_std"],
        "source_alpha": source_ridge["alpha"],
        "critic": critic,
        "hybrid": hybrid,
        "beta_mean": beta_mean,
        "beta_std": beta_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "post_prior": post_prior,
        "safety_config": safety_config,
    })
    return out_dir, {
        "fold": fold_id,
        "selected_config": selected_config,
        "selected_internal_score": hybrid_meta["selected_internal_score"],
        "critic_auc": critic_meta["critic_train_auc"],
        "ridge_alpha": ridge_meta["ridge_alpha"],
        "event_threshold": event_threshold,
    }

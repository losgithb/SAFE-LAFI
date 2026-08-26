"""Run the inference route on contaminated windows."""

import numpy as np
import torch

from .basis import classify_atoms, synthesize_from_basis
from .config import FS, POST
from .features import make_yonly_features
from .floor import batch_gate, build_floor
from .hybrid import apply_beta_calibration, critic_scores
from .map import apply_ratio_cap, free_map_optimize, reliability_raw
from .routing import (
    alpha_change_trigger,
    floor_delta_event_project,
    hybrid_safety,
    route_final_residual,
)
from .utils import moving_average


def apply_emission(r, shrink=0.75, gate_uv=20.0):
    """Scale posterior channels, then zero residuals below the gate threshold."""
    out = r.copy().astype(np.float32)
    out[:, POST, :] *= shrink
    norms = np.linalg.norm(out, axis=(1, 2))
    keep = norms >= gate_uv
    zeroed = int(np.sum(~keep))
    out[~keep] = 0.0
    return out, {"zeroed": zeroed, "norms": norms}


def _y_proxy(y, threshold):
    from .config import VIEWS
    high = np.mean(y[:, VIEWS["high_frontal5"], :], axis=1)
    fp = np.mean(y[:, VIEWS["fp_pair"], :], axis=1)
    smooth = moving_average((0.65 * fp + 0.35 * high)[:, None, :], width=25)[:, 0, :]
    deriv = np.concatenate(
        [np.zeros((smooth.shape[0], 1), dtype=np.float32), np.abs(np.diff(smooth, axis=1))],
        axis=1,
    )
    envelope = np.abs(smooth) + 1.5 * moving_average(deriv[:, None, :], width=15)[:, 0, :]
    mask = envelope > float(threshold)
    central = np.mean(y[:, VIEWS["central_only"], :], axis=1, keepdims=True)
    frontal = y[:, VIEWS["frontal7"], :] - 0.20 * central
    proxy = moving_average(frontal, width=25)
    return proxy.astype(np.float32), mask.astype(np.float32), {
        "map_event_threshold_yonly": float(threshold),
        "map_event_coverage": float(np.mean(mask)),
    }


def apply(state, y, emission=None):
    """Return the residual field and cleaned estimate for [N, 19, 500] input.

    emission: optional tuple (posterior_shrink, gate_uv); when given, the
    emission path is applied after routing (paper section 3.6).
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 3 or y.shape[1:] != (19, 500):
        raise ValueError(f"Expected [N,19,500], got {y.shape}")

    device = state["device"]
    basis = state["basis"]
    priors = state["priors"]
    map_config = state["map_config"]

    features = make_yonly_features(y, clean_priors=priors)
    x_aug = np.concatenate([
        np.ones((y.shape[0], 1), dtype=np.float32),
        (features - state["feature_mean"]) / state["feature_std"],
    ], axis=1).astype(np.float32)
    beta0 = (
        (x_aug @ state["map_weights"]) * state["map_target_std"]
        + state["map_target_mean"]
    ).astype(np.float32)
    reliability = np.clip(
        (reliability_raw(y) - float(map_config["reliability_train_p10"]))
        / (float(map_config["reliability_train_p90"]) - float(map_config["reliability_train_p10"]) + 1.0e-8),
        0.0, 1.0,
    ).astype(np.float32)
    post_prior = state["post_prior"]
    dynamic_cap = np.minimum(
        float(post_prior["p90"]),
        float(post_prior["median"]) + reliability * (float(post_prior["p90"]) - float(post_prior["median"])),
    ).astype(np.float32)

    proxy, mask, proxy_meta = _y_proxy(y, float(map_config["map_event_threshold_yonly"]))
    alpha_std = np.asarray(map_config["alpha_std"], dtype=np.float32)
    r_map, beta_map, map_meta = free_map_optimize(
        y, basis, beta0, alpha_std, mask, proxy, dynamic_cap, device
    )
    r_map, cap_meta = apply_ratio_cap(r_map, dynamic_cap)
    map_meta = {**map_meta, **cap_meta, **proxy_meta}

    beta_floor, r_floor, source_ridge, source_knn10 = build_floor(
        y, x_aug, basis, state["weights_proj"], state["atom_rows"],
        state["floor_estimator"], beta_map, state["source_models"],
        state["training_sources"], state["x_train_aug"],
    )
    baseline_row = batch_gate(
        y, r_map, priors["frontal7"],
        float(map_config["train_map_nll_q80"]),
        float(map_config["train_proxy_q80"]),
    )

    y_norm = ((y - state["y_mean"]) / state["y_std"]).astype(np.float32)
    with torch.no_grad():
        pred_z, _, _ = state["hybrid"](torch.tensor(y_norm, dtype=torch.float32, device=device))
    beta_hybrid = pred_z.cpu().numpy().astype(np.float32) * state["beta_std"] + state["beta_mean"]
    groups = classify_atoms(basis)
    beta_hybrid, calibration_meta = apply_beta_calibration(
        beta_hybrid, state["selected_config"], groups
    )
    r_hybrid = synthesize_from_basis(beta_hybrid, basis).astype(np.float32)
    critic_score = critic_scores(state["critic"], y - r_hybrid, state["y_mean"], state["y_std"], device)
    floor_critic = critic_scores(state["critic"], y - r_floor, state["y_mean"], state["y_std"], device)

    beta_event, r_event, delta_meta = floor_delta_event_project(
        y, basis, r_floor, r_hybrid, state["event_threshold"]
    )
    safety = dict(state["safety_config"])
    safety["event_threshold"] = state["event_threshold"]
    safety_ok, safety_meta = hybrid_safety(
        beta_event, r_event, y, r_floor, groups, critic_score, floor_critic, safety
    )
    trigger, trigger_meta = alpha_change_trigger(
        float(map_config["train_map_meta"]["free_map_alpha_change_mean"]),
        float(map_meta["free_map_alpha_change_mean"]),
    )
    final_r, selected_path, route_meta = route_final_residual(
        baseline_row, trigger, safety_ok, r_map, r_floor, r_event,
        {**calibration_meta, **delta_meta, **safety_meta, **trigger_meta},
    )
    if emission is not None:
        final_r, emission_meta = apply_emission(final_r, *emission)
        route_meta = {**route_meta, **emission_meta}
    return {
        "x_aug": x_aug,
        "beta0": beta0,
        "beta_map": beta_map,
        "r_map": r_map,
        "beta_floor": beta_floor,
        "r_floor": r_floor,
        "source_ridge": source_ridge,
        "source_knn10": source_knn10,
        "beta_hybrid": beta_hybrid,
        "r_hybrid": r_hybrid,
        "beta_event": beta_event,
        "r_event": r_event,
        "r_hat": final_r.astype(np.float32),
        "x_hat": (y - final_r).astype(np.float32),
        "selected_path": selected_path,
        "safety_ok": bool(safety_ok),
        "map_meta": map_meta,
        "baseline_row": baseline_row,
        "route_meta": route_meta,
    }

"""Safety routing: floor-anchored delta, checks and final route selection."""

import numpy as np

from .config import ALPHA, EVENT_GAIN, EVENT_THRESHOLD_SCALE, NON_EVENT_GAIN, VIEWS
from .basis import project_onto_basis, synthesize_from_basis
from .utils import rms64, view_rms


def beta_group_meta(beta, groups):
    post_idx = groups.get("posterior_heavy_atoms", np.asarray([], dtype=np.int64))
    source_idx = groups.get("source_atoms", np.asarray([], dtype=np.int64))
    prop_idx = groups.get("safe_propagation_atoms", np.asarray([], dtype=np.int64))
    post_norm = float(np.mean(rms64(beta[:, post_idx], axis=1))) if post_idx.size else 0.0
    source_norm = float(np.mean(rms64(beta[:, source_idx], axis=1))) if source_idx.size else 0.0
    prop_norm = float(np.mean(rms64(beta[:, prop_idx], axis=1))) if prop_idx.size else 0.0
    return {
        "beta_source_norm_mean": source_norm,
        "beta_safe_prop_norm_mean": prop_norm,
        "beta_postheavy_norm_mean": post_norm,
        "beta_postheavy_source_ratio": float(post_norm / (source_norm + 1.0e-8)),
    }


def non_event_energy(r_hat, y, event_threshold):
    env = rms64(y[:, VIEWS["frontal7"], :], axis=1)
    mask = env <= event_threshold
    values = []
    for i in range(r_hat.shape[0]):
        if np.any(mask[i]):
            values.append(float(rms64(r_hat[i, :, mask[i]])))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def floor_delta_event_project(y, basis, floor_r, hybrid_r, event_threshold):
    """Build the dense floor-relative delta and project it back to beta."""
    delta = hybrid_r.astype(np.float32) - floor_r
    dense = floor_r + ALPHA["other"] * delta
    dense[:, VIEWS["frontal7"], :] = floor_r[:, VIEWS["frontal7"], :] + ALPHA["frontal"] * delta[:, VIEWS["frontal7"], :]
    dense[:, VIEWS["high_frontal5"], :] = floor_r[:, VIEWS["high_frontal5"], :] + ALPHA["highfront"] * delta[:, VIEWS["high_frontal5"], :]
    dense[:, VIEWS["fp_pair"], :] = floor_r[:, VIEWS["fp_pair"], :] + ALPHA["fp"] * delta[:, VIEWS["fp_pair"], :]
    dense[:, VIEWS["posterior5"], :] = floor_r[:, VIEWS["posterior5"], :] + ALPHA["posterior"] * delta[:, VIEWS["posterior5"], :]

    env = rms64(y[:, VIEWS["frontal7"], :], axis=1)
    non_event = env <= event_threshold * EVENT_THRESHOLD_SCALE
    for i in range(dense.shape[0]):
        dense[i, :, non_event[i]] = (
            floor_r[i, :, non_event[i]]
            + NON_EVENT_GAIN * (dense[i, :, non_event[i]] - floor_r[i, :, non_event[i]])
        )
        dense[i, :, ~non_event[i]] = (
            floor_r[i, :, ~non_event[i]]
            + EVENT_GAIN * (dense[i, :, ~non_event[i]] - floor_r[i, :, ~non_event[i]])
        )

    # Posterior-aware weighted projection (equation 9).
    work = dense.copy()
    weights = np.ones(19, dtype=np.float32)
    work[:, VIEWS["posterior5"], :] *= 0.55
    weights[VIEWS["posterior5"]] = 0.40
    weights[VIEWS["frontal7"]] = 1.8
    weights[VIEWS["high_frontal5"]] = 2.5
    weights[VIEWS["fp_pair"]] = 3.5
    beta = project_onto_basis(work, basis, ridge=3.0e-6, weights=weights).astype(np.float32)
    r_hat = synthesize_from_basis(beta, basis).astype(np.float32)
    meta = {
        "alpha_fp": ALPHA["fp"],
        "alpha_highfront": ALPHA["highfront"],
        "alpha_frontal": ALPHA["frontal"],
        "alpha_other": ALPHA["other"],
        "alpha_post": ALPHA["posterior"],
        "event_gain": EVENT_GAIN,
        "non_event_gain": NON_EVENT_GAIN,
        "dense_fp_delta_energy": float(np.mean(view_rms(dense - floor_r, "fp_pair"))),
        "dense_posterior_delta_energy": float(np.mean(view_rms(dense - floor_r, "posterior5"))),
    }
    return beta, r_hat, meta


def hybrid_safety(beta, r_hat, y, floor_r, groups, critic_score, floor_critic, safety):
    """Strict safety checks on the projected hybrid delta."""
    post_front = float(np.mean(view_rms(r_hat, "posterior5") / (view_rms(r_hat, "frontal7") + 1.0e-8)))
    fp_ratio = float(np.mean(view_rms(r_hat, "fp_pair")) / (np.mean(view_rms(floor_r, "fp_pair")) + 1.0e-8))
    ne = float(np.mean(non_event_energy(r_hat, y, safety["event_threshold"])))
    beta_norm = float(np.mean(rms64(beta, axis=1)))
    group_meta = beta_group_meta(beta, groups)
    beta_post_source_ratio = float(group_meta["beta_postheavy_source_ratio"])
    critic_mean = float(np.nanmean(critic_score))
    floor_critic_mean = float(np.nanmean(floor_critic))
    clean_ok = critic_mean >= floor_critic_mean - 0.10
    post_limit = min(0.66, max(0.58, float(safety["train_posterior_front_q95"]) * 1.05))
    beta_ratio_limit = max(0.55, float(safety["train_beta_postheavy_source_ratio"]) * 1.05)
    ok = (
        post_front <= post_limit
        and fp_ratio >= 0.60
        and beta_norm <= float(safety["train_beta_norm_limit"])
        and beta_post_source_ratio <= beta_ratio_limit
        and clean_ok
    )
    return bool(ok), {
        "hybrid_safety_pass": bool(ok),
        "post_front_limit": post_limit,
        "pred_post_front_ratio": post_front,
        "pred_fp_to_floor_fp_ratio": fp_ratio,
        "non_event_energy_mean": ne,
        "beta_norm_mean": beta_norm,
        **group_meta,
        "beta_postheavy_source_ratio_limit": beta_ratio_limit,
        "clean_critic_score_mean": critic_mean,
        "floor_clean_critic_score_mean": floor_critic_mean,
    }


def alpha_change_trigger(train_alpha_change, val_alpha_change):
    ratio = val_alpha_change / (train_alpha_change + 1.0e-8)
    return bool(ratio > 1.01), {
        "train_alpha_change_mean": train_alpha_change,
        "val_alpha_change_mean": val_alpha_change,
        "alpha_change_ratio": ratio,
        "alpha_change_ratio_threshold": 1.01,
    }


def route_final_residual(baseline_row, trigger, safety_ok, r_map, r_floor, r_event, meta=None):
    """Equation 10: MAP when safe, projected hybrid when it passes checks,
    floor otherwise, with the alpha-change override in between."""
    map_safe = float(baseline_row.get("map_selected_rate", 0.0)) > 0.5
    if map_safe and trigger and safety_ok:
        selected, final_r, rates = "HYBRID_RBE_ALPHA_CHANGE_OVERRIDE", r_event, {
            "map_selected_rate": 0.0, "floor_selected_rate": 0.0, "hybrid_selected_rate": 1.0}
    elif map_safe:
        selected, final_r, rates = "MAP_CONTTOPO", r_map, {
            "map_selected_rate": 1.0, "floor_selected_rate": 0.0, "hybrid_selected_rate": 0.0}
    elif safety_ok:
        selected, final_r, rates = "HYBRID_RBE", r_event, {
            "map_selected_rate": 0.0, "floor_selected_rate": 0.0, "hybrid_selected_rate": 1.0}
    else:
        selected, final_r, rates = "VIEWBLEND_FLOOR_FIXED", r_floor, {
            "map_selected_rate": 0.0, "floor_selected_rate": 1.0, "hybrid_selected_rate": 0.0}
    result = {"selected_path": selected, "hybrid_safety_pass": bool(safety_ok), **rates}
    if meta:
        result.update(meta)
    return final_r.astype(np.float32), selected, result

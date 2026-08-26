"""Save and load per-fold model states."""

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .priors import CleanPrior


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_state(state_dir, components):
    """Write a fitted state using the layout expected by load_state."""
    state_dir = Path(state_dir)
    for sub in ["basis", "clean_priors", "features", "map", "floor", "source",
                "critic", "hybrid", "thresholds"]:
        (state_dir / "components" / sub).mkdir(parents=True, exist_ok=True)
    c = components
    _write_json(state_dir / "state.json", {
        "method": {"name": "SAFE-LAFI", "version": "1.0"},
        "event_threshold": float(c["event_threshold"]),
        "hybrid_model": {"selected_config": c["selected_config"]},
    })
    np.save(state_dir / "components/basis/basis.npy", c["basis"], allow_pickle=False)
    np.save(state_dir / "components/basis/projection_weights.npy", c["projection_weights"], allow_pickle=False)
    np.save(state_dir / "components/basis/temporal_atoms.npy", c["temporal_atoms"], allow_pickle=False)
    np.save(state_dir / "components/basis/topography_mean.npy", c["topography_mean"], allow_pickle=False)
    np.save(state_dir / "components/basis/topography_pcs.npy", c["topography_pcs"], allow_pickle=False)
    _write_json(state_dir / "components/basis/atom_diagnostics.json", c["atom_rows"])
    for name in ["frontal7", "high_frontal5", "fp_pair", "posterior5"]:
        c["priors"][name].save_npz(state_dir / f"components/clean_priors/{name}.npz")
    np.save(state_dir / "components/features/mean.npy", c["feature_mean"], allow_pickle=False)
    np.save(state_dir / "components/features/std.npy", c["feature_std"], allow_pickle=False)
    np.save(state_dir / "components/map/ridge_weights.npy", c["map_weights"], allow_pickle=False)
    np.save(state_dir / "components/map/target_mean.npy", c["map_target_mean"], allow_pickle=False)
    np.save(state_dir / "components/map/target_std.npy", c["map_target_std"], allow_pickle=False)
    np.save(state_dir / "components/map/x_train_aug.npy", c["x_train_aug"], allow_pickle=False)
    np.save(state_dir / "components/map/beta_oracle_train.npy", c["beta_oracle_train"], allow_pickle=False)
    np.save(state_dir / "components/map/beta_map_train.npy", c["beta_map_train"], allow_pickle=False)
    np.save(state_dir / "components/map/r_map_train.npy", c["r_map_train"], allow_pickle=False)
    np.save(state_dir / "components/map/dynamic_cap_train.npy", c["dynamic_cap_train"], allow_pickle=False)
    _write_json(state_dir / "components/map/map_config.json", c["map_config"])
    joblib.dump(c["floor_estimator"], state_dir / "components/floor/extratrees.joblib", compress=3)
    np.save(state_dir / "components/source/training_sources.npy", c["training_sources"], allow_pickle=False)
    np.savez_compressed(
        state_dir / "components/source/route_models.npz",
        weights=c["source_weights"],
        target_mean=c["source_target_mean"],
        target_std=c["source_target_std"],
        ridge_alpha=np.asarray(c["source_alpha"], dtype=np.float64),
    )
    torch.save({k: v.detach().cpu() for k, v in c["critic"].state_dict().items()},
               state_dir / "components/critic/weights.pt")
    torch.save({k: v.detach().cpu() for k, v in c["hybrid"].state_dict().items()},
               state_dir / "components/hybrid/weights.pt")
    np.save(state_dir / "components/hybrid/beta_mean.npy", c["beta_mean"], allow_pickle=False)
    np.save(state_dir / "components/hybrid/beta_std.npy", c["beta_std"], allow_pickle=False)
    np.save(state_dir / "components/hybrid/y_mean.npy", c["y_mean"], allow_pickle=False)
    np.save(state_dir / "components/hybrid/y_std.npy", c["y_std"], allow_pickle=False)
    _write_json(state_dir / "components/thresholds/post_prior.json", c["post_prior"])
    _write_json(state_dir / "components/thresholds/safety_config.json", c["safety_config"])


def load_state(state_dir, device=None):
    """Load a fitted state."""
    state_dir = Path(state_dir)
    state_json = _read_json(state_dir / "state.json")
    components = state_json.get("components", {})
    basis = np.load(state_dir / "components/basis/basis.npy", allow_pickle=False)
    weights_proj = np.load(state_dir / "components/basis/projection_weights.npy", allow_pickle=False)
    atom_rows = _read_json(state_dir / "components/basis/atom_diagnostics.json")
    priors = {
        name: CleanPrior.load_npz(state_dir / f"components/clean_priors/{name}.npz")
        for name in ["frontal7", "high_frontal5", "fp_pair", "posterior5"]
    }
    feature_mean = np.load(state_dir / "components/features/mean.npy", allow_pickle=False)
    feature_std = np.load(state_dir / "components/features/std.npy", allow_pickle=False)
    map_config = _read_json(state_dir / "components/map/map_config.json")
    map_weights = np.load(state_dir / "components/map/ridge_weights.npy", allow_pickle=False)
    map_target_mean = np.load(state_dir / "components/map/target_mean.npy", allow_pickle=False)
    map_target_std = np.load(state_dir / "components/map/target_std.npy", allow_pickle=False)
    x_train_aug = np.load(state_dir / "components/map/x_train_aug.npy", allow_pickle=False)
    beta_oracle_train = np.load(state_dir / "components/map/beta_oracle_train.npy", allow_pickle=False)
    floor_estimator = joblib.load(state_dir / "components/floor/extratrees.joblib")
    training_sources = np.load(state_dir / "components/source/training_sources.npy", allow_pickle=False)
    with np.load(state_dir / "components/source/route_models.npz", allow_pickle=False) as source_models:
        source_weights = source_models["weights"]
        source_target_mean = source_models["target_mean"]
        source_target_std = source_models["target_std"]
    post_prior = _read_json(state_dir / "components/thresholds/post_prior.json")
    safety_config = _read_json(state_dir / "components/thresholds/safety_config.json")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from .hybrid import CleanCritic, HybridRBE
    critic = CleanCritic().to(device)
    critic.load_state_dict(torch.load(state_dir / "components/critic/weights.pt",
                                      map_location=device, weights_only=True))
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    hybrid = HybridRBE(basis.shape[0]).to(device)
    hybrid.load_state_dict(torch.load(state_dir / "components/hybrid/weights.pt",
                                      map_location=device, weights_only=True))
    hybrid.eval()
    beta_mean = np.load(state_dir / "components/hybrid/beta_mean.npy", allow_pickle=False)
    beta_std = np.load(state_dir / "components/hybrid/beta_std.npy", allow_pickle=False)
    y_mean = np.load(state_dir / "components/hybrid/y_mean.npy", allow_pickle=False)
    y_std = np.load(state_dir / "components/hybrid/y_std.npy", allow_pickle=False)

    if "event_threshold" in state_json:
        event_threshold = float(state_json["event_threshold"])
    else:
        event_threshold = float(components["thresholds"]["event_threshold"])
    selected_config = state_json.get("hybrid_model", {}).get("selected_config")
    if selected_config is None:
        selected_config = components["hybrid_model"]["selected_config"]
    return {
        "state_dir": state_dir,
        "basis": basis,
        "weights_proj": weights_proj,
        "atom_rows": atom_rows,
        "priors": priors,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "map_config": map_config,
        "map_weights": map_weights,
        "map_target_mean": map_target_mean,
        "map_target_std": map_target_std,
        "x_train_aug": x_train_aug,
        "beta_oracle_train": beta_oracle_train,
        "floor_estimator": floor_estimator,
        "training_sources": training_sources,
        "source_models": {
            "weights": source_weights,
            "target_mean": source_target_mean,
            "target_std": source_target_std,
        },
        "post_prior": post_prior,
        "safety_config": safety_config,
        "event_threshold": event_threshold,
        "critic": critic,
        "hybrid": hybrid,
        "beta_mean": beta_mean,
        "beta_std": beta_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "selected_config": selected_config,
        "device": device,
    }

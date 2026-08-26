"""Hybrid residual-basis encoder and clean critic."""

import numpy as np
import torch
from torch import nn

from .config import N_CHANNELS, SEED, T, VIEWS, VARIANTS
from .utils import roc_auc, rms64, view_rms


class HybridRBE(nn.Module):
    def __init__(self, beta_dim):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv1d(19, 32, 7, padding=3), nn.GELU(),
            nn.Conv1d(32, 32, 7, padding=3), nn.GELU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(19, 32, 15, padding=7), nn.GELU(),
            nn.Conv1d(32, 32, 7, padding=6, dilation=2), nn.GELU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(19, 32, 31, padding=15), nn.GELU(),
            nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(96, 96, 1), nn.GELU(),
            nn.Conv1d(96, 64, 5, padding=2), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(18)
        self.global_head = nn.Sequential(
            nn.Linear(64 * 18 + 19 * 6, 192), nn.GELU(),
            nn.Dropout(0.05), nn.Linear(192, 128), nn.GELU(),
        )
        self.beta_head = nn.Linear(128, beta_dim)
        self.safety_head = nn.Linear(128, 4)
        self.source_head = nn.Linear(128, 64)

    def forward(self, y):
        z = torch.cat([self.b1(y), self.b2(y), self.b3(y)], dim=1)
        z = self.mix(z)
        pooled = self.pool(z).flatten(1)
        ch_mean = y.mean(dim=2)
        ch_std = y.std(dim=2)
        ch_abs = y.abs().mean(dim=2)
        ch_der = torch.diff(y, dim=2).pow(2).mean(dim=2).sqrt()
        ch_max = y.amax(dim=2)
        ch_min = y.amin(dim=2)
        stats = torch.cat([ch_mean, ch_std, ch_abs, ch_der, ch_max, ch_min], dim=1)
        h = self.global_head(torch.cat([pooled, stats], dim=1))
        return self.beta_head(h), self.safety_head(h), self.source_head(h)


class CleanCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(19, 24, 11, padding=5), nn.GELU(),
            nn.Conv1d(24, 32, 7, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(12),
            nn.Flatten(),
            nn.Linear(32 * 12, 64), nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def normalize_y(y_train, y_val=None):
    mean = y_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = y_train.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1.0e-6
    if y_val is None:
        return ((y_train - mean) / std).astype(np.float32), mean, std
    return ((y_train - mean) / std).astype(np.float32), \
        ((y_val - mean) / std).astype(np.float32), mean, std


def train_critic(x_train, y_train, r_train, device):
    mean = y_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = y_train.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1.0e-6
    x_norm = ((x_train - mean) / std).astype(np.float32)
    y_norm = ((y_train - mean) / std).astype(np.float32)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(r_train.shape[0])
    neg = np.concatenate([
        y_norm,
        ((x_train + 0.45 * r_train - mean) / std).astype(np.float32),
        ((x_train - 0.35 * r_train[perm] - mean) / std).astype(np.float32),
    ], axis=0)
    pos = x_norm
    data = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([
        np.ones(pos.shape[0], dtype=np.float32),
        np.zeros(neg.shape[0], dtype=np.float32),
    ])
    idx = np.arange(data.shape[0])
    rng.shuffle(idx)
    model = CleanCritic().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    xt = torch.tensor(data[idx], dtype=torch.float32, device=device)
    yt = torch.tensor(labels[idx], dtype=torch.float32, device=device)
    for _ in range(25):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        prob = torch.sigmoid(model(xt)).cpu().numpy()
    auc = roc_auc(labels[idx], prob)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    meta = {
        "critic_train_auc": auc,
        "critic_clean_score": float(np.mean(prob[labels[idx] == 1])),
        "critic_corrupt_score": float(np.mean(prob[labels[idx] == 0])),
    }
    return model, meta


def event_mask_np(y):
    env = np.mean(np.abs(y[:, VIEWS["frontal7"], :]), axis=1)
    threshold = float(np.quantile(env, 0.65))
    return (env > threshold).astype(np.float32), threshold


def _view_mse(pred, target, view):
    idx = torch.tensor(VIEWS[view], dtype=torch.long, device=pred.device)
    return nn.functional.mse_loss(pred.index_select(1, idx), target.index_select(1, idx))


def _synth_torch(beta, basis_flat):
    b = beta @ basis_flat
    return b.reshape(beta.shape[0], N_CHANNELS, -1)


def _hybrid_loss(pred_z, beta, safety_pred, r_hat, y_raw, x_raw, r_true,
                 beta_target_z, event_mask, critic, cfg, post_ratio_target, groups_t):
    loss_full = nn.functional.mse_loss(r_hat, r_true)
    loss_fp = _view_mse(r_hat, r_true, "fp_pair")
    loss_hf = _view_mse(r_hat, r_true, "high_frontal5")
    loss_fr = _view_mse(r_hat, r_true, "frontal7")
    x_hat = y_raw - r_hat
    clean_loss = nn.functional.mse_loss(x_hat, x_raw)
    beta_loss = nn.functional.smooth_l1_loss(pred_z, beta_target_z)
    front = torch.sqrt((r_hat[:, VIEWS["frontal7"], :] ** 2).mean(dim=(1, 2)) + 1.0e-8)
    post = torch.sqrt((r_hat[:, VIEWS["posterior5"], :] ** 2).mean(dim=(1, 2)) + 1.0e-8)
    post_loss = torch.relu(post / (front + 1.0e-6) - post_ratio_target).pow(2).mean()
    non_event = (1.0 - event_mask).unsqueeze(1)
    non_event_loss = ((r_hat * non_event) ** 2).mean()
    source_loss = torch.tensor(0.0, device=r_hat.device)
    event_loss = torch.tensor(0.0, device=r_hat.device)
    if cfg.get("source", 0.0) > 0:
        src_pred = r_hat[:, VIEWS["fp_pair"], :].mean(dim=1)
        src_true = r_true[:, VIEWS["fp_pair"], :].mean(dim=1)
        a, b = src_pred - src_pred.mean(dim=1, keepdim=True), src_true - src_true.mean(dim=1, keepdim=True)
        denom = torch.sqrt((a * a).sum(dim=1) * (b * b).sum(dim=1) + 1.0e-8)
        source_loss = 1.0 - ((a * b).sum(dim=1) / denom).mean()
        event_loss = nn.functional.mse_loss(src_pred.abs(), src_true.abs())
    critic_loss = torch.tensor(0.0, device=r_hat.device)
    if cfg.get("critic", 0.0) > 0 and critic is not None:
        critic_loss = nn.functional.binary_cross_entropy_with_logits(
            critic(x_hat), torch.ones(x_hat.shape[0], device=x_hat.device)
        )
    beta_postheavy_loss = torch.tensor(0.0, device=r_hat.device)
    beta_ratio_loss = torch.tensor(0.0, device=r_hat.device)
    beta_source_floor_loss = torch.tensor(0.0, device=r_hat.device)
    if groups_t is not None:
        post_idx = groups_t.get("posterior_heavy_atoms")
        source_idx = groups_t.get("source_atoms")
        if post_idx is not None and post_idx.numel() > 0:
            post_beta = torch.sqrt((beta.index_select(1, post_idx) ** 2).mean(dim=1) + 1.0e-8)
            beta_postheavy_loss = (post_beta ** 2).mean()
        else:
            post_beta = torch.zeros(beta.shape[0], device=r_hat.device)
        if source_idx is not None and source_idx.numel() > 0:
            source_beta = torch.sqrt((beta.index_select(1, source_idx) ** 2).mean(dim=1) + 1.0e-8)
        else:
            source_beta = torch.ones(beta.shape[0], device=r_hat.device)
        beta_ratio_loss = torch.relu(
            post_beta / (source_beta + 1.0e-6) - float(cfg.get("beta_ratio_tau", 0.85))
        ).pow(2).mean()
        beta_source_floor_loss = torch.relu(
            float(cfg.get("source_floor_tau", 0.0)) - source_beta
        ).pow(2).mean()
    safety_target = torch.stack([
        (post / (front + 1.0e-6)).detach(),
        torch.sqrt((r_true[:, VIEWS["fp_pair"], :] - r_hat[:, VIEWS["fp_pair"], :]).pow(2).mean(dim=(1, 2)) + 1.0e-8).detach(),
        torch.sqrt(((r_hat * non_event) ** 2).mean(dim=(1, 2)) + 1.0e-8).detach(),
        torch.sqrt(((x_hat - x_raw) ** 2).mean(dim=(1, 2)) + 1.0e-8).detach(),
    ], dim=1)
    safety_loss = nn.functional.smooth_l1_loss(safety_pred, safety_target)
    total = (
        loss_full
        + 2.0 * loss_fp
        + 1.8 * loss_hf
        + 1.5 * loss_fr
        + cfg.get("clean", 0.5) * clean_loss
        + cfg.get("beta", 0.1) * beta_loss
        + cfg.get("source", 0.0) * (source_loss + 0.25 * event_loss)
        + cfg.get("post", 0.5) * post_loss
        + cfg.get("non_event", 0.5) * non_event_loss
        + cfg.get("critic", 0.0) * critic_loss
        + cfg.get("beta_postheavy", 0.0) * beta_postheavy_loss
        + cfg.get("beta_ratio", 0.0) * beta_ratio_loss
        + cfg.get("source_floor", 0.0) * beta_source_floor_loss
        + 0.05 * safety_loss
    )
    return total


def _internal_score(y, x, r_true, r_hat):
    full_mse = float(np.mean((r_hat - r_true) ** 2))
    fp_mse = float(np.mean((r_hat[:, VIEWS["fp_pair"], :] - r_true[:, VIEWS["fp_pair"], :]) ** 2))
    frontal_mse = float(np.mean((r_hat[:, VIEWS["frontal7"], :] - r_true[:, VIEWS["frontal7"], :]) ** 2))
    post_ratio = float(np.mean(view_rms(r_hat, "posterior5") / (view_rms(r_hat, "frontal7") + 1.0e-8)))
    fp_miss = float(np.mean(rms64(r_hat[:, VIEWS["fp_pair"], :] - r_true[:, VIEWS["fp_pair"], :], axis=(1, 2)) / (view_rms(r_true, "fp_pair") + 1.0e-8)))
    clean_projection = float(np.mean(np.abs(np.mean((r_hat * x), axis=(1, 2)))))
    return -(
        full_mse + 1.8 * fp_mse + 0.8 * frontal_mse
        + 2.0 * max(0.0, post_ratio - 0.60)
        + 1.5 * max(0.0, fp_miss - 0.56)
        + 0.2 * clean_projection
    )


def train_hybrid(y_train, x_train, r_train, basis, beta_oracle, cfg, train_idx,
                 calib_idx, y_norm_train, groups, device, epochs, critic=None):
    torch.manual_seed(SEED + int(cfg.get("seed_offset", 0)))
    beta_dim = beta_oracle.shape[1]
    model = HybridRBE(beta_dim).to(device)
    basis_flat = torch.tensor(basis.reshape(beta_dim, -1), dtype=torch.float32, device=device)
    y_raw = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_raw = torch.tensor(x_train, dtype=torch.float32, device=device)
    r_true = torch.tensor(r_train, dtype=torch.float32, device=device)
    y_in = torch.tensor(y_norm_train, dtype=torch.float32, device=device)
    beta_mean = torch.tensor(beta_oracle[train_idx].mean(axis=0, keepdims=True).astype(np.float32),
                             dtype=torch.float32, device=device)
    beta_std = torch.tensor(beta_oracle[train_idx].std(axis=0, keepdims=True).astype(np.float32) + 1.0e-5,
                            dtype=torch.float32, device=device)
    beta_target = torch.tensor(beta_oracle, dtype=torch.float32, device=device)
    beta_target_z = (beta_target - beta_mean) / beta_std
    event_mask_np_, _ = event_mask_np(y_train)
    event_mask = torch.tensor(event_mask_np_, dtype=torch.float32, device=device)
    post_ratio_train = view_rms(r_train[train_idx], "posterior5") / (view_rms(r_train[train_idx], "frontal7") + 1.0e-8)
    post_ratio_target = float(np.quantile(post_ratio_train, 0.90))

    groups_t = None
    if groups is not None:
        groups_t = {
            "posterior_heavy_atoms": torch.tensor(groups["posterior_heavy_atoms"], dtype=torch.long, device=device),
            "source_atoms": torch.tensor(groups["source_atoms"], dtype=torch.long, device=device),
        }
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1.2e-3)),
                            weight_decay=float(cfg.get("weight_decay", 1.0e-4)))
    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    calib_t = torch.tensor(calib_idx, dtype=torch.long, device=device) if calib_idx is not None and calib_idx.size else None
    batch_size = min(64, max(8, train_idx.size))
    rng = np.random.default_rng(SEED)
    best_state, best_score = None, -1.0e18
    for epoch in range(epochs):
        order = train_idx.copy()
        rng.shuffle(order)
        model.train()
        for start in range(0, order.size, batch_size):
            bidx = torch.tensor(order[start:start + batch_size], dtype=torch.long, device=device)
            opt.zero_grad()
            pred_z, safety_pred, _ = model(y_in.index_select(0, bidx))
            beta = pred_z * beta_std + beta_mean
            r_hat = _synth_torch(beta, basis_flat).reshape(beta.shape[0], N_CHANNELS, T)
            loss = _hybrid_loss(
                pred_z, beta, safety_pred, r_hat,
                y_raw.index_select(0, bidx), x_raw.index_select(0, bidx),
                r_true.index_select(0, bidx), beta_target_z.index_select(0, bidx),
                event_mask.index_select(0, bidx), critic, cfg, post_ratio_target, groups_t,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        if epoch == epochs - 1 or epoch % 5 == 0:
            model.eval()
            eval_t = calib_t if calib_t is not None else train_t
            eval_np = calib_idx if calib_idx is not None and calib_idx.size else train_idx
            with torch.no_grad():
                pred_z, _, _ = model(y_in.index_select(0, eval_t))
                beta = pred_z * beta_std + beta_mean
                r_hat = _synth_torch(beta, basis_flat).cpu().numpy().reshape(eval_np.shape[0], N_CHANNELS, T)
            score = _internal_score(y_train[eval_np], x_train[eval_np], r_train[eval_np], r_hat)
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    meta = {
        "selected_internal_score": float(best_score),
        "post_ratio_target": post_ratio_target,
        **{f"lambda_{k}": v for k, v in cfg.items() if isinstance(v, (float, int))},
    }
    return model, meta


def select_and_refit(y_train, x_train, r_train, basis, beta_oracle, y_norm_train,
                     groups, device, critic, final_epochs=54, candidate_epochs=34):
    """Screen the eligible variants, then refit the winner on all windows."""
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(y_train.shape[0])
    cut = max(8, int(0.80 * y_train.shape[0]))
    train_idx, calib_idx = permutation[:cut], permutation[cut:]
    eligible = [dict(c) for c in VARIANTS if c["family"] in ("m1", "m2", "m3", "m5")]
    best_cfg, best_score = None, -1.0e18
    for cfg in eligible:
        use_critic = critic if float(cfg.get("critic", 0.0)) > 0 else None
        model, meta = train_hybrid(
            y_train, x_train, r_train, basis, beta_oracle, cfg,
            train_idx, calib_idx, y_norm_train, groups, device, candidate_epochs,
            critic=use_critic,
        )
        score = float(meta["selected_internal_score"])
        if score > best_score:
            best_score, best_cfg = score, dict(cfg)
        del model
    best_cfg["name"] = "BEST_INTERNAL"
    full_idx = np.arange(y_train.shape[0], dtype=np.int64)
    use_critic = critic if float(best_cfg.get("critic", 0.0)) > 0 else None
    model, meta = train_hybrid(
        y_train, x_train, r_train, basis, beta_oracle, best_cfg,
        full_idx, None, y_norm_train, groups, device, final_epochs,
        critic=use_critic,
    )
    return model, best_cfg, meta


def apply_beta_calibration(beta, cfg, groups):
    out = beta.copy().astype(np.float32)
    meta = {"post_gate_mean": float("nan"), "post_shrink": float(cfg.get("post_shrink", 1.0))}
    if groups is None:
        return out, meta
    post_idx = groups.get("posterior_heavy_atoms", np.asarray([], dtype=np.int64))
    source_idx = groups.get("source_atoms", np.asarray([], dtype=np.int64))
    if post_idx.size:
        out[:, post_idx] *= float(cfg.get("post_shrink", 1.0))
    if post_idx.size and source_idx.size and cfg.get("post_ratio_cap", 0.0) > 0:
        post_norm = rms64(out[:, post_idx], axis=1)
        source_norm = rms64(out[:, source_idx], axis=1)
        scale = np.minimum(1.0, (float(cfg.get("post_ratio_cap", 0.75)) * source_norm + 1.0e-8) / (post_norm + 1.0e-8)).astype(np.float32)
        out[:, post_idx] *= scale[:, None]
        meta["post_gate_mean"] = float(np.mean(scale))
    if source_idx.size and cfg.get("source_floor_boost", 0.0) > 0:
        source_norm = rms64(out[:, source_idx], axis=1)
        floor = float(cfg.get("source_floor_level", np.quantile(source_norm, 0.25) if source_norm.size else 0.0))
        boost = np.minimum(1.20, np.maximum(1.0, (floor + 1.0e-8) / (source_norm + 1.0e-8))).astype(np.float32)
        out[:, source_idx] *= boost[:, None]
        meta["source_boost_mean"] = float(np.mean(boost))
    return out, meta


def critic_scores(critic, arr, y_mean, y_std, device):
    if critic is None:
        return np.full(arr.shape[0], np.nan, dtype=np.float32)
    norm = ((arr - y_mean) / y_std).astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(critic(torch.tensor(norm, dtype=torch.float32, device=device))).cpu().numpy().astype(np.float32)

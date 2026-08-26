"""MAP candidate: ridge initialization and free conttopo optimization."""

import numpy as np
import torch

from .config import N_CHANNELS, VIEWS
from .features import fit_scaled_ridge
from .utils import moving_average, pearson_flat, rms

MAP_STEPS = 160
MAP_LR = 0.035


def reliability_raw(y):
    fp = np.mean(y[:, VIEWS["fp_pair"], :], axis=1)
    high = np.mean(y[:, VIEWS["high_frontal5"], :], axis=1)
    smooth = moving_average(high[:, None, :], width=25)[:, 0, :]
    strength = rms(smooth, axis=1)
    fpdom = rms(fp, axis=1) / (rms(high, axis=1) + 1.0e-8)
    agree = np.asarray(
        [max(0.0, pearson_flat(fp[i], high[i])) for i in range(y.shape[0])],
        dtype=np.float32,
    )
    return 0.55 * strength + 0.25 * fpdom + 0.20 * agree


def reliability_from_train(y_train, y_val):
    train_raw = reliability_raw(y_train)
    val_raw = reliability_raw(y_val)
    p10 = float(np.quantile(train_raw, 0.10))
    p90 = float(np.quantile(train_raw, 0.90))
    scaled = np.clip((val_raw - p10) / (p90 - p10 + 1.0e-8), 0.0, 1.0).astype(np.float32)
    return scaled, {"reliability_train_p10": p10, "reliability_train_p90": p90}


def y_proxy_and_mask(y_train, y_val):
    def envelope(y):
        high = np.mean(y[:, VIEWS["high_frontal5"], :], axis=1)
        fp = np.mean(y[:, VIEWS["fp_pair"], :], axis=1)
        smooth = moving_average((0.65 * fp + 0.35 * high)[:, None, :], width=25)[:, 0, :]
        deriv = np.concatenate(
            [np.zeros((smooth.shape[0], 1), dtype=np.float32), np.abs(np.diff(smooth, axis=1))],
            axis=1,
        )
        return np.abs(smooth) + 1.5 * moving_average(deriv[:, None, :], width=15)[:, 0, :]

    env_train = envelope(y_train)
    threshold = float(np.quantile(env_train, 0.62))
    env_val = envelope(y_val)
    mask = env_val > threshold
    central = np.mean(y_val[:, VIEWS["central_only"], :], axis=1, keepdims=True)
    frontal = y_val[:, VIEWS["frontal7"], :] - 0.20 * central
    proxy = moving_average(frontal, width=25)
    return proxy.astype(np.float32), mask.astype(np.float32), {
        "map_event_threshold_yonly": threshold,
        "map_event_coverage": float(np.mean(mask)),
    }


def free_map_optimize(y, basis, beta0, alpha_std, mask, proxy, dynamic_cap, device):
    """Run the 160-step Adam MAP solver and return residual + coefficients."""
    basis_t = torch.tensor(basis, dtype=torch.float32, device=device)
    alpha0_t = torch.tensor(beta0, dtype=torch.float32, device=device)
    alpha = torch.nn.Parameter(alpha0_t.clone())
    proxy_t = torch.tensor(proxy, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    cap_t = torch.tensor(dynamic_cap, dtype=torch.float32, device=device)
    alpha_std_t = torch.tensor(alpha_std, dtype=torch.float32, device=device)
    front_idx = torch.tensor(VIEWS["frontal7"], dtype=torch.long, device=device)
    post_idx = torch.tensor(VIEWS["posterior5"], dtype=torch.long, device=device)
    front_w = torch.tensor([4.0, 4.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                           dtype=torch.float32, device=device)[None, :, None]
    opt = torch.optim.Adam([alpha], lr=MAP_LR)
    losses = {}
    for _ in range(MAP_STEPS):
        r_hat = torch.einsum("bk,kct->bct", alpha, basis_t)
        r_front = _moving_average_torch(r_hat.index_select(1, front_idx), width=25)
        l_front = torch.mean(((r_front - proxy_t) * mask_t[:, None, :] * front_w) ** 2) / (
            torch.mean(mask_t) + 0.05
        )
        l_prior = torch.mean(((alpha - alpha0_t) / alpha_std_t) ** 2)
        l_coef = torch.mean((alpha / alpha_std_t) ** 2)
        non_mask = 1.0 - mask_t
        l_nonevent = torch.sum((r_hat * non_mask[:, None, :]) ** 2) / (
            torch.sum(non_mask) * N_CHANNELS + 1.0e-8
        )
        front_r = torch.sqrt(torch.mean(r_hat.index_select(1, front_idx) ** 2, dim=(1, 2)) + 1.0e-8)
        post_r = torch.sqrt(torch.mean(r_hat.index_select(1, post_idx) ** 2, dim=(1, 2)) + 1.0e-8)
        l_post = torch.mean(torch.relu(post_r - cap_t * front_r) ** 2)
        second = r_hat[:, :, 2:] - 2.0 * r_hat[:, :, 1:-1] + r_hat[:, :, :-2]
        l_smooth = torch.mean(second * second)
        loss = (
            0.18 * l_front + 0.85 * l_prior + 0.015 * l_coef
            + 0.18 * l_nonevent + 3.0 * l_post + 0.02 * l_smooth
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([alpha], 20.0)
        opt.step()
        with torch.no_grad():
            alpha.copy_(torch.clamp(alpha, alpha0_t - 3.5 * alpha_std_t, alpha0_t + 3.5 * alpha_std_t))
        losses = {
            "free_map_loss": float(loss.detach().cpu()),
            "free_map_l_front": float(l_front.detach().cpu()),
            "free_map_l_post": float(l_post.detach().cpu()),
            "free_map_l_prior": float(l_prior.detach().cpu()),
        }
    with torch.no_grad():
        r_final = torch.einsum("bk,kct->bct", alpha, basis_t).detach().cpu().numpy().astype(np.float32)
        alpha_final = alpha.detach().cpu().numpy().astype(np.float32)
    alpha_change = np.linalg.norm(alpha_final - beta0, axis=1) / (
        np.linalg.norm(beta0, axis=1) + 1.0e-8
    )
    return r_final, alpha_final, {
        **losses,
        "free_map_alpha_change_mean": float(np.mean(alpha_change)),
    }


def _moving_average_torch(x, width=25):
    pad = width // 2
    weight = torch.ones((x.shape[1], 1, width), dtype=x.dtype, device=x.device) / float(width)
    xp = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    return torch.nn.functional.conv1d(xp, weight, groups=x.shape[1])


def apply_ratio_cap(r_hat, cap_ratio):
    out = r_hat.copy()
    front = rms(out[:, VIEWS["frontal7"], :], axis=(1, 2))
    post = rms(out[:, VIEWS["posterior5"], :], axis=(1, 2))
    caps = np.full(out.shape[0], float(cap_ratio), dtype=np.float32) if np.isscalar(cap_ratio) else np.asarray(cap_ratio, dtype=np.float32)
    scale = np.minimum(1.0, (caps * front) / (post + 1.0e-8)).astype(np.float32)
    out[:, VIEWS["posterior5"], :] *= scale[:, None, None]
    return out.astype(np.float32), {
        "cap_activation_rate": float(np.mean(scale < 0.999)),
        "cap_scale_mean": float(np.mean(scale)),
        "cap_scale_min": float(np.min(scale)),
        "cap_scale_max": float(np.max(scale)),
    }


def fit_map_ridge(x_train, beta_oracle_train):
    weights, (target_mean, target_std), meta = fit_scaled_ridge(
        x_train, beta_oracle_train, x_train
    )
    # Portable dual-form weights so query-time prediction is a single matmul.
    k = x_train @ x_train.T
    alpha = float(meta["ridge_alpha"])
    dual = np.linalg.solve(
        k + alpha * np.eye(k.shape[0], dtype=np.float32),
        (beta_oracle_train - target_mean) / target_std,
    )
    portable = (x_train.T @ dual).astype(np.float32)
    check = ((x_train @ portable) * target_std + target_mean).astype(np.float32)
    if float(np.max(np.abs(check - weights))) > 1.0e-6:
        raise RuntimeError("Portable ridge check failed")
    return {
        "weights": portable,
        "target_mean": target_mean,
        "target_std": target_std,
        "alpha": alpha,
    }, meta

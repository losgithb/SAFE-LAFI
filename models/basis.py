"""Continuity-topography residual basis."""

import numpy as np

from .config import N_CHANNELS, T, VIEWS
from .utils import moving_average, rms


def estimate_source_field(r_train):
    """Per-segment temporal source and topography from frontal-seven SVD."""
    frontal7, fp_pair = VIEWS["frontal7"], VIEWS["fp_pair"]
    sources, topographies, amplitudes = [], [], []
    for sample in r_train:
        frontal = sample[frontal7, :]
        _, _, vt = np.linalg.svd(frontal.astype(np.float64), full_matrices=False)
        source = vt[0]
        fp_mean = np.mean(sample[fp_pair, :], axis=0)
        if np.dot(source, fp_mean) < 0:
            source = -source
        source = source / (rms(source) + 1.0e-8)
        denom = float(np.dot(source, source)) + 1.0e-8
        topo_raw = np.array(
            [np.dot(sample[ch], source) / denom for ch in range(N_CHANNELS)],
            dtype=np.float64,
        )
        amp = float(rms(topo_raw[frontal7]))
        sources.append(source.astype(np.float32))
        topographies.append((topo_raw / (amp + 1.0e-8)).astype(np.float32))
        amplitudes.append(amp)

    sources = np.stack(sources, axis=0).astype(np.float32)
    topographies = np.stack(topographies, axis=0).astype(np.float32)
    return {
        "sources": sources,
        "topographies": topographies,
        "raw_amplitudes": np.asarray(amplitudes, dtype=np.float32),
    }


def temporal_atoms_from_sources(sources, kt=16):
    _, _, vt = np.linalg.svd(sources.astype(np.float64), full_matrices=False)
    atoms = vt[:kt].astype(np.float32)
    return atoms / (rms(atoms, axis=1, keepdims=True) + 1.0e-8)


def build_topography_model(source_field, ktop=5):
    topographies = source_field["topographies"].astype(np.float64)
    mean_topo = np.mean(topographies, axis=0)
    mean_topo = mean_topo / (rms(mean_topo[VIEWS["frontal7"]]) + 1.0e-8)
    centered = topographies - mean_topo[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pcs = vt[: max(0, ktop - 1)].astype(np.float32)
    return {"ktop": ktop, "mean": mean_topo.astype(np.float32), "pcs": pcs}


def build_basis(topo_model, atoms):
    """Outer-product atoms: 5 topographic components x 16 temporal modes."""
    topo_parts = [topo_model["mean"]] + list(topo_model["pcs"])
    topo_arr = np.stack(topo_parts, axis=0).astype(np.float32)
    topo_arr = topo_arr / (rms(topo_arr[:, VIEWS["frontal7"]], axis=1, keepdims=True) + 1.0e-8)
    basis = [
        topo[:, None] * atom[None, :]
        for topo in topo_arr
        for atom in atoms
    ]
    return np.stack(basis, axis=0).astype(np.float32)


def project_onto_basis(field, basis, ridge=1.0e-6, weights=None):
    if weights is None:
        weights = np.ones(N_CHANNELS, dtype=np.float32)
    bw = (basis * weights[None, :, None]).reshape(basis.shape[0], -1).astype(np.float64)
    rf = (field * weights[None, :, None]).reshape(field.shape[0], -1).astype(np.float64)
    gram = bw @ bw.T + ridge * np.eye(bw.shape[0], dtype=np.float64)
    return np.linalg.solve(gram, (rf @ bw.T).T).T.astype(np.float32)


def synthesize_from_basis(beta, basis):
    flat = beta.astype(np.float32) @ basis.reshape(basis.shape[0], -1).astype(np.float32)
    return flat.reshape(beta.shape[0], N_CHANNELS, T).astype(np.float32)


def projection_weights():
    weights = np.ones(N_CHANNELS, dtype=np.float32)
    weights[VIEWS["fp_pair"]] = 4.0
    weights[[10, 11, 16]] = 3.0
    weights[[2, 3]] = 2.0
    weights[VIEWS["posterior5"]] = 0.6
    return weights


def atom_diagnostics(basis, kt=16):
    """Per-atom spatial/temporal energy labels used by the safety groups."""
    frontal7, fp_pair = VIEWS["frontal7"], VIEWS["fp_pair"]
    high5, post5 = VIEWS["high_frontal5"], VIEWS["posterior5"]
    fp_ratios, post_ratios, deriv_vals = [], [], []
    for atom in basis:
        fp_ratios.append(rms(atom[fp_pair]) / (rms(atom[frontal7]) + 1.0e-8))
        post_ratios.append(rms(atom[post5]) / (rms(atom[frontal7]) + 1.0e-8))
        deriv_vals.append(rms(np.diff(atom, axis=1)))
    fp_q75 = np.quantile(fp_ratios, 0.75)
    post_q75 = np.quantile(post_ratios, 0.75)
    deriv_q75 = np.quantile(deriv_vals, 0.75)
    deriv_q25 = np.quantile(deriv_vals, 0.25)

    rows = []
    for j, atom in enumerate(basis):
        front = float(rms(atom[frontal7]))
        fp = float(rms(atom[fp_pair]))
        high = float(rms(atom[high5]))
        post = float(rms(atom[post5]))
        deriv = float(rms(np.diff(atom, axis=1)))
        smooth = float(rms(moving_average(atom[None, :, :], width=81)[0]))
        fp_ratio = fp / (front + 1.0e-8)
        post_ratio = post / (front + 1.0e-8)
        high_ratio = high / (front + 1.0e-8)
        labels = []
        if fp_ratio >= fp_q75:
            labels.append("fp_dominant")
        if high_ratio >= 0.95:
            labels.append("highfrontal_dominant")
        if post_ratio >= post_q75:
            labels.append("posterior_heavy")
        else:
            labels.append("posterior_light")
        if deriv >= deriv_q75:
            labels.append("temporal_derivative_heavy")
        if deriv <= deriv_q25 or smooth / (float(rms(atom)) + 1.0e-8) > 0.80:
            labels.append("lowfreq_slow")
        rows.append({
            "coef_index": j,
            "topography_group": f"topo_{j // kt}",
            "temporal_group": f"temp_{j % kt}",
            "topography_index": int(j // kt),
            "temporal_index": int(j % kt),
            "frontal7_rms": front,
            "fp_pair_rms": fp,
            "high_frontal5_rms": high,
            "posterior5_rms": post,
            "fp_over_front": fp_ratio,
            "posterior_over_front": post_ratio,
            "high_over_front": high_ratio,
            "derivative_energy": deriv,
            "smooth_energy": smooth,
            "soft_labels": ";".join(labels),
        })
    return rows


def _indices_for(rows, token):
    return np.asarray(
        [r["coef_index"] for r in rows if token in str(r["soft_labels"])], dtype=int
    )


def group_indices_from_atoms(rows):
    return {
        "fp_dominant": _indices_for(rows, "fp_dominant"),
        "highfrontal_dominant": _indices_for(rows, "highfrontal_dominant"),
        "posterior_heavy": _indices_for(rows, "posterior_heavy"),
        "posterior_light": _indices_for(rows, "posterior_light"),
        "lowfreq_slow": _indices_for(rows, "lowfreq_slow"),
        "temporal_derivative_heavy": _indices_for(rows, "temporal_derivative_heavy"),
    }


def proto_group_indices(rows):
    groups = group_indices_from_atoms(rows)
    post_idx = groups["posterior_heavy"]
    fp_idx = np.unique(
        np.concatenate([groups["fp_dominant"], groups["highfrontal_dominant"]])
    ).astype(int)
    return post_idx, fp_idx


def classify_atoms(basis):
    """Route-level beta groups (fp/highfrontal source, posterior-heavy, ...)."""
    ratios, fpfront_vals = [], []
    for atom in basis:
        front = _energy(atom, "frontal7")
        post = _energy(atom, "posterior5")
        fp = _energy(atom, "fp_pair")
        hf = _energy(atom, "high_frontal5")
        ratios.append(post / (front + 1.0e-8))
        fpfront_vals.append(0.5 * (fp + hf))
    ratios_np = np.asarray(ratios, dtype=np.float32)
    fpfront_np = np.asarray(fpfront_vals, dtype=np.float32)
    ratio_p65 = float(np.quantile(ratios_np, 0.65))
    ratio_p75 = float(np.quantile(ratios_np, 0.75))
    ratio_p85 = float(np.quantile(ratios_np, 0.85))
    fp_p60 = float(np.quantile(fpfront_np, 0.60))
    fp_p45 = float(np.quantile(fpfront_np, 0.45))

    groups = {
        "fp_highfront_source_atoms": [],
        "frontal_source_atoms": [],
        "safe_propagation_atoms": [],
        "posterior_heavy_atoms": [],
        "neutral_atoms": [],
    }
    for j, atom in enumerate(basis):
        fp = _energy(atom, "fp_pair")
        hf = _energy(atom, "high_frontal5")
        frontal = _energy(atom, "frontal7")
        posterior = _energy(atom, "posterior5")
        fpfront = 0.5 * (fp + hf)
        post_front = posterior / (frontal + 1.0e-8)
        post_fp = posterior / (fpfront + 1.0e-8)
        if post_front >= max(0.75, ratio_p75) or post_fp >= max(1.0, ratio_p85):
            group = "posterior_heavy_atoms"
        elif fpfront >= fp_p60 and post_front <= ratio_p65:
            group = "fp_highfront_source_atoms"
        elif frontal >= fp_p45 and post_front <= ratio_p75:
            group = "frontal_source_atoms"
        elif posterior > 0 and post_front <= ratio_p85:
            group = "safe_propagation_atoms"
        else:
            group = "neutral_atoms"
        groups[group].append(j)
    groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
    groups["source_atoms"] = np.unique(
        np.concatenate([groups["fp_highfront_source_atoms"], groups["frontal_source_atoms"]])
    ).astype(np.int64)
    return groups


def _energy(atom, view_name):
    from .utils import rms64
    return float(rms64(atom[VIEWS[view_name], :]))


def first_pc_waveforms(frontal):
    pcs = []
    for sample in frontal:
        _, _, vt = np.linalg.svd(sample.astype(np.float64), full_matrices=False)
        pc = vt[0]
        fp_mean = np.mean(sample[:2, :], axis=0)
        if np.dot(pc, fp_mean) < 0:
            pc = -pc
        pcs.append((pc / (rms(pc) + 1.0e-8)).astype(np.float32))
    return np.stack(pcs, axis=0)


def topo_from_sources(field, sources):
    denom = np.sum(sources * sources, axis=1) + 1.0e-8
    return (np.einsum("nct,nt->nc", field, sources) / denom[:, None]).astype(np.float32)


def field_from_topo_source(topo, sources):
    return (topo[:, :, None] * sources[:, None, :]).astype(np.float32)


def estimate_y_sources(y):
    sources = []
    for i in range(y.shape[0]):
        f = y[i, VIEWS["frontal7"], :].astype(np.float64)
        f = f - f.mean(axis=1, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(f, full_matrices=False)
            s = vh[0]
        except np.linalg.LinAlgError:
            s = np.mean(f[:2], axis=0)
        fp_mean = np.mean(y[i, VIEWS["fp_pair"], :], axis=0)
        if float(np.dot(s, fp_mean)) < 0:
            s = -s
        s = s - np.mean(s)
        s = s / (np.sqrt(np.mean(s * s)) + 1.0e-8)
        sources.append(s.astype(np.float32))
    return np.stack(sources, axis=0).astype(np.float32)

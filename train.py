"""Train the model on the semi-simulated ocular-artifact benchmark."""

from pathlib import Path

import numpy as np

from models import apply, fit_fold, load_state
from models.apply import apply_emission
from opts import get_opts
from preprocess.data import load_current19, rotating_splits, select_windows
from tools import fold_metrics


def main():
    opts = get_opts()
    clean_mat = Path(opts.EEG_path)
    contaminated_mat = Path(opts.NOS_path)
    save_dir = Path(opts.save_dir)

    x, y, rids = load_current19(clean_mat, contaminated_mat)
    _, val_list, _ = rotating_splits()

    rows = []
    for fold in range(10):
        state_dir, report = fit_fold(
            clean_mat, contaminated_mat, fold, save_dir, device=opts.device
        )
        state = load_state(state_dir, device=opts.device)
        (x_val,) = select_windows((x,), rids, val_list[fold])
        (y_val,) = select_windows((y,), rids, val_list[fold])
        out = apply(state, y_val)
        r_opt, _ = apply_emission(out["r_hat"], shrink=0.75, gate_uv=20.0)
        rows.append(fold_metrics(y_val, x_val, y_val - r_opt))
        print(f"fold {fold}: {report['selected_config']}  "
              f"rrmse={rows[-1]['rrmse']:.4f}  "
              f"cc={rows[-1]['cc']:.4f}  "
              f"delta_snr={rows[-1]['delta_snr_db']:.2f} dB")

    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("\nmean over folds:")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()))


if __name__ == "__main__":
    main()

"""Configuration and montage constants."""

FS = 200            # sampling rate (Hz)
T = 500             # samples per window
N_CHANNELS = 19
N_RECORDINGS = 54
WINDOWS_PER_RECORDING = 10
POST = [6, 7, 8, 9, 18]          # posterior-five channels

CHANNEL_ORDER = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T3", "C3", "CZ", "C4", "T4",
    "T5", "P3", "PZ", "P4", "T6", "O1", "O2",
]

VIEWS = {
    "full19": list(range(19)),
    "frontal7": [0, 1, 2, 3, 10, 11, 16],
    "high_frontal5": [0, 1, 10, 11, 16],
    "fp_pair": [0, 1],
    "posterior5": [6, 7, 8, 9, 18],
    "central_only": [4, 5, 17],
    "occipital2": [8, 9],
}

SEED = 20260531
BASIS_SEED = 20260530
EPS = 1.0e-8

# Floor-anchored residual delta (paper section 3.4).
ALPHA = {
    "fp": 0.72,
    "highfront": 0.62,
    "frontal": 0.35,
    "other": 0.03,
    "posterior": 0.0,
}
EVENT_THRESHOLD_SCALE = 1.10
EVENT_GAIN = 1.15
NON_EVENT_GAIN = 0.05

# Posterior-aware ridge projection (equation 9).
PROJECTION_RIDGE = 3.0e-6
PROJECTION_WEIGHTS = {"fp": 3.5, "highfront": 2.5, "frontal": 1.8, "posterior": 0.40}
PROJECTION_POST_SHRINK = 0.55

# Routing thresholds.
ALPHA_CHANGE_RATIO = 1.01
BATCH_GATE_QUANTILE = 0.80

# Optimized emission path (paper section 3.6 and appendix A).
EMISSION_POST_SHRINK = 0.75
EMISSION_GATE_UV = 20.0

# Hybrid encoder selection.
ELIGIBLE_FAMILIES = ["m1", "m2", "m3", "m5"]
SELECTION_FRACTION = 0.80
CANDIDATE_EPOCHS = 34
FINAL_EPOCHS = 54
HYBRID_LR = 1.2e-3
HYBRID_WEIGHT_DECAY = 1.0e-4
CRITIC_EPOCHS = 25
CRITIC_LR = 1.0e-3
CRITIC_WEIGHT_DECAY = 1.0e-4

VARIANTS = [
    {"name": "M1_SAFE_SELECT_BASE", "family": "m1", "clean": 0.6, "beta": 0.10,
     "source": 0.0, "post": 0.9, "non_event": 0.7, "critic": 0.0, "seed_offset": 2},
    {"name": "M2_BETA_COMBINED_SAFE", "family": "m2", "clean": 0.6, "beta": 0.12,
     "source": 0.20, "post": 1.0, "non_event": 0.8, "critic": 0.0,
     "beta_postheavy": 0.30, "beta_ratio": 0.30, "source_floor": 0.05,
     "post_shrink": 0.70, "post_ratio_cap": 0.80, "seed_offset": 3},
    {"name": "M3_DECOUPLED_COMBINED_SAFE", "family": "m3", "clean": 0.7, "beta": 0.12,
     "source": 0.25, "post": 1.2, "non_event": 0.9, "critic": 0.0,
     "beta_postheavy": 0.80, "beta_ratio": 0.80, "source_floor": 0.10,
     "post_shrink": 0.35, "post_ratio_cap": 0.55, "source_floor_boost": 0.10, "seed_offset": 4},
    {"name": "M5_COMBINED_GUARD", "family": "m5", "clean": 0.7, "beta": 0.12,
     "source": 0.25, "post": 1.4, "non_event": 1.0, "critic": 0.05,
     "beta_postheavy": 0.60, "beta_ratio": 0.60, "source_floor": 0.10,
     "post_shrink": 0.45, "post_ratio_cap": 0.60, "seed_offset": 5},
]

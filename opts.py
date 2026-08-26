"""Training options."""

import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--EEG_path", type=str, default="./Pure_Data.mat",
                        help="clean EEG data")
    parser.add_argument("--NOS_path", type=str, default="./Contaminated_Data.mat",
                        help="contaminated EEG data")
    parser.add_argument("--save_dir", type=str, default="./states/")
    return parser.parse_args()

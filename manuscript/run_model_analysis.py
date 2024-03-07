import os
import pickle
import argparse
import warnings

# NOTE: sklearn warns that models have not converged; however, increasing
# the number of training iterations until this warning disappears does
# not change the results (a better solution would be to change the
# convergence criteria in the sklearn functions).
warnings.filterwarnings("ignore")

import pandas as pd

from vam.model_analysis import ModelAnalysis
from vam.config import get_default_config


# This CLI runs the main model analyses for both the VAM and task-optimized models

parser = argparse.ArgumentParser()
parser.add_argument(
    "models_dir",
    help=(
        "Directory with models (checkpoints/splits). This should contain "
        "metadata.csv and three subdirectories: "
        "vam_models, task_opt_models, and binned_rt_models"
    ),
)
parser.add_argument("-u", "--users", nargs="*", help="Users to process, optional")
parser.add_argument(
    "-d",
    "--derivatives_dir",
    default="derivatives",
    nargs="?",
    type=str,
    help="Name of derivatives directory, optional",
)
parser.add_argument(
    "-s",
    "--summary_dir",
    default="summary_stats",
    nargs="?",
    type=str,
    help="Name of directory for summary stats (under derivatives directory), optional",
)
args = parser.parse_args()

models_dir = args.models_dir
derivatives_dir = args.derivatives_dir
summary_dir = args.summary_dir
metadata = pd.read_csv(os.path.join(models_dir, "metadata.csv"))
binned_rt_metadata = pd.read_csv(os.path.join(models_dir, "binned_rt_metadata.csv"))

binned_rt_users = binned_rt_metadata["user_id"].values
users = args.users
if users is None:
    users = metadata["user_id"].values

rand_seed = 1
config = get_default_config()
config.data.n_rt_bins = 5

for uid in users:
    print(f"Running analysis for user {uid}")

    analyzer = ModelAnalysis(
        models_dir,
        config,
        uid,
        rand_seed,
        derivatives_dir=derivatives_dir,
        summary_dir=summary_dir,
        has_binned_rt_model=uid in binned_rt_users,
    )
    analyzer.run_analysis()

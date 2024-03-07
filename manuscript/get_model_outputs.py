import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse

import jax.numpy as jnp
import pandas as pd

from vam.model_outputs import ModelOutputs
from vam.config import get_test_config


# This CLI gets the model outputs for the VAM, task-optimized, and binned RT models

parser = argparse.ArgumentParser()
parser.add_argument("inputs_dir", help="Directory with model inputs")
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
args = parser.parse_args()

inputs_dir = args.inputs_dir
models_dir = args.models_dir
derivatives_dir = args.derivatives_dir
metadata = pd.read_csv(os.path.join(models_dir, "metadata.csv"))
binned_rt_metadata = pd.read_csv(os.path.join(models_dir, "binned_rt_metadata.csv"))

binned_rt_users = binned_rt_metadata["user_id"].values
users = args.users
if users is None:
    users = metadata["user_id"].values

vam_epoch = 199
task_opt_epoch = 24
binned_rt_epoch = 499
n_rt_bins = 5
rand_seed = 1


def _get_binned_rt_outputs(
    inputs_dir, models_dir, epoch, rand_seed, derivatives_dir, user_id, n_rt_bins
):
    for rt_bin in range(n_rt_bins):
        config = get_test_config(
            "binned_rt", user_id, n_rt_bins=n_rt_bins, rt_bin=rt_bin
        )

        outputs = ModelOutputs(
            inputs_dir,
            models_dir,
            config,
            user_id,
            "binned_rt",
            epoch,
            rand_seed,
            derivatives_dir=derivatives_dir,
        )
        outputs.get_model_outputs()
        outputs.process_model_outputs()


for uid in users:
    print(f"Getting model outputs for user {uid}")
    for model_type, analysis_epoch in zip(
        ["vam", "task_opt", "binned_rt"],
        [vam_epoch, task_opt_epoch, binned_rt_epoch],
    ):
        config = get_test_config(model_type, uid)

        if model_type in ["vam", "task_opt"]:
            outputs = ModelOutputs(
                inputs_dir,
                models_dir,
                config,
                uid,
                model_type,
                analysis_epoch,
                rand_seed,
                derivatives_dir=derivatives_dir,
            )
            outputs.get_model_outputs()
            outputs.process_model_outputs()

        elif model_type == "binned_rt" and uid in binned_rt_users:
            _get_binned_rt_outputs(
                inputs_dir,
                models_dir,
                binned_rt_epoch,
                rand_seed,
                derivatives_dir,
                uid,
                n_rt_bins,
            )

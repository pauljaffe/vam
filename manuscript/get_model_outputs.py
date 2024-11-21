import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse

import jax.numpy as jnp
import pandas as pd

from vam.model_outputs import ModelOutputs
from vam.config import get_test_config


# This CLI gets the model outputs for the VAM and task-optimized models

parser = argparse.ArgumentParser()
parser.add_argument(
    "models_dir",
    help=(
        "Directory with models (checkpoints/splits). This should contain "
        "metadata.csv and two subdirectories: vam_models and task_opt_models"
    ),
)
parser.add_argument("-u", "--users", nargs="*", help="Users to process, optional")
parser.add_argument(
    "-i",
    "--inputs_dir",
    nargs="?",
    type=str,
    help="Full path to directory with model inputs, optional",
)
parser.add_argument(
    "-d",
    "--derivatives_dir",
    default="derivatives",
    nargs="?",
    type=str,
    help="Name of derivatives directory, optional",
)
args = parser.parse_args()

models_dir = args.models_dir
inputs_dir = args.inputs_dir
if inputs_dir is None:
    inputs_dir = os.path.join(models_dir, "model_inputs")

derivatives_dir = args.derivatives_dir
metadata = pd.read_csv(os.path.join(models_dir, "metadata.csv"))

users = args.users
if users is None:
    users = metadata["user_id"].values

vam_epoch = 199
task_opt_epoch = 24
n_rt_bins = 5
rand_seed = 1


for uid in users:
    print(f"Getting model outputs for user {uid}")
    for model_type, analysis_epoch in zip(
        ["vam", "task_opt"],
        [vam_epoch, task_opt_epoch],
    ):
        config = get_test_config(model_type, uid)

        outputs = ModelOutputs(
            models_dir,
            inputs_dir,
            config,
            uid,
            model_type,
            analysis_epoch,
            rand_seed,
            derivatives_dir=derivatives_dir,
        )
        outputs.get_model_outputs()
        outputs.process_model_outputs()

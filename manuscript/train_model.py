import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse

from vam.training import Trainer
from vam.config import get_config_from_cli


# This CLI trains a VAM on data specified in the command-line arguments.
# By default, a VAM will be trained using the default parameters.
# Set the -m flag to "task_opt" to train a task-optimized model and "binned_rt"
# to train a VAM on data from a specified RT quantile. See additional options below.

# For more control over the model architecture and training parameters,
# update the config struct returned by get_config_from_cli.

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Directory with data used to train the model")
parser.add_argument(
    "save_dir",
    help=(
        "Directory to save checkpoints and other info from model training. "
        "Note the info for this training run will be saved in "
        "the subfolder save_dir/expt_name."
    ),
)
parser.add_argument(
    "expt_name",
    help=(
        "Name for this experiment/run as it will be logged by wandb. The info "
        "for this training run will be saved in save_dir/run_name."
    ),
)
parser.add_argument(
    "-p",
    "--project",
    nargs="?",
    type=str,
    help="Name of the project in wandb to log the run to, optional",
)
parser.add_argument(
    "-n",
    "--notes",
    nargs="?",
    type=str,
    help="Notes that will be attached to the run logged in wandb, optional",
)
parser.add_argument(
    "-m",
    "--model_type",
    default="vam",
    nargs="?",
    type=str,
    help=(
        "Type of model to train. Set to 'vam' to train a VAM (default), "
        "'task_opt' to train a task-optimized model, and 'binned_rt' "
        "to train a VAM on data from a single RT quantile. If training "
        "a binned_rt model, the --rt_bin flag must be set, and optionally "
        "the n_rt_bins flag."
    ),
)
parser.add_argument(
    "--n_rt_bins",
    default=5,
    nargs="?",
    type=int,
    help=(
        "Number of RT quantiles to divide data into (default=5), "
        "only relevant for binned_rt model training."
    ),
)
parser.add_argument(
    "--rt_bin",
    nargs="?",
    type=int,
    help=(
        "Which RT quantile to train the data on, should be an integer "
        "between 0 and n_rt_bins-1 (inclusive), only relevant for "
        "binned_rt model training."
    ),
)

args = parser.parse_args()
config = get_config_from_cli(args)

trainer = Trainer(config, args.save_dir, args.data_dir)
trainer.train()

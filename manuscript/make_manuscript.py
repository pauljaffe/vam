# Run the analyses and reproduce the figures from the paper.
# See the README for detailed instructions.
import os
import time
import pickle
import argparse
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from vam.config import get_default_config
from figures import (
    Figure2,
    Figure3,
    Figure4,
    Figure5,
    Figure6,
    FigureS1,
    FigureS2,
    FigureS3,
    FigureS4,
    FigureS5,
    FigureS6,
    FigureS7,
    FigureS8,
)


rand_seed = 0
n_bootstrap = 1000
fontsize = 6
titlesize = 8
linewidth = 0.5
plot_figs = True

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["axes.titlesize"] = titlesize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = fontsize
plt.rcParams["lines.linewidth"] = linewidth
plt.rcParams["legend.title_fontsize"] = fontsize

parser = argparse.ArgumentParser()
parser.add_argument(
    "models_dir",
    help=(
        "Directory with models (checkpoints/splits). This should contain "
        "metadata.csv and three subdirectories: vam_models, task_opt_models, "
        "and derivatives"
    ),
)
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
    help="Name of summary stats directory, optional",
)
parser.add_argument(
    "-f",
    "--figures_to_make",
    nargs="*",
    help="Figures to make, optional. E.g. -f 2 3 S1",
)
args = parser.parse_args()

models_dir = args.models_dir
derivs_dir = os.path.join(models_dir, args.derivatives_dir)

figures_to_make = args.figures_to_make
if figures_to_make is None:
    figures_to_make = [
        "2",
        "3",
        "4",
        "5",
        "6",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
    ]

metadata = pd.read_csv(os.path.join(models_dir, "metadata.csv"))

config = get_default_config()

# Load stats for each model
all_stats = []
for row_idx, row in metadata.iterrows():
    user_id = str(row["user_id"])
    user_stats_fn = os.path.join(
        derivs_dir, args.summary_dir, f"user{user_id}", "summary_stats.pkl"
    )
    with open(user_stats_fn, "rb") as f:
        stats = pickle.load(f)
    all_stats.append(stats)

stats_keys = all_stats[0].keys()
stats_dict = {key: pd.concat([s[key] for s in all_stats]) for key in stats_keys}

figure_inputs = (
    stats_dict,
    derivs_dir,
    metadata,
    config,
    rand_seed,
    n_bootstrap,
    args.summary_dir,
)
figure_map = {
    "2": Figure2,
    "3": Figure3,
    "4": Figure4,
    "5": Figure5,
    "6": Figure6,
    "S1": FigureS1,
    "S2": FigureS2,
    "S3": FigureS3,
    "S4": FigureS4,
    "S5": FigureS5,
    "S6": FigureS6,
    "S7": FigureS7,
    "S8": FigureS8,
}

for key in figures_to_make:
    if key not in figure_map:
        raise ValueError(f"Figure {key} not found")
    figure_map[key](*figure_inputs).make_figure()

if plot_figs:
    plt.tight_layout()
    plt.show()

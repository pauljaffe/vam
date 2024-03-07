import os
from PIL import Image
import argparse

import numpy as np
import pandas as pd
from psychopy import visual, core
import torchvision.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument(
    "base_dir",
    help=(
        "This directory should contain the gameplay_data and graphics folders, "
        " and the metadata.csv file."
        "A model_inputs directory will be created with the processed "
        "stimuli and other inputs for model training."
    ),
)
parser.add_argument("-u", "--users", nargs="*", help="Users to process, optional")
args = parser.parse_args()

base_dir = args.base_dir
users = args.users
metadata = pd.read_csv(os.path.join(base_dir, "metadata.csv"))

if users is None:
    users = metadata["user_id"].values


def prep_model_inputs(
    base_dir,
    user_id,
    min_rt=250,  # ms
    outlier_mult=10,
    resize_h=128,
    resize_w=128,
    n_trials=25000,
):
    """
    Creates model inputs from a single user's Lost in Migration
    gameplay data.
    """

    graphics_dir = os.path.join(base_dir, "graphics")
    bgrnd_path = os.path.join(graphics_dir, "bkgrnd.png")
    save_dir = os.path.join(base_dir, "model_inputs")
    os.makedirs(save_dir, exist_ok=True)

    raw_path = os.path.join(base_dir, "gameplay_data", f"user{user_id}df.csv")
    chunks = pd.read_csv(raw_path, header=0, chunksize=1e6)
    df = pd.concat(chunks)
    n_orig = len(df)

    dir_map = {"L": 0, "R": 1, "U": 2, "D": 3}
    horiz_line_space = [51, 0]
    vert_line_space = [0, 51]
    cross_space = [51, 51]
    v_space = [34, 34]
    layout_key = {
        0: horiz_line_space,  # horizontal line
        1: vert_line_space,  # vertical line
        2: cross_space,  # cross
        3: v_space,  # V left
        4: v_space,  # V right
        5: v_space,  # V down
        6: v_space,  # V up
    }
    win_size = [640, 480]
    keys = [
        "targ_dirs",
        "dis_dirs",
        "response_dirs",
        "rts",
        "congruency",
        "layouts",
        "xpositions",
        "ypositions",
    ]

    if min_rt is not None:
        df = df[df["response_time"] >= min_rt]
        print(f"Filtered out {n_orig - len(df)} trials with RT < {min_rt}")
    df, _ = filter_rt_outliers(df, outlier_mult, "response_time")

    stim_dir = os.path.join(save_dir, f"user{user_id}", "processed_stimuli")
    os.makedirs(stim_dir, exist_ok=True)

    trial_ind = 0
    data_dict = {k: np.zeros((n_trials)) for k in keys}
    win = visual.Window(
        win_size,
        monitor="testMonitor",
        units="pix",
        colorSpace="rgb",
        backgroundImage=bgrnd_path,
    )

    for row in df.itertuples(index=False):
        row = row._asdict()

        if trial_ind % 1000 == 0:
            print(f"Processed {trial_ind}/{n_trials} trials for user {user_id}")

        targ_dir = dir_map[row["target_direction"]]
        dis_dir = dir_map[row["flanker_direction"]]
        resp_dir = dir_map[row["response_direction"]]
        rt = row["response_time"]
        layout = row["stimulus_layout"]
        xpos = row["xpos"]
        ypos = row["ypos"]
        spacer = layout_key[layout]
        if targ_dir == dis_dir:
            con = 0
        else:
            con = 1

        # Save raw and processed stimulus
        xpos_centered = xpos - win_size[0] / 2
        ypos_centered = -ypos + win_size[1] / 2
        win = _make_lim_stimulus(
            graphics_dir,
            win,
            (xpos_centered, ypos_centered),
            targ_dir,
            dis_dir,
            layout,
            spacer,
        )

        img = win._getFrame(buffer="back")
        _make_processed_stim(img, stim_dir, trial_ind, resize_h, resize_w)
        win.flip()

        data_dict["targ_dirs"][trial_ind] = targ_dir
        data_dict["dis_dirs"][trial_ind] = dis_dir
        data_dict["response_dirs"][trial_ind] = resp_dir
        data_dict["rts"][trial_ind] = rt
        data_dict["congruency"][trial_ind] = con
        data_dict["layouts"][trial_ind] = layout
        data_dict["xpositions"][trial_ind] = xpos_centered
        data_dict["ypositions"][trial_ind] = ypos_centered
        trial_ind += 1

        if trial_ind >= n_trials:
            break

    # Save other data
    win.close()
    user_dir = os.path.join(save_dir, f"user{user_id}")
    for key in keys:
        np.save(os.path.join(user_dir, f"{key}.npy"), data_dict[key])


def _make_lim_stimulus(img_dir, win, targ_pos, d_targ, d_dis, layout, spacer):
    dis_path = os.path.join(img_dir, f"bird{d_dis}.png")
    targ_path = os.path.join(img_dir, f"bird{d_targ}.png")
    # Draw target stimulus
    _add_bird(win, targ_path, targ_pos)
    # Draw distractors
    dis_pos = _get_distractor_pos(targ_pos, layout, spacer)
    for dpos in dis_pos:
        _add_bird(win, dis_path, dpos)
    return win


def _make_processed_stim(im, save_dir, trial_ind, resize_h, resize_w):
    resized = T.functional.resize(im, [resize_h, resize_w])
    orig_img_np = np.reshape(
        np.asarray(resized).astype("float32"), (resize_h, resize_w, 3)
    )
    orig_img_np = orig_img_np / 255.0
    with open(os.path.join(save_dir, f"img{trial_ind}.npy"), "wb") as f:
        np.save(f, orig_img_np)


def _add_bird(win, img_path, pos):
    stim = visual.ImageStim(win, image=img_path, pos=pos, colorSpace="rgb")
    stim.draw()


def _get_distractor_pos(targ_pos, layout, spacer):
    if layout == 0:  # horizontal line
        dis_pos = [
            (targ_pos[0] - 2 * spacer[0], targ_pos[1]),
            (targ_pos[0] - spacer[0], targ_pos[1]),
            (targ_pos[0] + spacer[0], targ_pos[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1]),
        ]
    elif layout == 1:  # vertical line
        dis_pos = [
            (targ_pos[0], targ_pos[1] - 2 * spacer[1]),
            (targ_pos[0], targ_pos[1] - spacer[1]),
            (targ_pos[0], targ_pos[1] + spacer[1]),
            (targ_pos[0], targ_pos[1] + 2 * spacer[1]),
        ]
    elif layout == 2:  # cross
        dis_pos = [
            (targ_pos[0] - spacer[0], targ_pos[1]),
            (targ_pos[0] + spacer[0], targ_pos[1]),
            (targ_pos[0], targ_pos[1] - spacer[1]),
            (targ_pos[0], targ_pos[1] + spacer[1]),
        ]
    elif layout == 3:  # V left
        dis_pos = [
            (targ_pos[0] + spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
            (targ_pos[0] + spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
        ]
    elif layout == 4:  # V right
        dis_pos = [
            (targ_pos[0] - spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
            (targ_pos[0] - spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
        ]
    elif layout == 5:  # V down
        dis_pos = [
            (targ_pos[0] - spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
            (targ_pos[0] + spacer[0], targ_pos[1] + spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] + 2 * spacer[1]),
        ]
    elif layout == 6:  # V up
        dis_pos = [
            (targ_pos[0] - spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] - 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
            (targ_pos[0] + spacer[0], targ_pos[1] - spacer[1]),
            (targ_pos[0] + 2 * spacer[0], targ_pos[1] - 2 * spacer[1]),
        ]
    return dis_pos


def _median_deviations(data, median=None):
    if median is None:
        median = np.median(data)
    return np.abs(data - median)


def filter_rt_outliers(df, mult, column, median=None, mad=None):
    df = df.copy(deep=True)
    devs = _median_deviations(df[column].values, median=median)
    df.loc[:, "devs"] = devs
    if mad is None:
        mad = np.median(devs)
    df_filt = df.query("devs < @mult * @mad")
    keep_idx = df_filt.index.values
    n_filt = len(df) - len(df_filt)
    f_filt = n_filt / len(df)
    df_filt = df_filt.drop(columns=["devs"])
    print(f"Dropped n = {n_filt}/{len(df)} = {f_filt} outliers from {column}")
    return df_filt, keep_idx


for uid in users:
    prep_model_inputs(base_dir, uid)

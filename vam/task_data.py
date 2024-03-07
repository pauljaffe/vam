import pdb
import os

import jax.numpy as jnp
from jax import random
import jax
import numpy as np
import augmax
from augmax import ColorJitter
import pandas as pd

from .lba import jittable_sim_lba
from .transforms import StochasticWarp, RandomTranslate


class TaskData:
    """
    Container for cognitive task data (currently limited to Lost in Migration).
    Implements data augmentation and data loader for use in model training.
    """

    def __init__(
        self,
        data_dir,
        save_dir,
        rand_key,
        config,
        split_dir=None,
        data_save_dir=None,
    ):
        self.data_dir = data_dir
        if split_dir is None:
            self.split_dir = os.path.join(save_dir, config.expt_name, "splits")
        else:
            self.split_dir = split_dir
        self.img_dir = os.path.join(data_dir, "processed_stimuli")
        self.expt_name = config.expt_name
        self.train_frac = config.training.train_frac
        self.val_frac = config.training.val_frac
        self.rand_key = rand_key
        self.n_prep = config.data.n_prep
        self.splits = config.data.splits
        self.batch_size = config.training.batch_size
        self.prep_type = config.data.data_prep_type
        self.data_save_dir = data_save_dir
        self.transform = self._build_transform(config)
        self.data_keys = [
            "response_dirs",
            "rts",
            "targ_dirs",
            "dis_dirs",
            "congruency",
            "layouts",
            "xpositions",
            "ypositions",
        ]

        # Prep data for training (define splits etc.)
        if self.prep_type == "slow":
            self.full_keys = ["trial_idx"] + self.data_keys
            self.split_data = self._prep_data_slow()
        elif self.prep_type == "fast":
            if self.data_save_dir is None:
                self.data_save_dir = os.path.join(save_dir, config.expt_name, "data")
            self.full_keys = ["imgs"] + self.data_keys
            self.split_data = self._prep_data_fast()
        elif self.prep_type == "binned_rt":
            if self.data_save_dir is None:
                self.data_save_dir = os.path.join(save_dir, config.expt_name, "data")
            self.full_keys = ["imgs"] + self.data_keys
            self.split_data = self._prep_binned_rt_data(config)

    def _prep_binned_rt_data(self, config):
        n_rt_bins = config.data.n_rt_bins
        rt_bin = config.data.rt_bin
        with open(os.path.join(self.data_dir, "rts.npy"), "rb") as f:
            rts = jnp.load(f) / 1000
        rt_bins = pd.qcut(rts, n_rt_bins, labels=False)
        idx = np.arange(len(rts))
        bin_idx = idx[rt_bins == rt_bin]
        split_data = self._prep_data_fast(trial_idx=bin_idx)
        return split_data

    def _load_cast_data(self, data_dir, trial_idx=None, save_dir=None):
        data = {}
        for key in self.data_keys:
            with open(os.path.join(data_dir, f"{key}.npy"), "rb") as f:
                if key in [
                    "response_dirs",
                    "targ_dirs",
                    "dis_dirs",
                    "congruency",
                    "layouts",
                ]:
                    data[key] = jnp.asarray(jnp.load(f), dtype=jnp.int32)
                else:
                    data[key] = jnp.load(f)

                if trial_idx is not None:
                    data[key] = data[key][trial_idx]

        if save_dir is not None:
            for key in self.data_keys:
                with open(os.path.join(save_dir, f"{key}.npy"), "wb") as f:
                    jnp.save(f, data[key])

        return data

    def _prep_data_slow(self):
        data = self._load_cast_data(self.data_dir)
        data["rts"] = data["rts"] / 1000
        data["trial_idx"] = jnp.arange(len(data["targ_dirs"]))

        split_data = {}
        split_idx = self._get_split_idx(len(data["targ_dirs"]))
        for split in self.splits:
            split_data[split] = self._get_split(split_idx[split], data)
        return split_data

    def _prep_data_fast(self, trial_idx=None):
        imgs_path = os.path.join(self.data_save_dir, "prep_imgs.npy")
        if os.path.isfile(imgs_path):
            with open(imgs_path, "rb") as f:
                prep_imgs = jnp.load(f)
            data = self._load_cast_data(self.data_save_dir)
        else:
            os.makedirs(self.data_save_dir)
            if trial_idx is None:
                with open(os.path.join(self.data_dir, f"targ_dirs.npy"), "rb") as f:
                    n_total = len(jnp.load(f))
                prep_idx = random.permutation(self.rand_key, np.arange(n_total))[
                    : self.n_prep
                ]
            else:
                prep_idx = random.permutation(self.rand_key, trial_idx)

            prep_imgs = jnp.zeros((len(prep_idx), 128, 128, 3))
            for i, idx in enumerate(prep_idx):
                if i % 100 == 0:
                    print(f"Loaded {i}/{len(prep_idx)} images")
                with open(os.path.join(self.img_dir, f"img{idx}.npy"), "rb") as f:
                    prep_imgs = prep_imgs.at[i, :, :, :].set(jnp.load(f))
            with open(imgs_path, "wb") as f:
                jnp.save(f, prep_imgs)

            data = self._load_cast_data(
                self.data_dir,
                trial_idx=prep_idx,
                save_dir=self.data_save_dir,
            )

        data["rts"] = data["rts"] / 1000
        data["imgs"] = prep_imgs

        split_data = {}
        split_idx = self._get_split_idx(len(prep_imgs))
        for split in self.splits:
            split_data[split] = self._get_split(split_idx[split], data)
        return split_data

    def _build_transform(self, config):
        if config.data.do_data_aug:
            transform = augmax.Chain(
                RandomTranslate(config.data.translate_xmax, config.data.translate_ymax),
                StochasticWarp(
                    config.data.p_warp,
                    config.data.warp_str,
                    config.data.warp_coarseness,
                ),
            )
            transform = jax.jit(jax.vmap(transform))
        else:
            transform = None
        return transform

    def _get_split_idx(self, n_data):
        splits = ["train", "val", "test"]
        split_idx = {}
        for split in splits:
            try:
                with open(os.path.join(self.split_dir, f"{split}_idx.npy"), "rb") as f:
                    split_idx[split] = jnp.load(f)
            except FileNotFoundError:
                os.makedirs(self.split_dir)
                n_train = int(jnp.round(n_data * self.train_frac))
                n_val = int(jnp.round(n_data * self.val_frac))
                idx = jnp.arange(n_data)
                sidx = random.permutation(self.rand_key, idx)
                split_idx["train"] = sidx[:n_train]
                split_idx["val"] = sidx[n_train : n_train + n_val]
                split_idx["test"] = sidx[n_train + n_val :]
                for split in split_idx.keys():
                    with open(
                        os.path.join(self.split_dir, f"{split}_idx.npy"), "wb"
                    ) as f:
                        jnp.save(f, split_idx[split])
        return split_idx

    def _get_split(self, idx, data):
        return {key: data[key][idx] for key in self.full_keys}

    def data_generator(self, key, split, shuffle=True):
        data = self.split_data[split]
        n_split = len(data["targ_dirs"])
        num_batches = int(np.ceil(n_split / self.batch_size))
        if shuffle:
            idx = random.permutation(key, n_split)
            key, subkey = random.split(key)
        else:
            idx = np.arange(n_split)

        for batch in range(num_batches):
            curr_idx = idx[batch * self.batch_size : (batch + 1) * self.batch_size]
            if self.prep_type == "slow":
                batch_idx = data["trial_idx"][curr_idx]
                batch_imgs = []
                for bi in batch_idx:
                    with open(os.path.join(self.img_dir, f"img{bi}.npy"), "rb") as f:
                        batch_imgs.append(jnp.load(f))
                batch_imgs = jnp.stack(batch_imgs, axis=0)
            elif self.prep_type in ["fast", "binned_rt"]:
                batch_imgs = data["imgs"][curr_idx]

            if self.transform is not None:
                sub_keys = random.split(key, len(batch_imgs))
                batch_imgs = self.transform(sub_keys, batch_imgs)

            yield tuple([batch_imgs] + [data[i][curr_idx] for i in self.data_keys])

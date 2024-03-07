import pdb
import pickle
import os
import itertools
import time
import shutil

import jax
from jax import random
import jax.numpy as jnp
from flax.training import train_state
import numpy as np
import pandas as pd
import h5py

from .training import Trainer
from .utils import get_vam_lba_params
from .lba import generate_vam_rts


def vam_test_step(state: train_state.TrainState, cnn, batch, root_key, n_acc):
    root_key, mc_key = random.split(root_key)
    imgs, choices, rts, targ, dis = batch[0], batch[1], batch[2], batch[3], batch[4]
    (
        elbo,
        drifts,
    ) = state.apply_fn(
        {"params": state.params},
        imgs,
        rts,
        choices,
        mc_key,
        training=False,
    )
    n = imgs.shape[0]
    lba_params = get_vam_lba_params(state)
    root_key, mc_key = random.split(root_key)
    sim_data = generate_vam_rts(lba_params, drifts, n_acc, mc_key)

    _, mod_vars = cnn.apply(
        {"params": state.params["get_drifts"]},
        imgs,
        training=False,
        mutable="intermediates",
    )
    features = mod_vars["intermediates"]["features"]

    return (
        state,
        root_key,
        sim_data,
        np.array(drifts),
        features,
    )


def task_opt_test_step(state: train_state.TrainState, batch):
    imgs = batch[0]

    logits, mod_vars = state.apply_fn(
        {"params": state.params},
        imgs,
        training=False,
        mutable="intermediates",
    )

    choices = logits.argmax(axis=-1)
    features = mod_vars["intermediates"]["features"]

    return state, {"response_dirs": choices}, features


class ModelOutputs:
    def __init__(
        self,
        models_dir,
        inputs_dir,
        config,
        user_id,
        model_type,
        analysis_epoch,
        rand_seed,
        derivatives_dir="derivatives",
        filter_dead_units=True,
    ):
        self.model_inputs_dir = os.path.join(inputs_dir, f"user{user_id}")
        self.config = config
        self.user_id = user_id
        self.model_type = model_type
        self.epoch = analysis_epoch
        self.rand_seed = rand_seed
        self.derivatives_dir = derivatives_dir
        self.n_conv = len(config.model.conv_n_features)
        self.n_layers = self.n_conv + len(config.model.fc_n_units)
        self.filter_dead_units = filter_dead_units

        if model_type == "vam":
            self.model_dir = os.path.join(
                models_dir,
                "vam_models",
            )
            self.user_derivs_dir = os.path.join(
                models_dir, self.derivatives_dir, "vam", f"user{user_id}"
            )
            self.split_dir = None
            self.data_save_dir = None
        elif model_type == "task_opt":
            self.model_dir = os.path.join(
                models_dir,
                "task_opt_models",
            )
            self.user_derivs_dir = os.path.join(
                models_dir, self.derivatives_dir, "task_opt", f"user{user_id}"
            )
            # Use same dataset splits as VAM model
            self.split_dir = os.path.join(
                models_dir, "vam_models", f"user{user_id}", "splits"
            )
            self.data_save_dir = None
        elif model_type == "binned_rt":
            self.model_dir = os.path.join(
                models_dir,
                "binned_rt_models",
            )
            self.user_derivs_dir = os.path.join(
                models_dir,
                self.derivatives_dir,
                "binned_rt",
                self.config.expt_name,
            )
            self.split_dir = None
            self.data_save_dir = os.path.join(
                self.model_dir,
                self.config.expt_name,
                "data",
            )
        os.makedirs(self.user_derivs_dir, exist_ok=True)

        self.mask_path = os.path.join(self.user_derivs_dir, "mask_info.pkl")
        self.lba_params_path = os.path.join(self.user_derivs_dir, "lba_params.pkl")
        self.user_data_path = os.path.join(self.user_derivs_dir, "user_data.csv")
        self.model_outputs_path = os.path.join(
            self.user_derivs_dir, "model_outputs.pkl"
        )
        self.tmp_dir = os.path.join(self.user_derivs_dir, "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.features_path = os.path.join(self.user_derivs_dir, "feature_stats.pkl")
        self.behavior_path = os.path.join(self.user_derivs_dir, "trial_df.csv")

    def get_mask_info(self, outputs):
        # Find trials with all negative drift rates
        rts = outputs["sim_data"]["rts"]
        valid_mask = outputs["sim_data"]["valid_idx"].astype(bool)
        info = {
            "n_invalid": len(valid_mask) - np.sum(valid_mask),
            "mask": valid_mask,
        }
        with open(self.mask_path, "wb") as f:
            pickle.dump(info, f)
        return info

    def get_model_outputs(self):
        trainer = Trainer(
            self.config,
            self.model_dir,
            self.model_inputs_dir,
            reload_epoch=self.epoch,
            split_dir=self.split_dir,
            data_save_dir=self.data_save_dir,
        )

        if self.model_type in ["vam", "binned_rt"]:
            _ = get_vam_lba_params(trainer.state, save_path=self.lba_params_path)

        model_outputs = self._get_model_outputs(trainer)

        user_data = trainer.task_data.split_data["test"]
        if self.model_type == "binned_rt":
            del user_data["imgs"]
        user_data = pd.DataFrame(user_data)

        if self.model_type in ["vam", "binned_rt"]:
            mask = self.get_mask_info(model_outputs)
        elif self.model_type == "task_opt":
            mask = {"mask": np.ones(len(user_data), dtype=bool)}
            with open(self.mask_path, "wb") as f:
                pickle.dump(mask, f)

        user_data.to_csv(self.user_data_path, index=False)
        with open(self.model_outputs_path, "wb") as f:
            pickle.dump(model_outputs, f)

    def _get_model_outputs(self, trainer):
        trainer.root_key, data_key = random.split(trainer.root_key)
        test_gen = trainer.task_data.data_generator(data_key, "test", shuffle=False)

        batch_idx = 0
        sim_data = []
        drifts = []
        for batch in test_gen:
            if self.model_type in ["vam", "binned_rt"]:
                (
                    trainer.state,
                    trainer.root_key,
                    batch_sim_data,
                    batch_drifts,
                    batch_features,
                ) = vam_test_step(
                    trainer.state,
                    trainer.cnn,
                    batch,
                    trainer.root_key,
                    trainer.config.model.n_acc,
                )
                drifts.append(batch_drifts)

            elif self.model_type == "task_opt":
                (
                    trainer.state,
                    batch_sim_data,  # just the choices of the classifier
                    batch_features,
                ) = task_opt_test_step(
                    trainer.state,
                    batch,
                )

            for layer in range(self.n_layers):
                fts = batch_features[layer]
                if layer < self.n_conv:
                    max_fts = np.array(np.max(fts, axis=(1, 2)))
                else:
                    max_fts = np.array(fts)

                save_fn = f"layer{layer}_batch{batch_idx}_fts.hdf5"
                h5f = h5py.File(os.path.join(self.tmp_dir, save_fn), "w")
                h5f.create_dataset("data", data=max_fts, dtype="f8")
                h5f.close()

            batch_idx += 1
            sim_data.append(batch_sim_data)

        sim_data = {
            k: np.concatenate([x[k] for x in sim_data], axis=0)
            for k in sim_data[0].keys()
        }

        if self.model_type in ["vam", "binned_rt"]:
            drifts = np.concatenate(drifts, axis=0)
            model_outputs = {
                "sim_data": sim_data,
                "drifts": drifts,
            }
        elif self.model_type == "task_opt":
            model_outputs = {"sim_data": sim_data}

        return model_outputs

    def process_model_outputs(self):
        user_data = pd.read_csv(self.user_data_path)
        with open(self.model_outputs_path, "rb") as f:
            model_outputs = pickle.load(f)
        with open(self.mask_path, "rb") as f:
            mask_info = pickle.load(f)

        n_trials = len(model_outputs["sim_data"]["response_dirs"])
        bs = self.config.training.batch_size
        n_batches = int(np.ceil(n_trials / bs))

        if self.model_type in ["vam", "binned_rt"]:
            self.process_vam_behavior(
                mask_info,
                user_data,
                model_outputs,
            )
        elif self.model_type == "task_opt":
            self.process_task_opt_behavior(
                user_data,
                model_outputs,
            )

        ft_stats = self.process_features(
            mask_info,
            n_batches,
        )

        with open(self.features_path, "wb") as f:
            pickle.dump(ft_stats, f)

        # Clean up
        self._clean_up()

    def _clean_up(self):
        shutil.rmtree(self.tmp_dir)
        os.remove(self.user_data_path)
        os.remove(self.model_outputs_path)

    def _load_layer_fts(self, layer, n_batches):
        for i in range(n_batches):
            batch_fn = os.path.join(self.tmp_dir, f"layer{layer}_batch{i}_fts.hdf5")
            h5f = h5py.File(batch_fn, "r")
            if i == 0:
                x = h5f["data"]
            else:
                x = np.concatenate([x, h5f["data"]], axis=0)
        return x

    def process_features(self, mask_info, n_batches):
        mask = mask_info["mask"]
        all_stats = {f"layer{layer}": {} for layer in range(self.n_layers)}

        for layer in range(self.n_layers):
            layer_stats = {}
            fts = self._load_layer_fts(layer, n_batches)
            fts = fts[mask]

            if self.filter_dead_units:
                n_orig = fts.shape[1]
                alive_idx = np.where(fts.max(axis=0) > 0)[0]
                fts = fts[:, fts.max(axis=0) > 0]
                n_alive = fts.shape[1]
                layer_stats["n_dead_units"] = n_orig - n_alive
                layer_stats["n_alive_units"] = n_alive
                layer_stats["n_total_units"] = n_orig
                layer_stats["f_dead_units"] = (n_orig - n_alive) / n_orig

            layer_stats["features"] = fts
            all_stats[f"layer{layer}"] = layer_stats

        return all_stats

    def process_vam_behavior(self, mask_info, user_data, model_outputs):
        # Concatenate user, model data (in that order)
        mask = mask_info["mask"]
        model_rts = model_outputs["sim_data"]["rts"][mask]
        model_responses = model_outputs["sim_data"]["response_dirs"][mask]
        drifts = model_outputs["drifts"][mask]
        target_drifts, flanker_drifts, other_drifts = self._process_drifts(
            drifts,
            user_data["targ_dirs"].values[mask],
            user_data["dis_dirs"].values[mask],
            self.config.model.n_acc,
        )

        model_data = user_data.copy(deep=True)[mask]
        model_data["model_user"] = "model"
        model_data["rts"] = model_rts
        model_data["response_dirs"] = model_responses
        model_data["target_drifts"] = target_drifts
        model_data["flanker_drifts"] = flanker_drifts
        model_data["other_drifts"] = other_drifts

        user_data["model_user"] = "user"
        df = pd.concat((user_data, model_data)).reset_index(drop=True)
        df.to_csv(self.behavior_path, index=False)

    def process_task_opt_behavior(self, user_data, model_outputs):
        # Concatenate user, model data (in that order)
        model_responses = model_outputs["sim_data"]["response_dirs"]

        model_data = user_data.copy(deep=True)
        model_data["model_user"] = "model"
        model_data["response_dirs"] = model_responses
        model_data["rts"] = np.nan

        user_data["model_user"] = "user"
        df = pd.concat((user_data, model_data)).reset_index(drop=True)
        df.to_csv(self.behavior_path, index=False)

    def _process_drifts(self, drifts, targets, flankers, n_acc):
        # Get mean drift for targets, flankers, and non-target/non-flanker (other)
        # Congruent trials: other drift rates calculated as the average
        # of two non-target/non-flanker drift rates
        # Incongruent trials: other drift rates calculated as the average
        # of the three non-target drift rates
        con_idx = np.where(targets == flankers)[0]
        incon_idx = np.where(targets != flankers)[0]
        target_drifts = drifts[np.arange(len(targets)), targets]
        flanker_drifts = drifts[np.arange(len(flankers)), flankers]
        other_drifts = np.zeros(len(targets))
        other_drifts[con_idx] = (
            np.sum(drifts[con_idx], axis=1) - target_drifts[con_idx]
        ) / (n_acc - 1)
        other_drifts[incon_idx] = (
            np.sum(drifts[incon_idx], axis=1)
            - target_drifts[incon_idx]
            - flanker_drifts[incon_idx]
        ) / (n_acc - 2)
        return target_drifts, flanker_drifts, other_drifts

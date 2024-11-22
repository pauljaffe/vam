import pdb
import pickle
import os
import itertools
import time
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mutual_info_score
from scipy.stats import (
    pearsonr,
    f_oneway,
    chi2,
    zscore,
    ranksums,
    PermutationMethod,
    BootstrapMethod,
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd


class BasicAnalysisMixin:
    """
    Functions for calculating mean RT, accuracy, congruency effects, and
    the number of dead units in each CNN layer. Also provides some commonly used
    operations for reformatting the data.
    """

    def remap_vars(self, df=None):
        # Remap direction columns, change column and value names,
        # facilitates plotting
        direction_map = {0: "Left", 1: "Right", 2: "Up", 3: "Down"}
        cols = ["targ_dirs", "dis_dirs", "response_dirs"]
        for c in cols:
            if df is not None:
                df[c] = df[c].replace(direction_map)
            else:
                self.vam_df[f"{c}_num"] = self.vam_df[c]
                self.vam_df[c] = self.vam_df[c].replace(direction_map)
                self.task_opt_df[f"{c}_num"] = self.task_opt_df[c]
                self.task_opt_df[c] = self.task_opt_df[c].replace(direction_map)

        # Remap layouts
        layout_map = {0: "---", 1: "|", 2: "+", 3: "<", 4: ">", 5: "v", 6: "^"}
        if df is not None:
            df["layouts"] = df["layouts"].replace(layout_map)
        else:
            self.vam_df["layouts_num"] = self.vam_df["layouts"]
            self.vam_df["layouts"] = self.vam_df["layouts"].replace(layout_map)
            self.task_opt_df["layouts_num"] = self.task_opt_df["layouts"]
            self.task_opt_df["layouts"] = self.task_opt_df["layouts"].replace(
                layout_map
            )

        return df

    def get_layers(self, config):
        layers = []
        conv_idx = 0
        fc_idx = 0
        for i in range(len(config.model.conv_n_features)):
            layers.append(f"Conv{conv_idx+1}")
            conv_idx += 1
        for i in range(len(config.model.fc_n_units)):
            layers.append(f"FC{fc_idx+1}")
            fc_idx += 1
        return layers

    def basic_stats(self):
        stat_dfs = []
        for mtype, trial_df, model_fts in zip(
            ["vam", "task_opt"],
            [self.vam_df, self.task_opt_df],
            [self.vam_fts, self.task_opt_fts],
        ):
            acc_rt_stats = self._get_acc_rt_stats(trial_df, mtype)
            congruency_stats = self._get_congruency_stats(trial_df, mtype)
            dead_unit_stats = self._get_dead_unit_stats(model_fts, mtype)

            stat_dfs.append(
                pd.concat([acc_rt_stats, congruency_stats, dead_unit_stats])
            )

        return pd.concat(stat_dfs).reset_index(drop=True)

    def _get_acc_rt_stats(self, df, model_type):
        # Get invalid trial count (all drift rates negative)
        if model_type == "task_opt":
            frac_invalid = np.nan
        else:
            n_invalid = self.mask_info["n_invalid"]
            n_total = len(self.mask_info["mask"])
            frac_invalid = n_invalid / n_total

        # Accuracy
        model_acc = df.query("model_user == 'model'")["correct"].mean()
        user_acc = df.query("model_user == 'user'")["correct"].mean()

        # RTs
        model_mean_rt = df.query("model_user == 'model' and correct == 1")["rts"].mean()
        user_mean_rt = df.query("model_user == 'user' and correct == 1")["rts"].mean()

        stats = pd.DataFrame(
            {
                "user_id": 5 * [self.user_id],
                "stat": ["mean_rt", "mean_rt", "accuracy", "accuracy", "frac_invalid"],
                "value": [
                    user_mean_rt,
                    model_mean_rt,
                    user_acc,
                    model_acc,
                    frac_invalid,
                ],
                "model_user": ["user", "model", "user", "model", "model"],
                "model_type": 5 * [model_type],
                "layer": 5 * [np.nan],
            }
        )

        return stats

    def _get_congruency_stats(self, df, model_type):
        stat_dfs = []
        for mu in ["model", "user"]:
            mu_df = df.query("model_user == @mu")

            # Accuracy
            con_df = mu_df.query("congruency == 0")
            incon_df = mu_df.query("congruency == 1")
            con_acc = con_df["correct"].mean()
            incon_acc = incon_df["correct"].mean()
            acc_con_effect = con_acc - incon_acc

            # RTs
            con_correct_df = con_df.query("correct == 1")
            incon_correct_df = incon_df.query("correct == 1")
            con_rt = con_correct_df["rts"].mean()
            incon_rt = incon_correct_df["rts"].mean()
            rt_con_effect = incon_rt - con_rt

            mu_stats = pd.DataFrame(
                {
                    "model_user": 6 * [mu],
                    "user_id": 6 * [self.user_id],
                    "model_type": 6 * [model_type],
                    "stat": [
                        "acc_con_effect",
                        "rt_con_effect",
                        "con_rt",
                        "incon_rt",
                        "con_acc",
                        "incon_acc",
                    ],
                    "value": [
                        acc_con_effect,
                        rt_con_effect,
                        con_rt,
                        incon_rt,
                        con_acc,
                        incon_acc,
                    ],
                    "layer": 6 * [np.nan],
                }
            )
            stat_dfs.append(mu_stats)

        return pd.concat(stat_dfs)

    def _get_dead_unit_stats(self, layer_info, model_type):
        all_stats = []
        for layer_idx, layer in enumerate(self.layers):
            n_dead = layer_info[f"layer{layer_idx}"]["n_dead_units"]
            n_total = layer_info[f"layer{layer_idx}"]["n_total_units"]
            f_dead = n_dead / n_total

            layer_stats = pd.DataFrame(
                {
                    "user_id": 3 * [self.user_id],
                    "stat": ["n_dead_units", "n_total_units", "f_dead_units"],
                    "value": [n_dead, n_total, f_dead],
                    "model_user": ["model", "model", "model"],
                    "model_type": 3 * [model_type],
                    "layer": 3 * [layer],
                }
            )
            all_stats.append(layer_stats)

        return pd.concat(all_stats)

    def add_correct(self, df=None):
        def _get_correct(targ, response):
            if targ == response:
                return 1
            else:
                return 0

        if df is not None:
            df["correct"] = df.apply(
                lambda x: _get_correct(x.targ_dirs, x.response_dirs), axis=1
            )
        else:
            self.vam_df["correct"] = self.vam_df.apply(
                lambda x: _get_correct(x.targ_dirs, x.response_dirs), axis=1
            )
            self.task_opt_df["correct"] = self.task_opt_df.apply(
                lambda x: _get_correct(x.targ_dirs, x.response_dirs), axis=1
            )
            if self.has_binned_rt_model:
                self.binned_rt_df["correct"] = self.binned_rt_df.apply(
                    lambda x: _get_correct(x.targ_dirs, x.response_dirs), axis=1
                )

        return df

    def _get_incon_idx(self, trial_df):
        # Filter for incongruent trials from model outputs
        mdf = trial_df.query("model_user == 'model'").reset_index(drop=True)
        incon_df = mdf.query("congruency == 1")
        incon_idx = incon_df.index.to_numpy()
        return incon_df.reset_index(drop=True), incon_idx


class LBAParamsMixin:
    """
    Functions for extracting the LBA params from the VAM and calculating
    the drift rates. Also calculates stats for the output logits of
    the task-optimized models.
    """

    def lba_param_analysis(self):
        lba_dfs = []
        for mtype, trial_df in zip(
            ["vam", "task_opt"],
            [self.vam_df, self.task_opt_df],
        ):
            if mtype == "vam":
                with open(os.path.join(self.vam_dir, "lba_params.pkl"), "rb") as f:
                    vam_lba_params = pickle.load(f)
                lba_df = self._lba_param_stats(trial_df, vam_lba_params, mtype)
            elif mtype == "task_opt":
                lba_df = self._drift_logit_stats(trial_df, mtype)
            lba_dfs.append(lba_df)

        return pd.concat(lba_dfs)

    def _drift_logit_stats(self, trial_df, model_type, rt_bin=None):
        if model_type == "vam":
            label = "drifts"
        elif model_type == "task_opt":
            label = "logits"

        model_df = trial_df.query("model_user == 'model'")
        stat_dfs = []
        for con in [0, 1]:
            con_df = model_df.query("congruency == @con")
            targ_drift = con_df[f"target_{label}"].mean()
            flnk_drift = con_df[f"flanker_{label}"].mean()
            other_drift = con_df[f"other_{label}"].mean()

            this_df = pd.DataFrame(
                {
                    "stat": [
                        f"target_{label[:-1]}",
                        f"flanker_{label[:-1]}",
                        f"other_{label[:-1]}",
                    ],
                    "congruency": 3 * [con],
                    "value": [targ_drift, flnk_drift, other_drift],
                    "user_id": 3 * [self.user_id],
                    "model_type": 3 * [model_type],
                    "rt_bin": 3 * [rt_bin],
                }
            )
            stat_dfs.append(this_df)
        return pd.concat(stat_dfs).reset_index(drop=True)

    def _lba_param_stats(self, trial_df, lba_params, model_type, rt_bin=None):
        # Get drift rates for congruent/incongruent trials
        model_df = trial_df.query("model_user == 'model'")
        all_dfs = []
        all_dfs.append(self._drift_logit_stats(trial_df, model_type, rt_bin=rt_bin))

        # Other LBA params
        response_caution = lba_params["b"][0] - lba_params["a"][0]
        lba_df = pd.DataFrame(
            {
                "stat": ["a", "b", "t0", "response_caution"],
                "value": [
                    lba_params["a"][0],
                    lba_params["b"][0],
                    lba_params["t0"][0],
                    response_caution,
                ],
                "congruency": 4 * [np.nan],
                "user_id": 4 * [self.user_id],
                "model_type": 4 * [model_type],
                "rt_bin": 4 * [rt_bin],
            }
        )
        all_dfs.append(lba_df)

        return pd.concat(all_dfs).reset_index(drop=True)


class SingleUnitMixin:
    """
    Functions for calculating direction selectivity of single units
    in the CNN and mutual info conveyed by single units about stimulus features.
    """

    def mutual_info_analysis(self, n_bins=10):
        model_fts = [self.vam_fts, self.task_opt_fts]
        model_dfs = [self.vam_df, self.task_opt_df]
        model_types = ["vam", "task_opt"]
        decoded_vars = [
            "targ_dirs_num",
            "dis_dirs_num",
            "layouts_num",
            "xpos_bin_repr",
            "ypos_bin_repr",
        ]
        all_mi_dfs = []

        for x, model_df, model_type in zip(model_fts, model_dfs, model_types):
            for layer_idx, layer in enumerate(self.layers):
                for dv in decoded_vars:
                    this_mdf, incon_idx = self._get_incon_idx(model_df)
                    layer_act = x[f"layer{layer_idx}"]["features"][incon_idx]
                    this_dv = this_mdf[dv]

                    this_mi = self._calc_condition_mi(
                        layer_act,
                        this_dv,
                        n_bins,
                    )
                    mi_mean = np.mean(this_mi)
                    mi_std = np.std(this_mi)

                    mi_stats = {
                        "user_id": 2 * [self.user_id],
                        "layer": 2 * [layer],
                        "mi_variable": 2 * [dv],
                        "model_type": 2 * [model_type],
                        "stat": ["mi_mean", "mi_std"],
                        "value": [mi_mean, mi_std],
                    }

                    all_mi_dfs.append(pd.DataFrame(mi_stats))

        stat_df = pd.concat(all_mi_dfs)
        stat_df.reset_index(drop=True, inplace=True)
        return stat_df

    def _calc_condition_mi(self, x, y, nxbins):
        n_units = x.shape[1]
        all_norm_mi = np.zeros(n_units)

        # Calculate entropy of feature distribution
        y_hist = y.value_counts(normalize=True).values
        y_entropy = -np.sum(y_hist * np.log(y_hist))
        nybins = len(y_hist)

        for unit_idx in range(n_units):
            xy_hist = np.histogram2d(x[:, unit_idx], y.values, bins=[nxbins, nybins])[0]
            # Calculate MI in nats (natural log)
            mi = mutual_info_score(None, None, contingency=xy_hist)
            all_norm_mi[unit_idx] = mi / y_entropy

        return all_norm_mi

    def unit_modulation_analysis(self, alpha=0.001, n_heatmap_trials=100):
        unit_df, activation_df = self._get_all_unit_modulation(
            alpha=alpha, n_heatmap_trials=n_heatmap_trials
        )

        summary_dfs = []
        model_fts = [self.vam_fts, self.task_opt_fts]
        model_dfs = [self.vam_df, self.task_opt_df]
        model_types = ["vam", "task_opt"]

        for x, model_df, model_type in zip(model_fts, model_dfs, model_types):
            for layer_idx, layer in enumerate(self.layers):
                this_unit_df = unit_df.query(
                    f"layer == @layer and model_type == @model_type"
                ).reset_index(drop=True)
                summary_df = self._get_modulation_summary(
                    this_unit_df, layer, model_type
                )
                summary_dfs.append(summary_df)

        summary_df = pd.concat(summary_dfs).reset_index(drop=True)

        return summary_df, activation_df

    def _get_all_unit_modulation(self, alpha=0.001, n_heatmap_trials=100):
        model_fts = [self.vam_fts, self.task_opt_fts]
        model_dfs = [self.vam_df, self.task_opt_df]
        model_types = ["vam", "task_opt"]
        stimulus_vars = ["targ_dirs_num"]
        unit_dfs = []
        activation_dfs = []

        for x, model_df, model_type in zip(model_fts, model_dfs, model_types):
            for layer_idx, layer in enumerate(self.layers):
                this_mdf, incon_idx = self._get_incon_idx(model_df)
                layer_act = x[f"layer{layer_idx}"]["features"][incon_idx]

                for sv in stimulus_vars:
                    stim_vals = this_mdf[sv].values

                    for uidx in np.arange(layer_act.shape[1]):
                        (
                            p,
                            is_signif,
                            mod_type,
                            mod_dir,
                        ) = self._get_single_unit_modulation(
                            layer_act[:, uidx],
                            stim_vals,
                            alpha,
                        )

                        if model_type == "vam" and sv == "targ_dirs_num":
                            # For plotting activations
                            activation_df = self._get_activation_df(
                                layer_act, uidx, n_heatmap_trials, layer, stim_vals
                            )
                            activation_df["modulation_type"] = mod_type
                            activation_df["modulation_dir"] = mod_dir
                            activation_dfs.append(activation_df)

                        unit_stats = pd.DataFrame(
                            {
                                "anova_p": [p],
                                "anova_signif": [is_signif],
                                "modulation_type": [mod_type],
                                "modulation_dir": [mod_dir],
                                "stimulus_feature": [sv],
                                "unit_idx": [uidx],
                                "layer": [layer],
                                "model_type": [model_type],
                                "user_id": [self.user_id],
                            },
                            index=[0],
                        )
                        unit_dfs.append(unit_stats)

        return (
            pd.concat(unit_dfs).reset_index(drop=True),
            pd.concat(activation_dfs).reset_index(drop=True),
        )

    def _get_activation_df(self, x, uidx, n_trials, layer, targ_dirs):
        rng = np.random.default_rng(seed=self.rand_seed)  # same trials for all units
        rand_idx = rng.choice(len(targ_dirs), n_trials, replace=False)
        x = x[rand_idx, uidx]
        df = pd.DataFrame(
            {
                "x": x,
                "targ_dir": targ_dirs[rand_idx],
                "layer": len(rand_idx) * [layer],
                "unit_idx": len(rand_idx) * [uidx],
            }
        ).sort_values(by=["targ_dir"])
        df["stim_idx"] = np.arange(len(rand_idx))
        return df.reset_index(drop=True)

    def _get_single_unit_modulation(self, x, stim_vals, alpha):
        x = zscore(x)
        this_df = pd.DataFrame({"x": x, "stim_val": stim_vals})
        this_anova = (
            this_df.groupby("stim_val")["x"].apply(list).reset_index()["x"].values
        )
        anova_p = f_oneway(*this_anova).pvalue

        # Post-hoc comparisons
        if anova_p < alpha:
            anova_signif = True
            pairs_sign = np.zeros((4, 4))
            tukey_res = pairwise_tukeyhsd(x, stim_vals)
            pair_idx = tukey_res._multicomp.pairindices
            n_pairs = len(pair_idx[0])

            for stat_idx, d1, d2 in zip(range(n_pairs), pair_idx[0], pair_idx[1]):
                if tukey_res.reject[stat_idx]:
                    # Get sign of difference
                    # tukeyhsd mean diff is group2 - group1
                    pairs_sign[d1, d2] = -np.sign(tukey_res.meandiffs[stat_idx])
                    pairs_sign[d2, d1] = -pairs_sign[d1, d2]

            # Get selectivity type and direction
            n_pos = np.sum(np.where(pairs_sign == 1, 1, 0), axis=1)
            n_neg = np.sum(np.where(pairs_sign == -1, 1, 0), axis=1)
            selective_pos_idx = np.where(n_pos == 3)[0]
            selective_neg_idx = np.where(n_neg == 3)[0]

            if (len(selective_pos_idx) == 1) and (len(selective_neg_idx) == 0):
                modulation_type = "p"  # selective w/ positive modulation
                modulation_dir = str(selective_pos_idx[0])
            elif (len(selective_neg_idx) == 1) and (len(selective_pos_idx) == 0):
                modulation_type = "n"  # selective w/ positive modulation
                modulation_dir = str(selective_neg_idx[0])
            elif (len(selective_pos_idx) == 1) and (len(selective_neg_idx) == 1):
                # Check if magnitude is significantly greater for pos or neg
                pos_vals = np.abs(this_anova[selective_pos_idx[0]])
                neg_vals = np.abs(this_anova[selective_neg_idx[0]])
                _, rs_p = ranksums(pos_vals, neg_vals)
                if rs_p < 0.05:
                    if np.mean(pos_vals) > np.mean(neg_vals):
                        modulation_type = "p"
                        modulation_dir = str(selective_pos_idx[0])
                    else:
                        modulation_type = "n"
                        modulation_dir = str(selective_neg_idx[0])
                else:
                    modulation_type = "c"
                    modulation_dir = np.nan
            else:
                # Complex modulation
                modulation_type = "c"
                modulation_dir = np.nan
        else:
            anova_signif = False
            modulation_type = np.nan
            modulation_dir = np.nan

        return anova_p, anova_signif, modulation_type, modulation_dir

    def _get_modulation_summary(self, unit_df, layer, model_type):
        all_dfs = []
        n_units = len(unit_df["unit_idx"].unique())
        for stim_ft in ["targ_dirs_num", "dis_dirs_num"]:
            this_df = unit_df.query("stimulus_feature == @stim_ft")
            n_signif = np.sum(this_df["anova_signif"])
            f_signif = n_signif / n_units
            f_not_signif = 1 - f_signif
            n_pos = np.sum(this_df["modulation_type"] == "p")
            n_neg = np.sum(this_df["modulation_type"] == "n")
            n_complex = np.sum(this_df["modulation_type"] == "c")
            f_pos = n_pos / n_units
            f_neg = n_neg / n_units
            f_complex = n_complex / n_units
            vals = [f_signif, f_not_signif, f_pos, f_neg, f_complex, n_units]
            stat_names = [
                "f_signif",
                "f_not_signif",
                "f_pos",
                "f_neg",
                "f_complex",
                "n_alive",
            ]

            all_dfs.append(
                pd.DataFrame(
                    {
                        "stat": stat_names,
                        "value": vals,
                        "layer": len(vals) * [layer],
                        "user_id": len(vals) * [self.user_id],
                        "model_type": len(vals) * [model_type],
                        "stimulus_feature": len(vals) * [stim_ft],
                    }
                )
            )
        return pd.concat(all_dfs)

    def plot_activation_heatmap(
        self,
        df,
        layers,
        n_rand=None,
        axes=None,
        modulation=["p"],
    ):
        if axes is None:
            _, axes = plt.subplots(1, len(layers), figsize=(7, 3.2))
        df = df.dropna(subset=["modulation_type"])
        if modulation is not None:
            df = df.query("modulation_type in @modulation")
        df["modulation_type"] = df["modulation_type"].apply(
            lambda x: {"p": 0, "n": 1, "c": 2}[x]
        )
        cbar_ax = axes[-1]
        for ax, layer in zip(axes, layers):
            plot_df = df.query("layer == @layer")
            if ax == axes[-2]:
                plot_cbar = True
            else:
                plot_cbar = False
            if ax == axes[0]:
                ylab = "Unit (sorted by\ndirection preference)"
            else:
                ylab = ""
            if ax == axes[3]:
                xlab = "Stimulus (sorted by target direction)"
            else:
                xlab = ""

            if n_rand is not None:
                unit_idx = plot_df["unit_idx"].unique()
                rng = np.random.default_rng(seed=self.rand_seed)
                rand_idx = rng.choice(unit_idx, size=n_rand, replace=False)
                plot_df = plot_df.query("unit_idx in @rand_idx")

            plot_df = plot_df.sort_values(by=["modulation_type", "modulation_dir"])
            plot_df = plot_df[["x", "stim_idx", "unit_idx"]].reset_index(drop=True)
            plot_df = plot_df.pivot_table(
                index="unit_idx", columns="stim_idx", values="x", sort=False
            )

            # Remove units without modulation for these stimuli
            # Note such units must still be modulated by other stimuli in the test set,
            # in order to be included
            plot_df = plot_df.drop(plot_df[plot_df.max(axis=1) == 0].index)
            plot_df = plot_df.apply(self._normalize_center_unit, axis=1)

            sns.heatmap(
                plot_df,
                ax=ax,
                cmap="viridis",
                center=0.0,
                cbar=plot_cbar,
                cbar_kws={"label": "Norm. activation"},
                cbar_ax=cbar_ax,
            )
            ax.set(xticks=[], yticks=[])
            ax.set_xlabel(xlab, loc="right", fontsize=6)
            ax.set_ylabel(ylab, fontsize=6)
            ax.set_title(layer, fontsize=6, pad=2)

        return axes

    def _normalize_center_unit(self, x):
        # Normalize unit activations by dividing all activations by max(abs(x))
        x = x - np.nanmean(x)
        xmax = np.nanmax(np.abs(x))
        return x / xmax


class DeltaPlotCAFMixin:
    """
    Functions for analyzing how RT congruency effects change
    with mean RT (delta plots) and how accuracy on incongruent/congruent
    trials changes with mean RT (conditional accuracy functions = CAFs).
    """

    def delta_caf_analysis(self):
        if self.has_binned_rt_model:
            model_dfs = [self.vam_df, self.binned_rt_df]
            model_types = ["vam", "binned_rt"]
        else:
            model_dfs = [self.vam_df]
            model_types = ["vam"]
        delta_dfs = []
        caf_dfs = []

        for model_df, model_type in zip(model_dfs, model_types):
            for mu in ["model", "user"]:
                mu_df = model_df.query("model_user == @mu")
                mu_correct_df = model_df.query("model_user == @mu and correct == 1")
                mu_delta_df = self._get_delta(mu_correct_df, mu, model_type)
                mu_caf_df = self._get_caf(mu_df, mu, model_type)
                delta_dfs.append(mu_delta_df)
                caf_dfs.append(mu_caf_df)

        return pd.concat(delta_dfs).reset_index(drop=True), pd.concat(
            caf_dfs
        ).reset_index(drop=True)

    def _get_delta(self, df, model_user, model_type, deciles=np.arange(0.1, 1.0, 0.1)):
        con_df = df.query("congruency == 0")
        inc_df = df.query("congruency == 1")
        con_deciles = np.quantile(con_df["rts"], deciles)
        inc_deciles = np.quantile(inc_df["rts"], deciles)
        delta = inc_deciles - con_deciles
        avg_rts = (inc_deciles + con_deciles) / 2
        delta_df = pd.DataFrame(
            {
                "decile_idx": deciles,
                "con_decile": con_deciles,
                "incon_decile": inc_deciles,
                "delta": delta,
                "avg_rt": avg_rts,
                "model_user": len(deciles) * [model_user],
                "model_type": len(deciles) * [model_type],
                "user_id": len(deciles) * [self.user_id],
            }
        )
        return delta_df

    def plot_delta(self, df, x_var, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        sns.lineplot(
            data=df, x=x_var, y="delta", hue="model_user", ax=ax, err_style="bars"
        )
        ax.set_xlabel("Mean RT (s)")
        ax.set_ylabel("RT congruency effect (s)")
        return ax

    def caf_analysis(self):
        model_df = self.vam_df.query("model_user == 'model'")
        user_df = self.vam_df.query("model_user == 'user'")
        model_caf = self._get_caf(model_df, "model")
        user_caf = self._get_caf(user_df, "user")
        return pd.concat([model_caf, user_caf])

    def _get_caf(self, df, model_user, model_type, quantiles=5):
        con_df = df.query("congruency == 0")
        inc_df = df.query("congruency == 1")
        con_df["rt_decile"] = pd.qcut(con_df["rts"], quantiles, labels=False)
        inc_df["rt_decile"] = pd.qcut(inc_df["rts"], quantiles, labels=False)
        con_acc = con_df.groupby("rt_decile")["correct"].mean().reset_index()
        inc_acc = inc_df.groupby("rt_decile")["correct"].mean().reset_index()
        con_rts = con_df.groupby("rt_decile")["rts"].mean().reset_index()
        inc_rts = inc_df.groupby("rt_decile")["rts"].mean().reset_index()
        con_df = pd.DataFrame(
            {
                "acc": con_acc["correct"],
                "rt": con_rts["rts"],
                "decile_idx": con_acc["rt_decile"],
                "model_user": len(con_acc) * [model_user],
                "congruency": len(con_acc) * ["congruent"],
                "model_type": len(con_acc) * [model_type],
                "user_id": len(con_acc) * [self.user_id],
            }
        )
        inc_df = pd.DataFrame(
            {
                "acc": inc_acc["correct"],
                "rt": inc_rts["rts"],
                "decile_idx": inc_acc["rt_decile"],
                "model_user": len(inc_acc) * [model_user],
                "congruency": len(inc_acc) * ["incongruent"],
                "model_type": len(inc_acc) * [model_type],
                "user_id": len(inc_acc) * [self.user_id],
            }
        )
        return pd.concat([con_df, inc_df])

    def plot_caf(self, df, x_var, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 4))
        sns.lineplot(
            data=df,
            x=x_var,
            y="acc",
            style="congruency",
            hue="model_user",
            err_style="bars",
            ax=ax,
        )
        ax.set_xlabel("Mean RT (s)")
        ax.set_ylabel("Accuracy")
        return ax


class LayoutPositionMixin:
    """
    Functions for analyzing how RT / accuracy changes with stimulus layout and
    stimulus position (binned, both horizontal and vertical).
    """

    def layout_position_analysis(self):
        all_summary_dfs = []
        all_mean_dfs = []
        for stim_feature in ["layouts", "xpos_bin_behavior", "ypos_bin_behavior"]:
            summary_df, mean_df = self._layout_position_analysis(stim_feature)
            all_summary_dfs.append(summary_df)
            all_mean_dfs.append(mean_df)
        summary_df = pd.concat(all_summary_dfs).reset_index(drop=True)
        mean_df = pd.concat(all_mean_dfs).reset_index(drop=True)
        return summary_df, mean_df

    def _layout_position_analysis(self, feature):
        all_mean_dfs = []
        for mu in ["model", "user"]:
            mean_df = self._layout_pos_means(mu, feature, self.vam_df)
            all_mean_dfs.append(mean_df)
        mean_df = pd.concat(all_mean_dfs)
        summary_df = self._layout_pos_summary(mean_df, feature)
        return summary_df, mean_df

    def _layout_pos_means(self, model_user, feature, df):
        df2 = df.query("model_user == @model_user")
        df2_correct = df2.query("correct == 1")

        mean_rts = df2_correct.groupby([feature])["rts"].mean()
        mean_correct = df2.groupby([feature])["correct"].mean()

        # Center mean RTs / accuracy so that users can be averaged
        mean_rts_centered = mean_rts - mean_rts.mean()
        mean_correct_centered = mean_correct - mean_correct.mean()

        n_vals = len(df2[feature].unique())

        stats_raw = pd.DataFrame(
            {
                "model_user": 2 * n_vals * [model_user],
                "user_id": 2 * n_vals * [self.user_id],
                "stimulus_feature": 2 * n_vals * [feature],
                "feature_value": [
                    *mean_rts.index,
                    *mean_correct.index,
                ],
                "stat": [
                    *n_vals * ["mean_rt"],
                    *n_vals * ["mean_correct"],
                ],
                "value": [
                    *mean_rts,
                    *mean_correct,
                ],
            }
        )
        stats_centered = pd.DataFrame(
            {
                "model_user": 2 * n_vals * [model_user],
                "user_id": 2 * n_vals * [self.user_id],
                "stimulus_feature": 2 * n_vals * [feature],
                "feature_value": [
                    *mean_rts_centered.index,
                    *mean_correct_centered.index,
                ],
                "stat": [
                    *n_vals * ["mean_rt_centered"],
                    *n_vals * ["mean_correct_centered"],
                ],
                "value": [
                    *mean_rts_centered,
                    *mean_correct_centered,
                ],
            }
        )

        return pd.concat([stats_raw, stats_centered]).reset_index(drop=True)

    def _check_layout_pos_effect(self, stat, feature):
        # Check if effect of stimulus feature (layout/position) on
        # RT / accuracy is significant by ANOVA or chi-squared test
        feature_vals = self.vam_df[feature].unique()
        if stat == "mean_rt":
            df = self.vam_df.query("correct == 1 and model_user == 'user'")
            behavior_stat_user = (
                df.groupby(feature)["rts"].apply(list).reset_index()["rts"].values
            )
            stat_info = f_oneway(*behavior_stat_user)
            p = stat_info.pvalue
        elif stat == "mean_correct":
            df = self.vam_df.query("model_user == 'user'")
            mean_acc = df.groupby(feature)["correct"].mean().mean()
            observed_counts = df.groupby(feature)["correct"].sum().values
            total_counts = df.groupby(feature)["correct"].count().values
            expected_counts = total_counts * mean_acc
            stat = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
            p = chi2.sf(stat, len(observed_counts) - 1)
        return p

    def _layout_pos_summary(self, mean_df, feature):
        all_dfs = []
        for stat in ["mean_rt", "mean_correct"]:
            model_df = mean_df.query("stat == @stat and model_user == 'model'")
            user_df = mean_df.query("stat == @stat and model_user == 'user'")
            assert all(
                model_df["feature_value"].values == user_df["feature_value"].values
            )

            pearson_stats = pearsonr(model_df["value"].values, user_df["value"].values)

            user_p = self._check_layout_pos_effect(stat, feature)

            stat_df = pd.DataFrame(
                {
                    "user_id": [self.user_id],
                    "stat": [stat],
                    "pearsons_p": [pearson_stats.pvalue],
                    "pearsons_r": [pearson_stats.statistic],
                    "user_effect_p": [user_p],
                    "stimulus_feature": [feature],
                },
                index=[0],
            )
            all_dfs.append(stat_df)

        return pd.concat(all_dfs).reset_index(drop=True)


class DecoderMixin:
    """
    Functions for decoding stimulus features from model activations.
    Also saves the target/flanker decoder axes for calculating the
    target/flanker subspaces (SubspaceMixin). Incongruent trials only.
    """

    def decoder_analysis(self, n_bins=10):
        model_fts = [self.vam_fts, self.task_opt_fts]
        model_dfs = [self.vam_df, self.task_opt_df]
        model_types = ["vam", "task_opt"]
        decoded_vars = [
            "targ_dirs_num",
            "dis_dirs_num",
            "layouts_num",
            "xpos_bin_repr",
            "ypos_bin_repr",
        ]
        all_stat_dfs = []
        decoder_ax = {
            mtype: {layer: {} for layer in self.layers} for mtype in model_types
        }

        for x, model_df, model_type in zip(model_fts, model_dfs, model_types):
            for layer_idx, layer in enumerate(self.layers):
                for dv in decoded_vars:
                    this_mdf, incon_idx = self._get_incon_idx(model_df)
                    layer_act = x[f"layer{layer_idx}"]["features"][incon_idx]
                    layer_act_stnd = StandardScaler().fit_transform(layer_act)

                    dv_acc, dv_vecs = self._single_decoder_classifier(
                        layer_act_stnd,
                        this_mdf[dv],
                    )

                    decoder_ax[model_type][layer][dv] = dv_vecs

                    this_df = pd.DataFrame(
                        {
                            "decoder_accuracy": [dv_acc],
                            "decoded_variable": [dv],
                            "layer": [layer],
                            "user_id": [self.user_id],
                            "model_type": [model_type],
                        },
                        index=[0],
                    )
                    all_stat_dfs.append(this_df)

        stat_df = pd.concat(all_stat_dfs).reset_index(drop=True)

        return stat_df, decoder_ax

    def _single_decoder_classifier(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.33,
            random_state=self.rand_seed,
        )

        clf = LinearSVC(
            multi_class="ovr",
            fit_intercept=True,
            loss="squared_hinge",
        ).fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        weights = clf.coef_
        n_ft_vals = len(weights)
        vecs = np.stack(
            [weights[i, :] / np.linalg.norm(weights[i, :]) for i in range(n_ft_vals)],
            axis=0,
        )

        return acc, vecs


class SubspaceMixin:
    """
    Functions for calculating the principal alignment metric
    between target/flanker subspaces.
    """

    def subspace_analysis(self, decoder_ax):
        model_types = ["vam", "task_opt"]
        subspace_dfs = []

        for model_type in model_types:
            for layer_idx, layer in enumerate(self.layers):
                targ_mat = decoder_ax[model_type][layer]["targ_dirs_num"]
                flnk_mat = decoder_ax[model_type][layer]["dis_dirs_num"]

                subspace_df = self._subspace_analysis(
                    targ_mat,
                    flnk_mat,
                    layer,
                    model_type,
                )

                subspace_dfs.append(subspace_df)

        return pd.concat(subspace_dfs).reset_index(drop=True)

    def _subspace_analysis(
        self,
        T,
        F,
        layer,
        model_type,
    ):
        Ut, St, Vht = np.linalg.svd(T, full_matrices=False)
        Uf, Sf, Vhf = np.linalg.svd(F, full_matrices=False)

        # See https://www.merl.com/publications/docs/TR2012-058.pdf
        # Take transpose so notation is consistent
        X, Y = Vht.T, Vhf.T  # X, Y are n_features x 4

        # Calculate average of principal angles between target/flanker subspaces
        U_cross, S_cross, Vh_cross = np.linalg.svd(X.T @ Y)

        principal_alignment = np.mean(S_cross)
        targ_princip_v = X @ U_cross
        flnk_princip_v = Y @ Vh_cross.T
        # Get target subspace orthogonal to flanker subspace and vice-versa
        targ_flnk_proj = np.diag(targ_princip_v.T @ flnk_princip_v)
        targ_orthog = targ_princip_v - flnk_princip_v @ np.diag(targ_flnk_proj)
        flnk_orthog = flnk_princip_v - targ_princip_v @ np.diag(targ_flnk_proj)
        assert np.allclose(targ_orthog.T @ flnk_princip_v, np.zeros((4, 4)))
        assert np.allclose(flnk_orthog.T @ targ_princip_v, np.zeros((4, 4)))

        vals = [principal_alignment]
        stats = ["principal_alignment"]
        for i in range(4):
            stats.append(f"targ_singval{i}")
            stats.append(f"flnk_singval{i}")
            vals.append(St[i])
            vals.append(Sf[i])

        stats_df = pd.DataFrame(
            {
                "stat": stats,
                "value": vals,
                "layer": len(stats) * [layer],
                "user_id": len(stats) * [self.user_id],
                "model_type": len(stats) * [model_type],
            }
        )
        return stats_df


class InvarianceMixin:
    """Functions for analysis of invariance/tolerance to position, layout, etc.
    Incongruent trials only."""

    def invariance_analysis(
        self,
    ):
        model_fts = [self.vam_fts, self.task_opt_fts]
        model_dfs = [self.vam_df, self.task_opt_df]
        model_types = ["vam", "task_opt"]
        contexts = ["dis_dirs_num", "layouts_num", "xpos_bin_repr", "ypos_bin_repr"]
        all_dfs = []

        for x, model_df, model_type in zip(model_fts, model_dfs, model_types):
            idf, incon_idx = self._get_incon_idx(model_df)

            for layer_idx, layer in enumerate(self.layers):
                layer_act = x[f"layer{layer_idx}"]["features"][incon_idx]
                layer_act_stnd = StandardScaler().fit_transform(layer_act)

                for con in contexts:
                    con_results = []
                    con_vals = idf[con].unique()
                    for cv in con_vals:
                        other_vals = [v for v in con_vals if v != cv]
                        train_idx = idf.query(f"{con} == @cv").index.to_numpy()
                        test_idx = idf.query(f"{con} in @other_vals").index.to_numpy()
                        train_x = layer_act[train_idx, :]
                        test_x = layer_act[test_idx, :]
                        train_y = idf.loc[train_idx, "targ_dirs"].to_numpy()
                        test_y = idf.loc[test_idx, "targ_dirs"].to_numpy()
                        acc = self._invariance_classifier(
                            train_x, test_x, train_y, test_y
                        )
                        con_results.append({"acc": acc})

                    con_mean_acc = np.mean([res["acc"] for res in con_results])
                    stat_names = [f"{con}_acc"]

                    all_dfs.append(
                        pd.DataFrame(
                            {
                                "layer": [layer],
                                "stat": stat_names,
                                "value": [con_mean_acc],
                                "user_id": [self.user_id],
                                "model_type": [model_type],
                            },
                            index=[0],
                        )
                    )

        return pd.concat(all_dfs).reset_index(drop=True)

    def _invariance_classifier(self, x_train, x_test, y_train, y_test):
        clf = LinearSVC(
            multi_class="ovr",
            fit_intercept=True,
            loss="squared_hinge",
        ).fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        return acc


class DimensionalityMixin:
    """
    Functions for calculating the participation ratio measure of
    the dimensionality of model activations in each layer.
    """

    def dimensionality_analysis(self):
        stat_labels = [
            "centered_participation_ratio",
        ]

        fts = [self.vam_fts, self.task_opt_fts]
        trial_dfs = [self.vam_df, self.task_opt_df]
        model_types = ["vam", "task_opt"]
        all_dfs = []

        for x, df, model_type in zip(fts, trial_dfs, model_types):
            # Filter for incongruent trials
            incon_df, incon_idx = self._get_incon_idx(df)
            targets = incon_df["targ_dirs_num"].to_numpy()

            for layer_idx, layer in enumerate(self.layers):
                layer_fts = x[f"layer{layer_idx}"]["features"][incon_idx, :]
                pr = self._layer_dim_stats(layer_fts, targets)

                this_df = pd.DataFrame(
                    {
                        "value": [pr],
                        "stat": stat_labels,
                        "layer": [layer],
                        "model_type": [model_type],
                        "user_id": [self.user_id],
                    },
                    index=[0],
                )
                all_dfs.append(this_df)

        return pd.concat(all_dfs)

    def _layer_dim_stats(self, fts, targets):
        for targ in range(4):
            targ_mask = targets == targ
            targ_x = fts[targ_mask]
            targ_mean = targ_x.mean(axis=0)

            if targ == 0:
                class_centered = targ_x - targ_mean
            else:
                class_centered = np.concatenate(
                    [class_centered, targ_x - targ_mean], axis=0
                )

        # Calculate participation ratio
        centered_demeaned = class_centered - class_centered.mean(axis=0)
        cov = (centered_demeaned.T @ centered_demeaned) / len(fts)
        pr = np.trace(cov) ** 2 / np.trace(cov @ cov)

        return pr


class CorrelationMixin:
    """Functions for plotting and calculating stats on bivariate data."""

    def pearson_stats(self, x, y, alpha=0.05):
        """Calculate a bootstrap (1-alpha) CI for Pearson's r
        and p-value w/ a permutation test on the bivariate dataset (x, y).
        """

        rng = np.random.default_rng(seed=self.seed)
        perm_method = PermutationMethod(n_resamples=self.n_boot, random_state=rng)
        boot_method = BootstrapMethod(
            method="BCa", n_resamples=self.n_boot, random_state=rng
        )
        res = pearsonr(x, y, method=perm_method)
        ci = res.confidence_interval(confidence_level=1 - alpha, method=boot_method)
        return res.statistic, res.pvalue, ci[0], ci[1]

    def plot_scatter(
        self,
        x,
        y,
        x_label,
        y_label,
        title,
        ax,
        stats=None,
        plot_unity=False,
        lw=0.5,
        size=3,
    ):
        line_ext = 0.1 * (max(x) - min(x))
        # Plot best fit line
        plot_x = np.array([min(x) - line_ext, max(x) + line_ext])
        m, b = np.polyfit(x, y, 1)
        ax.plot(plot_x, m * plot_x + b, "r--", zorder=1, linewidth=lw)
        if plot_unity:
            ax.plot(plot_x, plot_x, "k-", zorder=1, linewidth=lw)

        # Plot scatter
        sns.scatterplot(x=x, y=y, ax=ax, s=size)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title, loc="left", fontweight="bold", pad=2)

        if stats is not None:
            # Add textbox to ax with stats
            textstr = ""
            textstr += f"r = {stats['r']:.2f}\n"
            textstr += f"p = {stats['p']:.2e}\n"

            ax.text(
                0.5,
                0.95,
                textstr,
                transform=ax.transAxes,
                fontsize=6,
                verticalalignment="top",
            )

    def calc_write_pearson_stats(
        self,
        x,
        y,
        x_label,
        y_label,
        stat,
        alpha=0.05,
    ):
        m, b = np.polyfit(x, y, 1)
        r, p, ci_lo, ci_hi = self.pearson_stats(
            x,
            y,
            alpha=alpha,
        )
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_sem = np.std(x) / np.sqrt(len(x))
        y_sem = np.std(y) / np.sqrt(len(y))
        p_str = "p = {:0.2e}".format(p)
        r_str = f"Pearson's r = {round(r, 2)}, 95% CI: ({round(ci_lo, 2)}, {round(ci_hi, 2)})"
        with open(self.stats_file, "a") as f:
            f.write(f"{stat} stats, N = {len(x)}:\n")
            f.write(f"{r_str}, {p_str}\n")
            f.write(f"Best-fit slope: {m}; intercept: {b}\n")
            f.write(f"{x_label} mean +/- s.e.m.: {x_mean} +/- {x_sem}\n")
            f.write(f"{y_label} mean +/- s.e.m.: {y_mean} +/- {y_sem}\n")
            f.write("--------------------------------------------------------\n")

    def pearson_bar_panel(self, df, ax, feature_stat, behavior_stat, title=""):
        plot_df = df.query(
            "behavior_stat == @behavior_stat and feature_stat == @feature_stat"
        )
        corr_stats = {}

        with open(self.stats_file, "a") as f:
            f.write(f"{feature_stat} vs. {behavior_stat}\n")

        for layer in self.layers:
            layer_df = plot_df.query("layer == @layer")
            r, p, r_ci_lo, r_ci_hi = self.pearson_stats(
                layer_df["behavior_value"], layer_df["feature_value"]
            )

            # Adjust for multiple comparisons
            p = p * len(self.layers)

            plot_ci_lo = r - r_ci_lo
            plot_ci_hi = r_ci_hi - r
            corr_stats[layer] = (r, plot_ci_lo, plot_ci_hi, p)
            with open(self.stats_file, "a") as f:
                f.write(f"Layer {layer}: r = {r:.3f}, p = {p:.3f}\n")

        with open(self.stats_file, "a") as f:
            f.write("--------------------------------------------------------\n")

        _ = self.plot_pearson_bar(self.layers, corr_stats, ax)

        ax.set_title(
            f"{title}",
            loc="center",
            pad=2,
            fontsize=6,
        )

    def plot_pearson_bar(self, layers, data, ax, **kwargs):
        # Plot Pearson's r vs. layer
        colors = sns.color_palette("viridis", n_colors=7).as_hex()
        width = kwargs.get("width", 0.75)
        asterisk_spacer = kwargs.get("asterisk_spacer", 0.05)
        x = 0
        y_max = np.amax([data[layer][0] + data[layer][2] for layer in layers])
        y_min = np.amin([data[layer][0] - data[layer][1] for layer in layers])

        for layer, color in zip(layers, colors):
            r, ci_lo, ci_hi, p = data[layer]

            ax.bar(
                x,
                r,
                yerr=[[ci_lo], [ci_hi]],
                width=width,
                error_kw={"elinewidth": 1},
                color=color,
            )

            if p < 0.05:
                ax.scatter(
                    x,
                    y_max + asterisk_spacer,
                    marker="$*$",
                    color="k",
                    s=10,
                    linewidths=0.1,
                )
            x += 1

        ax = self._adjust_bar(layers, ax, **kwargs)
        ax.set_ylim([y_min - asterisk_spacer, y_max + 5 * asterisk_spacer])

        return ax

    def _adjust_bar(self, layers, ax, **kwargs):
        ax.set_xlabel(kwargs.get("xlabel", None))
        ax.set_ylabel(kwargs.get("ylabel", None))
        ax.set_xticks(np.arange(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha="right", rotation_mode="anchor")
        if "yticks" in kwargs.keys():
            ax.set_yticks(kwargs["yticks"])
        ax.set_xlim(kwargs.get("xlim", None))
        ax.set_ylim(kwargs.get("ylim", None))
        if kwargs.get("plot_legend", False):
            ax.legend()
            ax.get_legend().get_frame().set_linewidth(0.0)
        return ax

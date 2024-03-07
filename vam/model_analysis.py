import pdb
import pickle
import os
import itertools
import time
import shutil

import numpy as np
import pandas as pd
import seaborn as sns

from .mixins import (
    BasicAnalysisMixin,
    SubspaceMixin,
    InvarianceMixin,
    DimensionalityMixin,
    LayoutPositionMixin,
    DeltaPlotCAFMixin,
    SingleUnitMixin,
    DecoderMixin,
    LBAParamsMixin,
)


class ModelAnalysis(
    BasicAnalysisMixin,
    DimensionalityMixin,
    InvarianceMixin,
    SubspaceMixin,
    LayoutPositionMixin,
    DeltaPlotCAFMixin,
    SingleUnitMixin,
    DecoderMixin,
    LBAParamsMixin,
):
    def __init__(
        self,
        models_dir,
        config,
        user_id,
        rand_seed,
        derivatives_dir="derivatives",
        summary_dir="summary_stats",
        has_binned_rt_model=False,
    ):
        self.vam_dir = os.path.join(models_dir, f"{derivatives_dir}/vam/user{user_id}")
        self.vam_df = pd.read_csv(os.path.join(self.vam_dir, "trial_df.csv"))
        with open(os.path.join(self.vam_dir, "feature_stats.pkl"), "rb") as f:
            self.vam_fts = pickle.load(f)

        self.task_opt_dir = os.path.join(
            models_dir, f"{derivatives_dir}/task_opt/user{user_id}"
        )
        self.task_opt_df = pd.read_csv(os.path.join(self.task_opt_dir, "trial_df.csv"))
        with open(os.path.join(self.task_opt_dir, "feature_stats.pkl"), "rb") as f:
            self.task_opt_fts = pickle.load(f)

        self.layers = self.get_layers(config)
        self.user_id = user_id
        self.rand_seed = rand_seed
        self.directions = ["Left", "Right", "Up", "Down"]
        self.has_binned_rt_model = has_binned_rt_model
        self.n_rt_bins = config.data.n_rt_bins

        if has_binned_rt_model:
            self.binned_rt_dir = os.path.join(
                models_dir, f"{derivatives_dir}/binned_rt"
            )
            self.binned_rt_df = self._combine_binned_rt_dfs(self.binned_rt_dir, config)

        with open(os.path.join(self.vam_dir, "mask_info.pkl"), "rb") as f:
            self.mask_info = pickle.load(f)

        self.save_dir = os.path.join(
            models_dir, f"{derivatives_dir}/{summary_dir}/user{user_id}"
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.summary_save_fn = os.path.join(self.save_dir, "summary_stats.pkl")

        # Define bins for stimulus position analyses (pixels)
        self.xpos_bins = [-210, -150, -100, -50, 0, 50, 100, 150, 210]
        self.ypos_bins = np.arange(-75, 76, 25)
        for df in [self.vam_df, self.task_opt_df]:
            df["xpos_bin_behavior"] = pd.cut(df["xpositions"], self.xpos_bins)
            df["xpos_bin_repr"] = pd.cut(df["xpositions"], self.xpos_bins, labels=False)
            df["ypos_bin_behavior"] = pd.cut(df["ypositions"], self.ypos_bins)
            df["ypos_bin_repr"] = pd.cut(df["ypositions"], self.ypos_bins, labels=False)

        self.add_correct()
        self.remap_vars()

    def run_analysis(self):
        basic_df = self.basic_stats()
        layout_pos_summary_df, layout_pos_mean_df = self.layout_position_analysis()
        lba_param_df = self.lba_param_analysis()
        dim_df = self.dimensionality_analysis()
        decoder_df, decoder_ax = self.decoder_analysis()
        subspace_df = self.subspace_analysis(decoder_ax)
        invariance_df = self.invariance_analysis()
        delta_df, caf_df = self.delta_caf_analysis()
        mi_df = self.mutual_info_analysis()
        basic_unit_df, activation_df = self.unit_modulation_analysis()

        stats = {
            "basic": basic_df,
            "layout_pos_summary": layout_pos_summary_df,
            "layout_pos_mean": layout_pos_mean_df,
            "lba_params": lba_param_df,
            "dimensionality": dim_df,
            "subspace": subspace_df,
            "invariance": invariance_df,
            "delta_plot": delta_df,
            "caf": caf_df,
            "mutual_info": mi_df,
            "basic_modulation": basic_unit_df,
            "decoding": decoder_df,
        }

        with open(self.summary_save_fn, "wb") as f:
            pickle.dump(stats, f)
        activation_df.to_csv(
            os.path.join(self.save_dir, "activation_df.csv"), index=False
        )

    def _combine_binned_rt_dfs(self, base_derivs_dir, config):
        all_dfs = []
        for rt_bin in range(config.data.n_rt_bins):
            rt_bin_dir = os.path.join(
                base_derivs_dir, f"user{self.user_id}_rt_bin{rt_bin}"
            )
            bin_df = pd.read_csv(os.path.join(rt_bin_dir, "trial_df.csv"))
            bin_df["rt_bin"] = rt_bin
            all_dfs.append(bin_df)

        return pd.concat(all_dfs)

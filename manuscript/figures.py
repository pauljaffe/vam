import os
import pickle
import pdb

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f_oneway, wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from vam.mixins import (
    CorrelationMixin,
    BasicAnalysisMixin,
    DeltaPlotCAFMixin,
    SingleUnitMixin,
)


class BaseFigure(BasicAnalysisMixin):
    """Base class for figures."""

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        self.stats = stats
        self.derivatives_dir = derivatives_dir
        self.figs_dir = os.path.join(derivatives_dir, summary_dir, "figures")
        self.metadata = metadata  # model/participant metadata
        self.config = config  # model config file
        self.seed = seed
        self.n_boot = n_boot
        self.layers = self.get_layers(config)
        self.stats_file = os.path.join(self.figs_dir, f"{self.fig_str}_stats.txt")
        self.figdpi = 300
        os.makedirs(self.figs_dir, exist_ok=True)
        if os.path.exists(self.stats_file):
            os.remove(self.stats_file)

    def add_title_ax(self, fig, ax, title, pad=2, ax_offset=1):
        colspan = ax.get_subplotspec().colspan
        rowspan = ax.get_subplotspec().rowspan
        gs = ax.get_gridspec()
        title_ax = fig.add_subplot(
            gs[
                rowspan[0] : rowspan[-1],
                colspan[0] - ax_offset : colspan[-1] - ax_offset,
            ],
            frameon=False,
        )
        title_ax.set_title(title, loc="left", pad=pad)
        title_ax.set_axis_off()
        return ax, title_ax

    def rotate_ax_labels(self, ax, adjust_spacing=False):
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        if adjust_spacing:
            ax.tick_params(axis="both", which="major", pad=2)

    def save_figure(self, save_eps=False):
        plt.savefig(
            os.path.join(self.figs_dir, f"{self.fig_str}.png"),
            dpi=self.figdpi,
            bbox_inches="tight",
        )
        if save_eps:
            plt.savefig(
                os.path.join(self.figs_dir, f"{self.fig_str}.eps"),
                format="eps",
                dpi=self.figdpi,
                bbox_inches="tight",
            )

    def _lineplot_panel(
        self,
        df,
        ax,
        stat,
        ylabel,
        plot_model_type=False,
        mult_stats=True,
        hue_order=None,
    ):
        if type(stat) is str:
            stat = [stat]
        plot_df = df.query("feature_stat in @stat")

        if len(stat) == 1:
            mult_stats = False

        if mult_stats:
            sns.lineplot(
                data=plot_df,
                x="layer",
                y="feature_value",
                hue="model_type",
                style="feature_stat",
                ax=ax,
                err_style="bars",
                palette="colorblind",
                hue_order=hue_order,
            )
        else:
            sns.lineplot(
                data=plot_df,
                x="layer",
                y="feature_value",
                hue="model_type",
                ax=ax,
                err_style="bars",
                palette="colorblind",
                hue_order=hue_order,
            )

        ax.set_ylabel(ylabel)
        if not mult_stats and not plot_model_type:
            ax.get_legend().remove()
        ax.set_xlabel("")
        self.rotate_ax_labels(ax)


class Figure2(BaseFigure, CorrelationMixin, BasicAnalysisMixin):
    """Create Figure 2 from the manuscript (model/participant comparison;
    LBA parameter summary). Also get stats on fraction of invalid trials
    (all drift rates negative).
    """

    figsize = (7, 4.7)
    fig_str = "Figure2"
    xpos_bins = [-210, -150, -100, -50, 0, 50, 100, 150, 210]
    ypos_bins = np.arange(-75, 76, 25)
    layout_order = ["---", "|", "+", "<", ">", "v", "^"]

    # Exemplars
    rt_example_id = 5256
    layout_example_id = 7522
    position_example_id = 7735

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )

        # Combine 70-79 and 80-89 age bins
        age_remap = {"70-79": "70-89", "80-89": "70-89"}
        metadata = metadata.replace({"binned_age": age_remap})

        basic_df = stats["basic"].merge(metadata, on=["user_id"], how="left")
        self.basic_df = basic_df.query("model_type == 'vam'")
        self.layout_pos_df = stats["layout_pos_summary"].merge(
            metadata, on=["user_id"], how="left"
        )

        self.lba_params_df = (
            stats["lba_params"]
            .query("model_type == 'vam'")
            .merge(metadata, on=["user_id"], how="left")
        )

        self.rt_example_df = pd.read_csv(
            os.path.join(derivatives_dir, f"vam/user{self.rt_example_id}/trial_df.csv")
        )

        self.layout_example_df = pd.read_csv(
            os.path.join(
                derivatives_dir, f"vam/user{self.layout_example_id}/trial_df.csv"
            )
        )
        self.layout_example_df = self.layout_example_df.query(
            "response_dirs == targ_dirs"
        )
        self.layout_example_df = self.remap_vars(self.layout_example_df)

        self.position_example_df = pd.read_csv(
            os.path.join(
                derivatives_dir, f"vam/user{self.position_example_id}/trial_df.csv"
            )
        )
        self.position_example_df = self.position_example_df.query(
            "response_dirs == targ_dirs"
        )
        self.position_example_df["xpos_bin"] = pd.cut(
            self.position_example_df["xpositions"], self.xpos_bins
        )

        # Rename some values for plotting
        entity_map = {"model": "Model", "user": "Participant"}
        for df in [
            self.basic_df,
            self.rt_example_df,
            self.layout_example_df,
            self.position_example_df,
        ]:
            df["model_user"] = df["model_user"].replace(entity_map)

        congruency_map = {0: "Congruent", 1: "Incongruent"}
        self.lba_params_df["congruency"] = self.lba_params_df["congruency"].replace(
            congruency_map
        )

        feature_map = {
            "layouts": "Layout",
            "xpos_bin_behavior": "X-position",
            "ypos_bin_behavior": "Y-position",
        }
        self.layout_pos_df["stimulus_feature"] = self.layout_pos_df[
            "stimulus_feature"
        ].replace(feature_map)

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(20, 35)
        ax1 = fig.add_subplot(gs[:4, 3:8])
        ax2 = fig.add_subplot(gs[:4, 12:17])
        ax3 = fig.add_subplot(gs[:4, 21:26])
        ax4 = fig.add_subplot(gs[:4, 30:35])
        ax5 = fig.add_subplot(gs[8:12, 7:12])
        ax6 = fig.add_subplot(gs[8:12, 16:21])
        ax7 = fig.add_subplot(gs[8:12, 25:30])
        ax8 = fig.add_subplot(gs[16:20, 7:12])
        ax9 = fig.add_subplot(gs[16:20, 16:21])
        ax10 = fig.add_subplot(gs[16:20, 25:30])
        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

        for ax, title in zip(axes, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]):
            self.add_title_ax(fig, ax, title, pad=3)

        # Write some basic stats
        n_participants = len(self.basic_df["user_id"].unique())
        with open(self.stats_file, "w") as f:
            f.write(f"N participants: {n_participants}\n")

        # Get stats on fraction of invalid trial stats
        inval_mean = self.basic_df.query("stat == 'frac_invalid'")["value"].mean()
        inval_sem = self.basic_df.query("stat == 'frac_invalid'")["value"].sem()
        with open(self.stats_file, "a") as f:
            f.write(
                "Mean +/- s.e.m. fraction invalid trials: "
                f"{inval_mean:.6f} +/- {inval_sem:.6f}\n"
            )

        # Example RT distribution
        A = sns.kdeplot(data=self.rt_example_df, x="rts", hue="model_user", ax=axes[0])
        axes[0].set_xlabel("RT (s)")
        axes[0].set_xlim([0.35, 1.1])
        sns.move_legend(
            axes[0], "upper left", bbox_to_anchor=(0, 1.4), frameon=False, title=None
        )

        # Model vs. participant behavior scatterplots
        self._scatter_panels([axes[1], axes[2], axes[3], axes[4]])

        # Drift rate summary
        self._drift_panel(axes[5])

        # Mean RT vs. age
        self._rt_age_panel(axes[6])

        # RT vs. layout, example
        self._layout_example_panel(axes[7])

        # RT vs. spatial position, example
        self._position_example_panel(axes[8])

        # Layout/position group summary
        self._layout_pos_summary_panel(axes[9])

        # Save summary stats and figure
        self._write_lba_stats()

        for feature in ["Layout", "X-position", "Y-position"]:
            for stat in ["mean_rt", "mean_correct"]:
                self._write_layout_pos_stats(feature, stat)

        self.save_figure()

        return fig

    def _scatter_panels(self, axes):
        # Model vs. participant scatter plots
        B_params = {
            "ax_lims": [],
            "metric": "mean_rt",
            "label": "mean RT (s)",
        }
        C_params = {
            "ax_lims": [],
            "metric": "accuracy",
            "label": "accuracy",
        }
        D_params = {
            "ax_lims": [],
            "metric": "rt_con_effect",
            "label": "RT\ncongruency effect (s)",
        }
        E_params = {
            "ax_lims": [],
            "metric": "acc_con_effect",
            "label": "accuracy\ncongruency effect",
        }

        params = [B_params, C_params, D_params, E_params]
        for p, ax in zip(params, axes):
            metric = p["metric"]
            user_label = f"Participant {p['label']}"
            model_label = f"Model {p['label']}"
            user_vals = self.basic_df.query(
                "model_user == 'Participant' and stat == @p['metric']"
            )["value"].values
            model_vals = self.basic_df.query(
                "model_user == 'Model' and stat == @p['metric']"
            )["value"].values
            self.plot_scatter(
                user_vals,
                model_vals,
                user_label,
                model_label,
                None,
                ax,
                plot_unity=True,
            )
            self.calc_write_pearson_stats(
                user_vals,
                model_vals,
                "Participant",
                "Model",
                p["label"],
            )

    def _drift_panel(self, ax):
        # Panel F: Drift rate summary
        df = self.lba_params_df.query(
            "stat in ['target_drift', 'flanker_drift', 'other_drift']"
        )
        df = df.replace(
            {
                "stat": {
                    "target_drift": "Target",
                    "flanker_drift": "Flanker",
                    "other_drift": "Other",
                }
            }
        )
        df = df.drop(df[(df.congruency == "Congruent") & (df.stat == "Flanker")].index)

        F = sns.barplot(
            data=df,
            x="stat",
            y="value",
            hue="congruency",
            order=["Target", "Other", "Flanker"],
            hue_order=["Incongruent", "Congruent"],
            ax=ax,
        )
        ax.set_ylabel("Drift rate (a.u.)")
        ax.set_xlabel("")
        self.rotate_ax_labels(ax)
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(0, 1.4), frameon=False, title=None
        )

    def _rt_age_panel(self, ax):
        rt_df = self.basic_df.query("stat == 'mean_rt'")
        G = sns.barplot(
            data=rt_df,
            x="binned_age",
            y="value",
            hue="model_user",
            ax=ax,
        )
        ax.set_xlabel("Age bin (years)")
        ax.set_ylabel("Mean RT (s)")
        self.rotate_ax_labels(ax)
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(0, 1.4), frameon=False, title=None
        )
        ax.set_ylim([0.4, 1.0])

    def _layout_example_panel(self, ax):
        layout_bar = sns.barplot(
            data=self.layout_example_df,
            x="layouts",
            y="rts",
            hue="model_user",
            ax=ax,
            order=self.layout_order,
        )

        ax.set_xlabel("Layout")
        ax.set_ylabel("Mean RT (s)")
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(0, 1.4), frameon=False, title=None
        )

        mean_rts = self.layout_example_df.groupby(["layouts"])["rts"].mean()
        sem_rts = self.layout_example_df.groupby(["layouts"])["rts"].sem()
        upper_y = np.max(mean_rts + sem_rts)
        lower_y = np.min(mean_rts - sem_rts)

        ax.set_ylim(lower_y - 0.05, upper_y + 0.05)

        layout_ex_r = self.layout_pos_df.query(
            "user_id == @self.layout_example_id & stat == 'mean_rt' & stimulus_feature == 'Layout'"
        )["pearsons_r"].values[0]
        with open(self.stats_file, "a") as f:
            f.write(f"RT vs. layout example Pearson's r: {layout_ex_r}\n")
            f.write("--------------------------------------------------------\n")

    def _position_example_panel(self, ax):
        pos_bar = sns.barplot(
            data=self.position_example_df,
            x="xpos_bin",
            y="rts",
            hue="model_user",
            ax=ax,
        )

        ax.set_xlabel("Horizontal position\n(pixels)")
        ax.set_ylabel("Mean RT (s)")
        self.rotate_ax_labels(ax)
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(0, 1.4), frameon=False, title=None
        )

        mean_rts = self.position_example_df.groupby(["xpos_bin"])["rts"].mean()
        sem_rts = self.position_example_df.groupby(["xpos_bin"])["rts"].sem()
        upper_y = np.max(mean_rts + sem_rts)
        lower_y = np.min(mean_rts - sem_rts)

        ax.set_ylim(lower_y - 0.05, upper_y + 0.05)

        pos_ex_r = self.layout_pos_df.query(
            "user_id == @self.position_example_id & stat == 'mean_rt' & stimulus_feature == 'X-position'"
        )["pearsons_r"].values[0]
        with open(self.stats_file, "a") as f:
            f.write(f"RT vs. x-position example Pearson's r: {pos_ex_r}\n")
            f.write("--------------------------------------------------------\n")

    def _write_lba_stats(self):
        # Write LBA parameter summary stats to stats file
        all_stats = []
        all_user_ids = []
        all_vals = []
        for stat in ["target_drift", "flanker_drift", "other_drift"]:
            # Drift rate averaged across congruent/incongruent,
            both_df = self.lba_params_df.query("stat == @stat")
            both_means = both_df.groupby("user_id")["value"].mean()
            # Add to lba_params_df
            for user_id, val in both_means.items():
                all_user_ids.append(user_id)
                all_vals.append(val)
                all_stats.append(f"{stat}_mean")

        combined_df = pd.DataFrame(
            {
                "user_id": all_user_ids,
                "value": all_vals,
                "stat": all_stats,
            }
        )
        self.lba_params_df = pd.concat([self.lba_params_df, combined_df])

        n_models = len(self.lba_params_df["user_id"].unique())
        lba_congruency_summary = self.lba_params_df.groupby(["stat", "congruency"])[
            "value"
        ].describe()
        lba_congruency_summary["s.e.m."] = lba_congruency_summary["std"] / np.sqrt(
            n_models
        )
        lba_congruency_summary = lba_congruency_summary.applymap(lambda x: f"{x:0.3f}")
        lba_congruency_summary.drop(
            columns=["25%", "50%", "75%", "min", "max"], inplace=True
        )

        summary_df = self.lba_params_df.query(
            "stat not in ['target_drift', 'flanker_drift', 'other_drift']"
        )
        lba_summary = summary_df.groupby(["stat"])["value"].describe()
        lba_summary["s.e.m."] = lba_summary["std"] / np.sqrt(n_models)
        lba_summary = lba_summary.applymap(lambda x: f"{x:0.3f}")
        lba_summary.drop(columns=["25%", "50%", "75%", "min", "max"], inplace=True)

        with open(self.stats_file, "a") as f:
            f.write("LBA parameter stats, watch out for change in indentation!:\n")
        lba_summary.to_csv(self.stats_file, sep="\t", mode="a")
        with open(self.stats_file, "a") as f:
            f.write("\n")
        lba_congruency_summary.to_csv(self.stats_file, sep="\t", mode="a")
        with open(self.stats_file, "a") as f:
            f.write("--------------------------------------------------------\n")

    def _layout_pos_summary_panel(self, ax):
        layout_pos_sig_df = self.layout_pos_df.query(
            "user_effect_p < 0.05 & stat == 'mean_rt'"
        )
        pos_bar2 = sns.histplot(
            data=layout_pos_sig_df,
            x="pearsons_r",
            ax=ax,
            hue="stimulus_feature",
            # palette="viridis",
            element="step",
            stat="density",
            cumulative=True,
            common_norm=False,
            fill=False,
            multiple="dodge",
        )
        ax.set_xlabel("Pearson's $\it{r}$")
        ax.set_ylabel("Probability")
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(1, 1.1), frameon=False, title=None
        )

    def _write_layout_pos_stats(self, stim_feature, stat):
        sig_df = self.layout_pos_df.query(
            "stimulus_feature == @stim_feature & user_effect_p < 0.05 & stat == @stat"
        )

        n_users = len(sig_df)
        stats = {}
        stats["num_users"] = n_users
        if n_users > 1:
            stats["median"] = sig_df.groupby("stat").median()["pearsons_r"].values
            stats["mean"] = sig_df.groupby("stat").mean()["pearsons_r"].values
            stats["SEM"] = sig_df.groupby("stat").std()["pearsons_r"].values / np.sqrt(
                n_users
            )
        else:
            stats["median"] = np.nan
            stats["mean"] = np.nan
            stats["SEM"] = np.nan
        stats = pd.DataFrame(stats, index=[0])
        stats = stats.applymap(lambda x: f"{x:0.3f}")

        with open(self.stats_file, "a") as f:
            f.write(f"Stats for {stat} vs {stim_feature} (Pearson's r):\n")
        stats.to_csv(self.stats_file, sep="\t", mode="a")


class Figure3(BaseFigure, BasicAnalysisMixin):
    """Create Figure 3 from the manuscript (target direction
    representation). Also calculates the fraction of dead units
    in each layer.
    """

    figsize = (4.5, 2.2)
    fig_str = "Figure3"
    activation_example_id = 677

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )

        self.decoding_stats = [
            "targ_dirs_num",
        ]
        self.decoding_df = stats["decoding"].query("model_type == 'vam'")

        self.mi_df = stats["mutual_info"].query(
            "model_type == 'vam' and stat == 'mi_mean'"
        )
        self.mi_vars = [
            "targ_dirs_num",
        ]

        self.dim_df = stats["dimensionality"].query("model_type  == 'vam'")
        self.dim_stats = ["centered_participation_ratio"]

        self.selectivity_df = stats["basic_modulation"].query(
            "model_type == 'vam' and stimulus_feature == 'targ_dirs_num'"
        )
        self.selectivity_vars = ["f_pos", "f_neg", "f_complex", "f_not_signif"]

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(13, 32)

        decoding_ax = fig.add_subplot(gs[:4, 5:15])
        self._decoding_panel(decoding_ax)

        mi_ax = fig.add_subplot(gs[9:, 5:15])
        self._mutual_info_panel(mi_ax)

        dim_ax = fig.add_subplot(gs[:4, 22:])
        self._dim_panel(dim_ax)

        selectivity_ax = fig.add_subplot(gs[9:, 22:])
        self._selectivity_panel(selectivity_ax)

        for ax, title in zip(
            [decoding_ax, mi_ax, dim_ax, selectivity_ax], ["B", "C", "D", "E"]
        ):
            self.add_title_ax(fig, ax, title, pad=6, ax_offset=5)

        self.save_figure(save_eps=True)

        return fig

    def _decoding_panel(self, ax):
        plot_df = self.decoding_df.query("decoded_variable in @self.decoding_stats")

        # Write summary stats to stats file
        N = len(plot_df["user_id"].unique())
        for dim in ["targ_dirs_num"]:
            dim_df = plot_df.query("decoded_variable == @dim")
            decoding_vals = dim_df.groupby("layer")["decoder_accuracy"]
            decoding_means = decoding_vals.mean()
            decoding_sems = decoding_vals.std() / np.sqrt(N)
            decoding_stats = pd.DataFrame(
                {"mean": decoding_means, "sem": decoding_sems}
            )
            decoding_stats = decoding_stats.applymap(lambda x: f"{x:0.3f}")

            with open(self.stats_file, "a") as f:
                f.write(f"{dim} direction decoding stats:\n")
            decoding_stats.to_csv(self.stats_file, sep="\t", mode="a")

        plot_df = plot_df.rename(
            columns={
                "decoder_accuracy": "Decoder performance",
                "layer": "CNN layer",
            }
        )

        sns.lineplot(
            data=plot_df,
            x="CNN layer",
            y="Decoder performance",
            ax=ax,
            err_style="bars",
            palette="colorblind",
        )
        self.rotate_ax_labels(ax)
        ax.set_ylabel("Decoder\nperformance")
        ax.set_xlabel("")

    def _mutual_info_panel(self, ax):
        plot_df = self.mi_df.query("mi_variable in @self.mi_vars")

        sns.lineplot(
            data=plot_df,
            x="layer",
            y="value",
            ax=ax,
            err_style="bars",
            palette="colorblind",
        )
        ax.set_xlabel("CNN layer")
        ax.set_ylabel("Norm. mutual\ninformation")
        self.rotate_ax_labels(ax)

    def _dim_panel(self, ax):
        plot_df = self.dim_df.query("stat in @self.dim_stats")
        sns.lineplot(
            data=plot_df,
            x="layer",
            y="value",
            ax=ax,
            err_style="bars",
            palette="colorblind",
        )
        self.rotate_ax_labels(ax)
        ax.set_ylabel("Participation\nratio")
        ax.set_xlabel("")

    def _selectivity_panel(self, ax):
        plot_df = self.selectivity_df.query("stat in @self.selectivity_vars")
        plot_df = plot_df.rename(
            columns={
                "value": "Proportion of units",
                "stat": "Unit type",
                "layer": "CNN layer",
            }
        )

        plot_df = plot_df.replace(
            {
                "Unit type": {
                    "f_not_signif": "No modulation",
                    "f_pos": "Selective (+)",
                    "f_neg": "Selective (-)",
                    "f_complex": "Complex",
                }
            }
        )
        sns.barplot(
            data=plot_df,
            x="CNN layer",
            y="Proportion of units",
            hue="Unit type",
            ax=ax,
            palette="viridis",
        )
        # ax.set_ylim([0, 1.1])
        sns.move_legend(
            ax,
            "lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.55, 0.9),
            title=None,
        )
        self.rotate_ax_labels(ax)
        ax.set_ylabel("Proportion\nof units")


class Figure4(BaseFigure, BasicAnalysisMixin):
    """Create figure 4 from the manuscript (suppression + tolerance)"""

    figsize = (7, 1.2)
    fig_str = "Figure4"

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )

        self.invariance_df = stats["invariance"].query("model_type == 'vam'")
        self.decoding_df = stats["decoding"].query("model_type == 'vam'")
        self.mi_df = stats["mutual_info"].query(
            "model_type == 'vam' and stat == 'mi_mean'"
        )
        self.stimulus_fts = [
            "dis_dirs_num",
            "layouts_num",
            "xpos_bin_repr",
            "ypos_bin_repr",
        ]
        self.hue_order = [
            "Flanker direction",
            "Layout",
            "Horizontal position",
            "Vertical position",
        ]

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(4, 43)

        invariance_ax = fig.add_subplot(gs[:, 3:13])
        self._invariance_panel(invariance_ax)

        decoding_ax = fig.add_subplot(gs[:, 18:28])
        self._decoding_panel(decoding_ax)

        mi_ax = fig.add_subplot(gs[:, 33:43])
        self._mutual_info_panel(mi_ax)

        for ax, title in zip([invariance_ax, decoding_ax, mi_ax], ["A", "B", "C"]):
            self.add_title_ax(fig, ax, title, pad=6, ax_offset=3)

        self.save_figure()

    def _invariance_panel(self, ax):
        plot_df = self.invariance_df
        plot_df["stat"] = plot_df["stat"].map(
            {
                "dis_dirs_num_acc": "Flanker direction",
                "layouts_num_acc": "Layout",
                "xpos_bin_repr_acc": "Horizontal position",
                "ypos_bin_repr_acc": "Vertical position",
            }
        )
        plot_df = plot_df.rename(
            columns={
                "value": "Generalization performance",
                "stat": "Irrelevant dimension",
                "layer": "CNN layer",
            }
        )
        sns.lineplot(
            data=plot_df,
            x="CNN layer",
            y="Generalization performance",
            hue="Irrelevant dimension",
            ax=ax,
            err_style="bars",
            palette="colorblind",
            hue_order=self.hue_order,
        )
        ax.get_legend().remove()
        self.rotate_ax_labels(ax)
        ax.set_xlabel("")
        ax.set_ylabel("Generalization\nperformance")

    def _decoding_panel(self, ax):
        plot_df = self.decoding_df.query("decoded_variable in @self.stimulus_fts")
        plot_df = plot_df.replace(
            {
                "decoded_variable": {
                    "dis_dirs_num": "Flanker direction",
                    "layouts_num": "Layout",
                    "xpos_bin_repr": "Horizontal position",
                    "ypos_bin_repr": "Vertical position",
                }
            }
        )

        plot_df = plot_df.rename(
            columns={
                "decoder_accuracy": "Decoder performance",
                "decoded_variable": "Stimulus feature",
                "layer": "CNN layer",
            }
        )

        sns.lineplot(
            data=plot_df,
            x="CNN layer",
            y="Decoder performance",
            hue="Stimulus feature",
            ax=ax,
            err_style="bars",
            palette="colorblind",
            hue_order=self.hue_order,
        )
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=4,
            frameon=False,
        )
        self.rotate_ax_labels(ax)
        ax.set_ylabel("Decoder\nperformance")

    def _mutual_info_panel(self, ax):
        plot_df = self.mi_df.query("mi_variable in @self.stimulus_fts")
        plot_df = plot_df.rename(
            columns={
                "value": "Norm. mutual information",
                "mi_variable": "Stimulus feature",
                "layer": "CNN layer",
            }
        )
        plot_df = plot_df.replace(
            {
                "Stimulus feature": {
                    "dis_dirs_num": "Flanker direction",
                    "layouts_num": "Layout",
                    "xpos_bin_repr": "Horizontal position",
                    "ypos_bin_repr": "Vertical position",
                }
            }
        )

        sns.lineplot(
            data=plot_df,
            x="CNN layer",
            y="Norm. mutual information",
            hue="Stimulus feature",
            ax=ax,
            err_style="bars",
            palette="colorblind",
            hue_order=self.hue_order,
        )
        self.rotate_ax_labels(ax)
        ax.get_legend().remove()
        ax.set_xlabel("")
        ax.set_ylabel("Norm. mutual\ninformation")


class Figure5(BaseFigure, BasicAnalysisMixin, CorrelationMixin):
    """Create Figure 5 from the manuscript
    (orthogonality vs. congruency effect)
    """

    figsize = (6, 2.5)
    fig_str = "Figure5"

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )
        self.orthog_df = stats["subspace"].query("model_type == 'vam'")

        self.behavior_df = (
            stats["basic"]
            .query("model_type == 'vam' and model_user == 'model'")
            .drop(columns=["layer"])
        )

        self.behavior_df = self.behavior_df.rename(
            columns={"stat": "behavior_stat", "value": "behavior_value"}
        )

        self.orthog_df = self.orthog_df.rename(
            columns={"stat": "feature_stat", "value": "feature_value"}
        )

        self.orthog_merged_df = self.behavior_df.merge(
            self.orthog_df, on=["user_id", "model_type"], how="left"
        )

        self.behavior_stat = "acc_con_effect"
        self.orthog_stat = "principal_alignment"

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(12, 31)
        orthog_line_ax = fig.add_subplot(gs[:4, 3:13])
        orthog_bar_ax = fig.add_subplot(gs[8:, 3:13])
        sc_ax1 = fig.add_subplot(gs[1:5, 18:23])
        sc_ax2 = fig.add_subplot(gs[1:5, 26:31])
        sc_ax3 = fig.add_subplot(gs[7:11, 18:23])
        sc_ax4 = fig.add_subplot(gs[7:11, 26:31])

        self._lineplot_panel(
            self.orthog_df,
            orthog_line_ax,
            self.orthog_stat,
            "Target/flanker\nsubspace alignment",
        )

        self.pearson_bar_panel(
            self.orthog_merged_df,
            orthog_bar_ax,
            self.orthog_stat,
            self.behavior_stat,
            "",
        )
        orthog_bar_ax.set_ylabel("Pearson's $\it{r}$")
        orthog_bar_ax.set_xlabel("CNN layer")

        # Subspace alignment vs. acc con effect scatterplots
        for ax, layer in zip(
            [sc_ax1, sc_ax2, sc_ax3, sc_ax4], ["Conv4", "Conv5", "Conv6", "FC1"]
        ):
            self._orthog_scatter_panel(ax, layer)
            ax.set_xlabel("")
            ax.set_ylabel("")

        for ax, title in zip([orthog_line_ax, orthog_bar_ax, sc_ax1], ["A", "B", "C"]):
            self.add_title_ax(fig, ax, title, pad=6, ax_offset=3)
        fig.text(
            0.5,
            0.5,
            "Target/flanker\nsubspace alignment",
            ha="center",
            va="center",
            fontsize=6,
            rotation="vertical",
        )
        fig.text(
            0.73,
            0.05,
            "Accuracy congruency effect",
            ha="center",
            va="center",
            fontsize=6,
        )

        self.save_figure()

        return fig

    def _orthog_scatter_panel(self, ax, layer):
        plot_df = self.orthog_merged_df.query(
            "layer == @layer and feature_stat == @self.orthog_stat and behavior_stat == @self.behavior_stat"
        )
        b_vals, o_vals = plot_df["behavior_value"], plot_df["feature_value"]
        self.plot_scatter(
            b_vals,
            o_vals,
            "Accuracy congruency effect",
            "Subspace alignment",
            "",
            ax,
            plot_unity=False,
        )
        ax.set_title(f"{layer}", pad=2, fontsize=6)


class Figure6(BaseFigure, BasicAnalysisMixin):
    """Create Figure 6 from the manuscript
    (analysis of task-optimized CNNs)
    """

    figsize = (7, 3)
    fig_str = "Figure6"

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )
        self.lba_params_df = (
            stats["lba_params"]
            .query("model_type == 'task_opt'")
            .merge(metadata, on=["user_id"], how="left")
        )
        congruency_map = {0: "Congruent", 1: "Incongruent"}
        self.lba_params_df["congruency"] = self.lba_params_df["congruency"].replace(
            congruency_map
        )

        self.decoding_df = stats["decoding"].query("model_type in ['vam', 'task_opt']")
        self.decoding_df = self.decoding_df.rename(
            columns={
                "decoded_variable": "feature_stat",
                "decoder_accuracy": "feature_value",
            }
        )

        self.dim_df = stats["dimensionality"].query("model_type in ['vam', 'task_opt']")

        self.behavior_df = stats["basic"].query(
            "stat == 'acc_con_effect' and model_user == 'model' and model_type in ['vam', 'task_opt']"
        )

        self.orthog_df = stats["subspace"].query("model_type in ['vam', 'task_opt']")

        self.mi_df = stats["mutual_info"].query(
            "model_type in ['vam', 'task_opt'] and stat == 'mi_mean'"
        )
        self.mi_df = self.mi_df.rename(
            columns={
                "mi_variable": "feature_stat",
                "value": "feature_value",
            }
        )

        self.selectivity_df = stats["basic_modulation"].query(
            "stimulus_feature == 'targ_dirs_num'"
        )
        self.selectivity_vars = ["f_pos", "f_neg", "f_complex", "f_not_signif"]

        for df in [
            self.decoding_df,
            self.dim_df,
            self.behavior_df,
            self.orthog_df,
            self.selectivity_df,
            self.mi_df,
        ]:
            df["model_type"] = df["model_type"].map(
                {"vam": "VAM", "task_opt": "Task-opt."}
            )

        for df in [self.dim_df, self.orthog_df]:
            df.rename(
                columns={
                    "stat": "feature_stat",
                    "value": "feature_value",
                },
                inplace=True,
            )

        self.behavior_stat = "acc_con_effect"
        self.decoding_mi_stats = ["dis_dirs_num", "targ_dirs_num"]
        self.orthog_stat = "principal_alignment"
        self.dim_stat = "centered_participation_ratio"
        self.unit_layers = ["Conv5", "Conv6"]

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(12, 46)

        ce_ax = fig.add_subplot(gs[:4, 3:7])
        logit_ax = fig.add_subplot(gs[:4, 12:16])
        orthog_ax = fig.add_subplot(gs[:4, 21:31])
        dim_ax = fig.add_subplot(gs[:4, 36:46])
        mi_ax = fig.add_subplot(gs[8:, 3:13])
        decoding_ax = fig.add_subplot(gs[8:, 19:29])
        unit_ax1 = fig.add_subplot(gs[8:, 35:39])
        unit_ax2 = fig.add_subplot(gs[8:, 42:46])

        for ax, title in zip(
            [ce_ax, logit_ax, orthog_ax, dim_ax, mi_ax, decoding_ax, unit_ax1],
            ["A", "B", "C", "D", "E", "F", "G"],
        ):
            self.add_title_ax(fig, ax, title, pad=6, ax_offset=3)

        sns.barplot(
            data=self.behavior_df,
            x="model_type",
            y="value",
            ax=ce_ax,
            palette="colorblind",
        )
        ce_ax.set_xlabel("")
        ce_ax.set_ylabel("Accuracy\ncongruency effect")
        ce_ax.set_xlim([-1, 2])
        self.rotate_ax_labels(ce_ax)

        self._logit_panel(logit_ax)

        self._lineplot_panel(
            self.orthog_df,
            orthog_ax,
            self.orthog_stat,
            "Target/flanker\nsubspace alignment",
            plot_model_type=True,
        )

        self._lineplot_panel(
            self.dim_df,
            dim_ax,
            self.dim_stat,
            "Participation ratio",
            plot_model_type=True,
        )

        self._lineplot_panel(
            self.mi_df,
            mi_ax,
            self.decoding_mi_stats,
            "Norm. mutual\ninformation",
            plot_model_type=True,
        )

        self._lineplot_panel(
            self.decoding_df,
            decoding_ax,
            self.decoding_mi_stats,
            "Decoder\nperformance",
            plot_model_type=True,
        )

        self._unit_panel(self.selectivity_df, [unit_ax1, unit_ax2], self.unit_layers)

        # Adjust
        for ax in [orthog_ax, dim_ax]:
            sns.move_legend(
                ax,
                "upper left",
                bbox_to_anchor=(-0.05, 1.25),
                ncol=2,
                title=None,
                frameon=False,
            )

        for ax in [mi_ax, decoding_ax]:
            h, l = ax.get_legend_handles_labels()
            l[4] = "Target dir."
            l[5] = "Flanker dir."
            h = [h[1], h[2], h[4], h[5]]
            l = [l[1], l[2], l[4], l[5]]
            ax.legend(h, l, ncol=2, loc="upper left")
            sns.move_legend(
                ax,
                "upper left",
                bbox_to_anchor=(-0.1, 1.4),
                ncol=2,
                title=None,
                frameon=False,
            )

        self.save_figure()

        return fig

    def _logit_panel(self, ax):
        # Panel F: Drift rate summary
        df = self.lba_params_df.query("stat in ['target_logit']")
        df = df.replace(
            {
                "stat": {
                    "target_logit": "Target",
                }
            }
        )

        F = sns.barplot(
            data=df,
            x="congruency",
            y="value",
            palette="viridis",
            ax=ax,
        )
        ax.set_ylabel("Target logit (a.u.)")
        ax.set_xlabel("")
        self.rotate_ax_labels(ax)
        ax.set_xlim([-1, 2])

        self._logit_panel_stats(df)

    def _logit_panel_stats(self, df):
        with open(self.stats_file, "a") as f:
            f.write(
                (
                    "Signed-rank test for difference in target logit on"
                    "congruent vs. incongruent trials (task-optimized models):\n"
                )
            )
        con_df = df.query("stat == 'Target' and congruency == 'Congruent'").sort_values(
            by=["user_id"]
        )
        incon_df = df.query(
            "stat == 'Target' and congruency == 'Incongruent'"
        ).sort_values(by=["user_id"])
        con_logits = con_df["value"].values
        incon_logits = incon_df["value"].values

        wstat, p = wilcoxon(con_logits, incon_logits)
        with open(self.stats_file, "a") as f:
            f.write(f"w = {wstat:.3f}, p = {p:.3f}\n")

    def _unit_panel(self, df, axes, layers):
        for ax, layer in zip(axes, layers):
            plot_df = df.query("layer == @layer and stat in @self.selectivity_vars")
            plot_df = plot_df.rename(
                columns={
                    "value": "Proportion of units",
                    "stat": "Unit type",
                    "layer": "CNN layer",
                }
            )
            plot_df = plot_df.replace(
                {
                    "Unit type": {
                        "f_not_signif": "No modulation",
                        "f_pos": "Selective (+)",
                        "f_neg": "Selective (-)",
                        "f_complex": "Complex",
                    }
                }
            )
            sns.barplot(
                data=plot_df,
                x="model_type",
                y="Proportion of units",
                hue="Unit type",
                ax=ax,
                palette="viridis",
            )
            self.rotate_ax_labels(ax)
            ax.set_xlabel("")
            ax.set_title(layer, fontsize=6, pad=2)
            ax.set_ylim([0, 0.8])

        axes[0].set_ylabel("Proportion\nof units")
        axes[1].set_ylabel("")
        sns.move_legend(
            axes[0],
            "upper left",
            bbox_to_anchor=(-0.5, 1.55),
            ncol=2,
            title=None,
            frameon=False,
        )
        axes[1].get_legend().remove()

        self._unit_panel_stats(df, layers)

    def _unit_panel_stats(self, df, layers):
        # Compare selective + and complex proportions between
        # VAMs / task-optimized models
        with open(self.stats_file, "a") as f:
            f.write(
                (
                    "Signed-rank test for difference in proportion of units, "
                    "VAMs vs. task-optimized models\n"
                )
            )
        for layer in layers:
            for unit_type in ["f_pos", "f_complex"]:
                this_df = df.query("layer == @layer and stat == @unit_type")
                vam_vals = this_df.query("model_type == 'VAM'")["value"].values
                task_opt_vals = this_df.query("model_type == 'Task-opt.'")[
                    "value"
                ].values

                wstat, p = wilcoxon(vam_vals, task_opt_vals)
                with open(self.stats_file, "a") as f:
                    f.write(f"{layer}, {unit_type}: w = {wstat:.3f}, p = {p:.3f}\n")


class FigureS1(BaseFigure):
    # Example model/participant RT distributions
    figsize = (7, 6)
    fig_str = "FigureS1"

    # Exemplars
    rt_dist_examples = [6786, 6519, 5573, 6609, 7109]
    layout_examples = [677, 6946, 4275, 2090, 3471]
    horizontal_examples = [5573, 3875, 5675, 1354, 2683]
    vertical_examples = [3497, 1127, 7735, 5383, 5256]
    all_example_users = [
        *rt_dist_examples,
        *layout_examples,
        *horizontal_examples,
        *vertical_examples,
    ]
    layout_order = ["---", "|", "+", "<", ">", "v", "^"]
    xpos_bin_order = [
        "(-210, -150]",
        "(-150, -100]",
        "(-100, -50]",
        "(-50, 0]",
        "(0, 50]",
        "(50, 100]",
        "(100, 150]",
        "(150, 210]",
    ]
    ypos_bin_order = [
        "(-75, -50]",
        "(-50, -25]",
        "(-25, 0]",
        "(0, 25]",
        "(25, 50]",
        "(50, 75]",
    ]

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )

        self.example_dfs = []
        self.layout_pos_stats = stats["layout_pos_summary"]

        for stim_ft, ex_ids in zip(
            ["layouts", "xpos_bin_behavior", "ypos_bin_behavior"],
            [self.layout_examples, self.horizontal_examples, self.vertical_examples],
        ):
            sig_df = self.layout_pos_stats.query(
                "stimulus_feature == @stim_ft & user_effect_p < 0.05 & stat == 'mean_rt'"
            )
            for uid in ex_ids:
                assert uid in sig_df["user_id"].unique(), f"user{uid} not significant"

        for user_id in [*self.rt_dist_examples]:
            df = pd.read_csv(
                os.path.join(self.derivatives_dir, f"vam/user{user_id}/trial_df.csv")
            )
            df = self.update_df(df, correct_only=False)
            self.example_dfs.append(df)

        for user_id in [
            *self.layout_examples,
            *self.horizontal_examples,
            *self.vertical_examples,
        ]:
            df = pd.read_csv(
                os.path.join(self.derivatives_dir, f"vam/user{user_id}/trial_df.csv")
            )
            df = self.update_df(df, correct_only=True)
            self.example_dfs.append(df)

    def update_df(self, df, correct_only=True):
        # Remap layouts
        layout_map = {0: "---", 1: "|", 2: "+", 3: "<", 4: ">", 5: "v", 6: "^"}
        df["layouts"] = df["layouts"].replace(layout_map)

        # Set spatial
        xpos_bins = [-210, -150, -100, -50, 0, 50, 100, 150, 210]
        ypos_bins = np.arange(-75, 76, 25)
        df["xpos_bin_behavior"] = pd.cut(df["xpositions"], xpos_bins).astype(str)
        df["ypos_bin_behavior"] = pd.cut(df["ypositions"], ypos_bins).astype(str)

        if correct_only:
            df = df.query("response_dirs == targ_dirs")

        entity_map = {"model": "Model", "user": "Participant"}
        df["model_user"] = df["model_user"].replace(entity_map)

        return df

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(44, 47)

        # User vs Model RT distribution
        A_ax1 = fig.add_subplot(gs[0:6, 1:7])
        A_ax2 = fig.add_subplot(gs[0:6, 11:17])
        A_ax3 = fig.add_subplot(gs[0:6, 21:27])
        A_ax4 = fig.add_subplot(gs[0:6, 31:37])
        A_ax5 = fig.add_subplot(gs[0:6, 41:47])
        # Layout User/Model examples
        B_ax1 = fig.add_subplot(gs[12:18, 1:7])
        B_ax2 = fig.add_subplot(gs[12:18, 11:17])
        B_ax3 = fig.add_subplot(gs[12:18, 21:27])
        B_ax4 = fig.add_subplot(gs[12:18, 31:37])
        B_ax5 = fig.add_subplot(gs[12:18, 41:47])
        # Horizontal User/Model examples
        C_ax1 = fig.add_subplot(gs[24:30, 1:7])
        C_ax2 = fig.add_subplot(gs[24:30, 11:17])
        C_ax3 = fig.add_subplot(gs[24:30, 21:27])
        C_ax4 = fig.add_subplot(gs[24:30, 31:37])
        C_ax5 = fig.add_subplot(gs[24:30, 41:47])
        # Vertical User/Model examples
        D_ax1 = fig.add_subplot(gs[38:44, 1:7])
        D_ax2 = fig.add_subplot(gs[38:44, 11:17])
        D_ax3 = fig.add_subplot(gs[38:44, 21:27])
        D_ax4 = fig.add_subplot(gs[38:44, 31:37])
        D_ax5 = fig.add_subplot(gs[38:44, 41:47])

        for ax, title in zip([A_ax1, B_ax1, C_ax1, D_ax1], ["A", "B", "C", "D"]):
            self.add_title_ax(fig, ax, title, pad=3)

        # Examples RT distribution
        axs_list = [A_ax1, A_ax2, A_ax3, A_ax4, A_ax5]

        for df_idx, ax in zip(np.arange(5), axs_list):
            rt_example_df = self.example_dfs[df_idx]
            A = sns.kdeplot(data=rt_example_df, x="rts", hue="model_user", ax=ax)
            ax.set_xlim([0, 2])
            if ax not in [A_ax1]:
                A.get_legend().remove()
            else:
                sns.move_legend(
                    ax,
                    "upper left",
                    bbox_to_anchor=(0, 1.5),
                    frameon=False,
                    title=None,
                    ncol=1,
                )
            ax.set_xlabel("")
            ax.set_ylabel("")

            axs_list[2].set_xlabel("RT (s)")
            axs_list[0].set_ylabel("Density")

        axs_list = [
            B_ax1,
            B_ax2,
            B_ax3,
            B_ax4,
            B_ax5,
            C_ax1,
            C_ax2,
            C_ax3,
            C_ax4,
            C_ax5,
            D_ax1,
            D_ax2,
            D_ax3,
            D_ax4,
            D_ax5,
        ]
        x_df_names = [
            *5 * ["layouts"],
            *5 * ["xpos_bin_behavior"],
            *5 * ["ypos_bin_behavior"],
        ]
        xlabels = [
            *5 * ["Layout"],
            *5 * ["Horizontal position (pixels)"],
            *5 * ["Vertical position (pixels)"],
        ]
        orders = (
            [self.layout_order for i in range(5)]
            + [self.xpos_bin_order for i in range(5)]
            + [self.ypos_bin_order for i in range(5)]
        )

        for df_idx, ax, x_name, xlabel, order in zip(
            np.arange(5, 20), axs_list, x_df_names, xlabels, orders
        ):
            trial_df = self.example_dfs[df_idx]
            user_id = self.all_example_users[df_idx]

            sns.barplot(
                data=trial_df,
                x=x_name,
                y="rts",
                hue="model_user",
                ax=ax,
                order=order,
            )

            pearsonr_df = self.layout_pos_stats.query(
                "stimulus_feature==@x_name & stat=='mean_rt' & user_id==@user_id"
            )
            pearsonr_value = np.round(pearsonr_df["pearsons_r"].values[0], 2)
            r_str = "$\it{r}$"
            ax.text(
                0.02,
                0.98,
                f"{r_str}={pearsonr_value}",
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=6,
            )

            ax.set_xlabel("")
            ax.set_ylabel("")

            mean_rts = trial_df.groupby([f"{x_name}"])["rts"].mean()
            sem_rts = trial_df.groupby([f"{x_name}"])["rts"].sem()
            upper_y = np.max(mean_rts + sem_rts)
            lower_y = np.min(mean_rts - sem_rts)

            if x_name == "ypos_bin_behavior":
                ax.set_ylim(lower_y - 0.02, upper_y + 0.04)
            else:
                ax.set_ylim(lower_y - 0.035, upper_y + 0.06)

            if ax not in [B_ax1, C_ax1, D_ax1]:
                ax.get_legend().remove()
            else:
                sns.move_legend(
                    ax,
                    "upper left",
                    bbox_to_anchor=(0, 1.5),
                    ncol=1,
                    frameon=False,
                    title=None,
                )

            if x_name in ["xpos_bin_behavior", "ypos_bin_behavior"]:
                self.rotate_ax_labels(ax)

        B_ax1.set_ylabel("Mean RT (s)")
        B_ax3.set_xlabel("Layout")
        C_ax1.set_ylabel("Mean RT (s)")
        C_ax3.set_xlabel("Horizontal position (pixels)")
        D_ax1.set_ylabel("Mean RT (s)")
        D_ax3.set_xlabel("Vertical position (pixels)")

        self.save_figure()

        return fig


class FigureS2(BaseFigure):
    # Age-dependence of LBA parameters
    figsize = (7, 1.3)
    fig_str = "FigureS2"
    age_bin_order = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-89"]

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )

        # Combine 70-79 and 80-89 age bins
        age_remap = {"70-79": "70-89", "80-89": "70-89"}
        metadata = metadata.replace({"binned_age": age_remap})
        self.lba_params_df = (
            stats["lba_params"]
            .query("model_type == 'vam'")
            .merge(metadata, on=["user_id"], how="left")
        )

    def _lba_param_age_stats(self, df, param, n_comp, do_bonferonni=True):
        data = [g["value"].values for _, g in df.groupby("binned_age")]
        f_stat, p = f_oneway(*data)
        if do_bonferonni:
            p = p * n_comp

        stats = {"lba_param": param, "f_stat": f"{f_stat:0.3f}", "p": p}
        stats = pd.DataFrame(stats, index=[0])
        stats.to_csv(self.stats_file, sep="\t", mode="a")

        # Post-hoc comparisons
        if p < 0.05:
            tukey_res = pairwise_tukeyhsd(df["value"].values, df["binned_age"].values)
            p_20_vs_70 = tukey_res.pvalues[4]
            with open(self.stats_file, "a") as f:
                f.write(tukey_res.summary().as_text())
                f.write(f"\n Age 20-29 vs. 70-89 adjusted p-value: {p_20_vs_70}\n")
                f.write("--------------------------------------------------------\n")

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(6, 35)

        col_idx = 3
        n_cols = 5
        spacer = 4

        params = ["t0", "response_caution", "target_drift", "flanker_drift"]
        ylabels = [
            "Non-decision time (t0)",
            "Response caution (b-A)",
            "Target drift rate",
            "Flanker drift rate",
        ]
        panels = ["A", "B", "C", "D"]
        n_comp = len(params)
        for param, ylabel, panel in zip(params, ylabels, panels):
            ax = fig.add_subplot(gs[:, col_idx : col_idx + n_cols])
            param_df = self.lba_params_df.query("stat == @param")
            sns.barplot(
                data=param_df,
                x="binned_age",
                y="value",
                order=self.age_bin_order,
                ax=ax,
                palette="viridis",
            )
            self._lba_param_age_stats(param_df, param, n_comp, do_bonferonni=True)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("")
            self.rotate_ax_labels(ax)
            self.add_title_ax(fig, ax, panel, pad=6, ax_offset=3)

            col_idx += n_cols + spacer

        fig.text(
            0.55,
            -0.2,
            "Age bin (years)",
            ha="center",
            va="center",
            fontsize=6,
            rotation="horizontal",
        )

        self.save_figure()

        return fig


class FigureS3(BaseFigure):
    # Average layout/position mean RTs across users
    # Should refactor since each panel is similar
    figsize = (7, 3.5)
    fig_str = "FigureS3"

    layout_order = ["---", "|", "+", "<", ">", "v", "^"]

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )

        self.layout_pos_mean = stats["layout_pos_mean"]
        entity_map = {"model": "Model", "user": "Participant"}
        self.layout_pos_mean["model_user"] = self.layout_pos_mean["model_user"].replace(
            entity_map
        )
        self.layout_pos_summary = stats["layout_pos_summary"].query("stat == 'mean_rt'")

    def _figS3_stats(self, df, feature, n_comp, do_bonferonni=True):
        mps = ["Model", "Participant"]
        for mp in mps:
            mpdf = df.query("model_user == @mp")
            data = [g["value"].values for _, g in mpdf.groupby("feature_value")]
            f_stat, p = f_oneway(*data)
            n_models = len(mpdf["user_id"].unique())
            n_ft_vals = len(mpdf["feature_value"].unique())
            df1 = n_ft_vals - 1
            df2 = n_models - n_ft_vals
            if do_bonferonni:
                p = p * n_comp

            tukey_res = pairwise_tukeyhsd(
                mpdf["value"].values, mpdf["feature_value"].values
            )

            summary = mpdf.groupby(["feature_value"])["value"].describe()
            summary["s.e.m."] = summary["std"] / np.sqrt(n_models)
            summary = summary.applymap(lambda x: f"{x:0.3f}")
            summary.drop(columns=["25%", "75%"], inplace=True)

            with open(self.stats_file, "a") as f:
                f.write(
                    (
                        f"Stats for {mp} mean centered RT vs {feature}:\n"
                        # f"F({df1},{df2})={f_stat:0.3f}, p={p:0.3f}\n")
                        f"F({df1},{df2})={f_stat:0.3f}, p={p}\n"
                    )
                )
                print(tukey_res, file=f)
                f.write(f"Bin mean/s.e.m./etc. for each feature value\n")
                print(summary, file=f)
                f.write("--------------------------------------------------------\n")

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(10, 23)

        # Panel A: layout
        A_ax = fig.add_subplot(gs[:, 1:7])
        layout_rt_sig_df = self.layout_pos_summary.query(
            "stimulus_feature == 'layouts' & user_effect_p < 0.05"
        )
        layout_sig_users = layout_rt_sig_df["user_id"].unique()
        layout_df = self.layout_pos_mean.query(
            "stimulus_feature == 'layouts' & stat == 'mean_rt_centered' & user_id in @layout_sig_users"
        )
        sns.boxplot(
            data=layout_df,
            x="feature_value",
            y="value",
            hue="model_user",
            ax=A_ax,
            order=self.layout_order,
            showfliers=False,
        )
        A_ax.plot([-1, 7], [0, 0], "k--", linewidth=0.5)
        A_ax.set_xlim(-1, 7)
        A_ax.set_ylabel("Mean subtracted RT (s)")
        A_ax.set_xlabel("Layout")

        self._figS3_stats(layout_df, "layout", 3)

        # Panel B: horizontal position
        B_ax = fig.add_subplot(gs[:, 9:15])
        xpos_rt_sig_df = self.layout_pos_summary.query(
            "stimulus_feature == 'xpos_bin_behavior' & user_effect_p < 0.05"
        )
        xpos_sig_users = xpos_rt_sig_df["user_id"].unique()
        xpos_df = self.layout_pos_mean.query(
            "stimulus_feature == 'xpos_bin_behavior' & stat == 'mean_rt_centered' and user_id in @xpos_sig_users"
        )
        sns.boxplot(
            data=xpos_df,
            x="feature_value",
            y="value",
            hue="model_user",
            ax=B_ax,
            showfliers=False,
            # order=self.xpos_bin_order,
        )
        B_ax.plot([-1, 8], [0, 0], "k--", linewidth=0.5)
        B_ax.set_xlim(-1, 8)
        B_ax.set_ylabel("")
        B_ax.set_xlabel("Horizontal position\n(pixels)")

        self._figS3_stats(xpos_df, "horizontal position", 3)

        # Panel C: vertical position
        C_ax = fig.add_subplot(gs[:, 17:])
        ypos_rt_sig_df = self.layout_pos_summary.query(
            "stimulus_feature == 'ypos_bin_behavior' & user_effect_p < 0.05"
        )
        ypos_sig_users = ypos_rt_sig_df["user_id"].unique()
        ypos_df = self.layout_pos_mean.query(
            "stimulus_feature == 'ypos_bin_behavior' & stat == 'mean_rt_centered' and user_id in @xpos_sig_users"
        )
        # sns.barplot(
        sns.boxplot(
            data=ypos_df,
            x="feature_value",
            y="value",
            hue="model_user",
            ax=C_ax,
            showfliers=False,
            # order=self.ypos_bin_order,
        )
        C_ax.plot([-1, 6], [0, 0], "k--", linewidth=0.5)
        C_ax.set_xlim(-1, 6)
        C_ax.set_ylabel("")
        C_ax.set_xlabel("Vertical position\n(pixels)")

        self._figS3_stats(ypos_df, "vertical position", 3)

        for ax, title in zip([A_ax, B_ax, C_ax], ["A", "B", "C"]):
            self.add_title_ax(fig, ax, title, pad=3)
            ax.get_legend().set_title(None)
            ax.get_legend().get_frame().set_alpha(0)

            if ax in [B_ax, C_ax]:
                self.rotate_ax_labels(ax)

        self.save_figure()

        return fig


class FigureS4(BaseFigure, DeltaPlotCAFMixin):
    # CAF/Delta plot analysis
    figsize = (7, 2)
    fig_str = "FigureS4"

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )
        self.delta_df = stats["delta_plot"]
        self.caf_df = stats["caf"]

        entity_map = {"model": "Model", "user": "Participant"}
        congruency_map = {"congruent": "Congruent", "incongruent": "Incongruent"}

        for df in [self.delta_df, self.caf_df]:
            df["model_user"] = df["model_user"].replace(entity_map)

        self.caf_df["congruency"] = self.caf_df["congruency"].replace(congruency_map)

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(6, 15)
        A_ax = fig.add_subplot(gs[:, 1:7])
        B_ax = fig.add_subplot(gs[:, 9:15])

        for ax, title in zip([A_ax, B_ax], ["A", "B"]):
            self.add_title_ax(fig, ax, title, pad=3, ax_offset=1)

        self._plot_delta(A_ax, self.delta_df.query("model_type == 'vam'"))
        self._plot_caf(B_ax, self.caf_df.query("model_type == 'vam'"))

        self.save_figure()

        return fig

    def _plot_delta(self, ax, plot_df):
        # Average RTs in each decile
        model_df = plot_df.query("model_user == 'Model'")
        user_df = plot_df.query("model_user == 'Participant'")
        for df in [model_df, user_df]:
            df["across_user_rt"] = df.groupby("decile_idx")["avg_rt"].transform("mean")
        ax = self.plot_delta(pd.concat([model_df, user_df]), "across_user_rt", ax=ax)
        sns.move_legend(ax, "lower right", ncol=2)
        ax.get_legend().get_frame().set_alpha(0)
        ax.get_legend().set_title(None)

    def _plot_caf(self, ax, plot_df):
        model_con_df = plot_df.query(
            "model_user == 'Model' and congruency == 'Congruent'"
        )
        model_incon_df = plot_df.query(
            "model_user == 'Model' and congruency == 'Incongruent'"
        )
        user_con_df = plot_df.query(
            "model_user == 'Participant' and congruency == 'Congruent'"
        )
        user_incon_df = plot_df.query(
            "model_user == 'Participant' and congruency == 'Incongruent'"
        )
        for df in [model_con_df, model_incon_df, user_con_df, user_incon_df]:
            df["across_user_rt"] = df.groupby("decile_idx")["rt"].transform("mean")
        plot_df = pd.concat([model_con_df, model_incon_df, user_con_df, user_incon_df])
        ax = self.plot_caf(plot_df, "across_user_rt", ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[i] for i in [1, 2, 4, 5]]
        labels = [labels[i] for i in [1, 2, 4, 5]]
        ax.legend(handles, labels, loc="lower right", ncol=2)
        ax.get_legend().get_frame().set_alpha(0)


class FigureS5(BaseFigure, SingleUnitMixin):
    # Example unit activation
    figsize = (6, 1.5)
    fig_str = "FigureS5"
    activation_example_id = 677

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )

        self.heatmap_df = pd.read_csv(
            os.path.join(
                derivatives_dir,
                f"{summary_dir}/user{self.activation_example_id}/activation_df.csv",
            )
        )
        self.heatmap_layers = ["Conv2", "Conv3", "Conv4", "Conv5", "Conv6", "FC1"]

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(4, 68)

        hmap_ax1 = fig.add_subplot(gs[:, 1:11])
        hmap_ax2 = fig.add_subplot(gs[:, 12:22])
        hmap_ax3 = fig.add_subplot(gs[:, 23:33])
        hmap_ax4 = fig.add_subplot(gs[:, 34:44])
        hmap_ax5 = fig.add_subplot(gs[:, 45:55])
        hmap_ax6 = fig.add_subplot(gs[:, 56:66])
        cbar_ax = fig.add_subplot(gs[:, 67:68])
        hmap_axes = [
            hmap_ax1,
            hmap_ax2,
            hmap_ax3,
            hmap_ax4,
            hmap_ax5,
            hmap_ax6,
            cbar_ax,
        ]

        self._heatmap_panels(hmap_axes, self.heatmap_layers)

        self.save_figure()

        return fig

    def _heatmap_panels(self, axes, layers):
        axes = self.plot_activation_heatmap(
            self.heatmap_df,
            layers,
            axes=axes,
        )


class FigureS6(BaseFigure, BasicAnalysisMixin, CorrelationMixin):
    """Suppression vs. congruency effects."""

    figsize = (4.5, 3.5)
    fig_str = "FigureS6"

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )
        self.decoding_df = stats["decoding"].query("model_type == 'vam'")
        self.decoding_df = self.decoding_df.rename(
            columns={
                "decoded_variable": "feature_stat",
                "decoder_accuracy": "feature_value",
            }
        )

        self.mi_df = stats["mutual_info"].query("model_type == 'vam'")
        self.mi_df = self.mi_df.rename(
            columns={"mi_variable": "feature_stat", "value": "feature_value"}
        )

        self.behavior_df = (
            stats["basic"]
            .query("model_type == 'vam' and model_user == 'model'")
            .drop(columns=["layer"])
        )

        self.behavior_df = self.behavior_df.rename(
            columns={"stat": "behavior_stat", "value": "behavior_value"}
        )

        self.decoding_merged_df = self.behavior_df.merge(
            self.decoding_df, on=["user_id", "model_type"], how="left"
        )
        self.mi_merged_df = self.behavior_df.merge(
            self.mi_df, on=["user_id", "model_type"], how="left"
        )

        self.decoding_stat = "dis_dirs_num"
        self.mi_stat = "dis_dirs_num"

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(17, 18)
        acc_decoding_ax = fig.add_subplot(gs[:6, 2:8])
        acc_mi_ax = fig.add_subplot(gs[:6, 12:])
        rt_decoding_ax = fig.add_subplot(gs[11:, 2:8])
        rt_mi_ax = fig.add_subplot(gs[11:, 12:])

        # Correlation bar plots
        corr_axes = [acc_decoding_ax, acc_mi_ax, rt_decoding_ax, rt_mi_ax]
        corr_titles = [
            "Flanker decoding acc. vs.\nacc. congruency effect",
            "Flanker mutual information vs.\nacc. congruency effect",
            "Flanker decoding acc. vs.\nRT congruency effect",
            "Flanker mutual information vs.\nRT congruency effect",
        ]
        corr_feature_stats = [
            self.decoding_stat,
            self.mi_stat,
            self.decoding_stat,
            self.mi_stat,
        ]
        corr_behavior_stats = [
            "acc_con_effect",
            "acc_con_effect",
            "rt_con_effect",
            "rt_con_effect",
        ]
        corr_dfs = [
            self.decoding_merged_df,
            self.mi_merged_df,
            self.decoding_merged_df,
            self.mi_merged_df,
        ]

        for df, ax, title, bstat, fstat in zip(
            corr_dfs,
            corr_axes,
            corr_titles,
            corr_behavior_stats,
            corr_feature_stats,
        ):
            self.pearson_bar_panel(df, ax, fstat, bstat, title)

        # Adjust
        # for ax, title in zip(corr_axes, ["A", "B", "C", "D"]):
        #    self.add_title_ax(fig, ax, title, pad=5, ax_offset=2)

        for ax in [acc_decoding_ax, rt_decoding_ax]:
            ax.set_ylabel("Pearson's r")
        for ax in [rt_decoding_ax, rt_mi_ax]:
            ax.set_xlabel("CNN layer")

        self.save_figure()

        return fig


class FigureS7(BaseFigure, BasicAnalysisMixin, CorrelationMixin):
    """Target/flanker subspace alignment vs. RT congruency effect."""

    figsize = (2.5, 2)
    fig_str = "FigureS7"

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )
        self.orthog_df = stats["subspace"].query("model_type == 'vam'")
        self.orthog_df = self.orthog_df.rename(
            columns={"stat": "feature_stat", "value": "feature_value"}
        )

        self.behavior_df = (
            stats["basic"]
            .query("model_type == 'vam' and model_user == 'model'")
            .drop(columns=["layer"])
        )
        self.behavior_df = self.behavior_df.rename(
            columns={"stat": "behavior_stat", "value": "behavior_value"}
        )

        self.orthog_merged_df = self.behavior_df.merge(
            self.orthog_df, on=["user_id", "model_type"], how="left"
        )

        self.behavior_stat = "rt_con_effect"
        self.orthog_stat = "principal_alignment"

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(4, 4)
        ax = fig.add_subplot(gs[:, :])

        self.pearson_bar_panel(
            self.orthog_merged_df,
            ax,
            self.orthog_stat,
            self.behavior_stat,
            "Target/flanker subspace alignment vs.\nRT congruency effect",
        )

        ax.set_xlabel("CNN layer")
        ax.set_ylabel("Pearson's r")

        self.save_figure()

        return fig


class FigureS8(BaseFigure, BasicAnalysisMixin):
    """Additional analysis of VAMs vs. task-optimized models"""

    figsize = (7, 1.5)
    fig_str = "FigureS8"

    def __init__(
        self,
        stats,
        derivatives_dir,
        metadata,
        config,
        seed=None,
        n_boot=None,
        summary_dir=None,
    ):
        super().__init__(
            stats, derivatives_dir, metadata, config, seed, n_boot, summary_dir
        )
        self.decoding_df = stats["decoding"].query("model_type in ['vam', 'task_opt']")
        self.decoding_df = self.decoding_df.rename(
            columns={
                "decoded_variable": "feature_stat",
                "decoder_accuracy": "feature_value",
            }
        )

        self.mi_df = stats["mutual_info"].query(
            "model_type in ['vam', 'task_opt'] and stat == 'mi_mean'"
        )
        self.mi_df = self.mi_df.rename(
            columns={
                "mi_variable": "feature_stat",
                "value": "feature_value",
            }
        )

        for df in [self.decoding_df, self.mi_df]:
            df["model_type"] = df["model_type"].map(
                {"vam": "VAM", "task_opt": "Task-opt."}
            )

        self.decoding_mi_stats = ["layouts_num", "xpos_bin_repr", "ypos_bin_repr"]

    def make_figure(self):
        fig = plt.figure(
            constrained_layout=False, figsize=self.figsize, dpi=self.figdpi
        )
        gs = fig.add_gridspec(4, 28)

        mi_ax = fig.add_subplot(gs[:, 3:13])
        decoding_ax = fig.add_subplot(gs[:, 18:])

        for ax, title in zip([mi_ax, decoding_ax], ["A", "B"]):
            self.add_title_ax(fig, ax, title, pad=6, ax_offset=3)

        self._lineplot_panel(
            self.mi_df,
            mi_ax,
            self.decoding_mi_stats,
            "Norm. mutual\ninformation",
            plot_model_type=True,
        )

        self._lineplot_panel(
            self.decoding_df,
            decoding_ax,
            self.decoding_mi_stats,
            "Decoder\nperformance",
            plot_model_type=True,
        )

        h, l = mi_ax.get_legend_handles_labels()
        l[4] = "Layout"
        l[5] = "Horizontal pos."
        l[6] = "Vertical pos."
        h = [h[1], h[2], h[4], h[5], h[6]]
        l = [l[1], l[2], l[4], l[5], l[6]]
        mi_ax.legend(h, l, ncol=5, loc="upper left")
        sns.move_legend(
            mi_ax,
            "upper left",
            bbox_to_anchor=(0.3, 1.35),
            ncol=5,
            title=None,
            frameon=False,
        )

        decoding_ax.get_legend().remove()
        for ax in [mi_ax, decoding_ax]:
            ax.set_xlabel("CNN layer")

        self.save_figure()

        return fig

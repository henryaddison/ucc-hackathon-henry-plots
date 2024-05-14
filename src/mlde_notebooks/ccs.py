from mlde_utils import cp_model_rotated_pole, TIME_PERIODS
import numpy as np
import string
import xarray as xr
import xskillscore as xs

from . import plot_map
from .distribution import plot_freq_density


def plot_tp_fd(pred_pr, cpm_pr, fig, source, model, spec):

    axd = fig.subplot_mosaic(
        np.array(["Emulator"] + list(TIME_PERIODS.keys())).reshape(2, 2),
        sharey=True,
        sharex=True,
    )

    for idx, ax in enumerate(fig.axes):
        ax.annotate(
            f"{string.ascii_lowercase[idx]}.",
            xy=(-0.04, 1.02),
            xycoords=("axes fraction", "axes fraction"),
            weight="bold",
            ha="left",
            va="bottom",
        )

    historical_cpm_pr = cpm_pr.where(cpm_pr["time_period"] == "historic", drop=True)

    for tp_idx, tp_key in enumerate(TIME_PERIODS.keys()):
        tp_cpm_pr = cpm_pr.where(cpm_pr["time_period"] == tp_key, drop=True)
        tp_hist_data = [
            dict(data=tp_cpm_pr, label=f"CPM", color="black", source="CPM"),
            dict(
                data=pred_pr.where(pred_pr["time_period"] == tp_key, drop=True),
                label=model,
                color=spec["color"],
                source=source,
            ),
        ]

        plot_freq_density(
            tp_hist_data,
            ax=axd[tp_key],
            target_da=historical_cpm_pr,
            target_label="historic CPM",
            title=tp_key.title(),
            legend=tp_idx == 0,
        )

    ax = axd["Emulator"]

    linestyles = ["-", "--", ":"]

    emulator_hist_data = [
        dict(
            data=pred_pr.where(pred_pr["time_period"] == tp_key, drop=True),
            label=tp_key,
            color="black",
            linestyle=linestyles[tp_idx],
            source=source,
        )
        for tp_idx, tp_key in enumerate(TIME_PERIODS.keys())
    ]

    plot_freq_density(
        emulator_hist_data, ax=ax, title="Emulator", alpha=0.75, linewidth=1
    )
    ax.set_ylim(None, 1)


def change_over_time(pr_da, from_tp, to_tp, dim, stat_func):
    from_val = stat_func(
        pr_da.where(pr_da["time_period"] == from_tp, drop=True), dim=dim
    )

    to_val = stat_func(pr_da.where(pr_da["time_period"] == to_tp, drop=True), dim=dim)

    return 100 * (to_val - from_val) / from_val


def compute_changes(pred_pr_das, cpm_pr, seasons, stat_func):
    changes = {}

    for season in seasons:
        cpm_change = change_over_time(
            cpm_pr.where(cpm_pr["time.season"] == season, drop=True),
            "historic",
            "future",
            ["ensemble_member", "time"],
            stat_func,
        )
        changes[season] = {
            "CPM change": cpm_change,
            "ML changes": [],
            "ML labels": [],
        }
        for pred_pr_da in pred_pr_das:
            changes[season]["ML changes"].append(
                change_over_time(
                    pred_pr_da.where(pred_pr_da["time.season"] == season, drop=True),
                    "historic",
                    "future",
                    ["ensemble_member", "time", "sample_id"],
                    stat_func,
                ),
            )
            changes[season]["ML labels"].append(pred_pr_da["model"].data.item())

    return changes


def plot_changes(changes, seasons, fig, show_change=[]):
    human_names = {
        "DJF": "Winter",
        "MAM": "Spring",
        "JJA": "Summer",
        "SON": "Autumn",
    }

    grid_spec = [
        [f"{season} CPM"]
        + [f"{season} {model}" for model in show_change]
        + [f"{season} Error {model}" for model in changes[season]["ML labels"]]
        for season in seasons
    ]
    change_axd = fig.subplot_mosaic(
        grid_spec,
        subplot_kw=dict(projection=cp_model_rotated_pole),
    )

    for irow, season in enumerate(seasons):
        coloffset = 0
        ax = change_axd[f"{season} CPM"]
        plot_map(
            changes[season]["CPM change"],
            ax,
            title="",
            style="discreteBias",
            add_colorbar=False,
        )
        ax.annotate(
            f"{string.ascii_lowercase[irow*len(grid_spec[0])+coloffset]}.",
            xy=(-0.2, 0.9),
            xycoords=("axes fraction", "axes fraction"),
            weight="bold",
            ha="left",
            va="bottom",
            fontsize="small",
        )
        if irow == 0:
            ax.set_title("CPM", fontsize="x-small")
        ax.text(
            -0.1,
            0.5,
            human_names[season],
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize="medium",
            rotation=90,
        )
        coloffset += 1

        for ml_change, model in zip(
            changes[season]["ML changes"], changes[season]["ML labels"]
        ):
            if model not in show_change:
                continue
            ax = change_axd[f"{season} {model}"]
            pcm = plot_map(
                ml_change,
                ax=ax,
                title="",
                style="discreteBias",
                add_colorbar=False,
            )
            if irow == 0:
                ax.set_title(model, fontsize="x-small")
            ax.annotate(
                f"{string.ascii_lowercase[irow*len(grid_spec[0])+coloffset]}.",
                xy=(-0.22, 0.9),
                xycoords=("axes fraction", "axes fraction"),
                weight="bold",
                ha="left",
                va="bottom",
                fontsize="small",
            )
            coloffset += 1

        for idiff, (ml_change, model) in enumerate(
            zip(changes[season]["ML changes"], changes[season]["ML labels"])
        ):
            ax = change_axd[f"{season} Error {model}"]
            plot_map(
                ml_change - changes[season]["CPM change"],
                ax,
                title="",
                style="discreteBias",
                add_colorbar=False,
            )
            if irow == 0:
                ax.set_title("Difference", fontsize="x-small")
            ax.annotate(
                f"{string.ascii_lowercase[irow*len(grid_spec[0])+coloffset+idiff]}.",
                xy=(-0.22, 0.9),
                xycoords=("axes fraction", "axes fraction"),
                weight="bold",
                ha="left",
                va="bottom",
                fontsize="small",
            )

    cb = fig.colorbar(
        pcm, ax=change_axd.values(), location="right", shrink=0.8, extend="both"
    )
    cb.set_label("Relative mean change\n(percent)", fontsize="small")
    cb.ax.tick_params(axis="both", labelsize="small")


def _tp_seasonal_domain_means_by_year(pr_da, tp, dim):
    return (
        pr_da.where(pr_da["time_period"] == tp, drop=True)
        .groupby("tp_season_year")
        .mean(dim=dim)
    )


def bootstrap_seasonal_mean_pr_change_samples(cpm_pr, pred_pr, nsamples=1000):
    ds = xr.merge([cpm_pr, pred_pr])

    hist_means = _tp_seasonal_domain_means_by_year(
        ds, tp="historic", dim=["grid_latitude", "grid_longitude", "sample_id", "time"]
    ).stack(member=["ensemble_member", "tp_season_year"])

    fut_means = _tp_seasonal_domain_means_by_year(
        ds, tp="future", dim=["grid_latitude", "grid_longitude", "sample_id", "time"]
    ).stack(member=["ensemble_member", "tp_season_year"])

    hist_mean_samples = xs.resampling.resample_iterations_idx(
        hist_means, nsamples, "member", replace=True
    ).unstack("member")

    fut_mean_samples = xs.resampling.resample_iterations_idx(
        fut_means, nsamples, "member", replace=True
    ).unstack("member")

    return hist_mean_samples.mean(
        ["ensemble_member", "tp_season_year"]
    ), fut_mean_samples.mean(["ensemble_member", "tp_season_year"])

    hist_mean_pr_differences = hist_mean_samples["pred_pr"].mean(
        ["ensemble_member", "tp_season_year"]
    ) - hist_mean_samples["target_pr"].mean(["ensemble_member", "tp_season_year"])

    fut_mean_pr_differences = fut_mean_samples["pred_pr"].mean(
        ["ensemble_member", "tp_season_year"]
    ) - fut_mean_samples["target_pr"].mean(["ensemble_member", "tp_season_year"])

    return fut_mean_pr_differences - hist_mean_pr_differences

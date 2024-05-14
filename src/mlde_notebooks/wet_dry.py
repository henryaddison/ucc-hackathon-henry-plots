import matplotlib.pyplot as plt
import xarray as xr

from mlde_utils import cp_model_rotated_pole

from . import plot_map


def wet_day_prop(pr_da, threshold):
    example_dims = set(pr_da.dims) - set(["grid_latitude", "grid_longitude"])
    return (
        100 * (pr_da > threshold).sum(dim=example_dims) / pr_da.count(dim=example_dims)
    ).rename("% Wet day")


def wet_day_prop_error(sample_pr, cpm_pr, threshold):
    return (
        wet_day_prop(cpm_pr, threshold) - wet_day_prop(sample_pr, threshold)
    ).rename("% Wet day error")


def wet_day_prop_change(pr_da, threshold):
    from_pr = pr_da.where(pr_da["time_period"] == "historic", drop=True)
    to_pr = pr_da.where(pr_da["time_period"] == "future", drop=True)

    from_prop = wet_day_prop(from_pr, threshold=threshold).rename(
        "% wet day (historic)"
    )
    to_prop = wet_day_prop(to_pr, threshold=threshold).rename("% wet day (future)")

    change = (to_prop - from_prop).rename("change in % wet day")
    relative_change = (change / from_prop * 100).rename("relative change in % wet day")

    return xr.merge([from_prop, to_prop, change, relative_change])


def wet_prop_stats(
    sample_pr_das, cpm_pr_da, threshold, wet_prop_statistic=wet_day_prop
):
    # cpm_stats = wet_prop_statistic(cpm_pr_da, threshold=threshold).expand_dims(
    #     model=["CPM"]
    # )

    # stats = xr.concat(
    #     [
    #         wet_prop_statistic(
    #             sample_pr_da,
    #             threshold=threshold,
    #         )
    #         for sample_pr_da in sample_pr_das
    #     ]
    #     + [cpm_stats],
    #     dim="model",
    # ).expand_dims(dim={"season": ["Annual"]})

    all_stats = []

    for season in ("Annual", "DJF", "JJA"):
        if season == "Annual":
            season_mask = {}
        else:
            season_mask = {"time": cpm_pr_da["time"]["time.season"] == season}

        seasonal_cpm_stats = wet_prop_statistic(
            cpm_pr_da.sel(season_mask), threshold=threshold
        ).expand_dims(model=["CPM"])
        seasonal_stats = xr.concat(
            [
                wet_prop_statistic(
                    sample_pr_da.sel(season_mask),
                    threshold=threshold,
                )
                for sample_pr_da in sample_pr_das
            ]
            + [seasonal_cpm_stats],
            dim="model",
        ).expand_dims(dim={"season": [season]})

        all_stats.append(seasonal_stats)

    return xr.concat(all_stats, dim="season")


def plot_wet_dry_errors(wet_day_stats, style="raw"):
    fig = plt.figure(layout="constrained", figsize=(10, 10))

    seasons = list(wet_day_stats["season"].data)

    spec = []
    for season in seasons:
        spec.extend(
            [
                [f"{season} CPM"]
                + [
                    f"{season} {model}"
                    for model in wet_day_stats["model"].data
                    if model != "CPM"
                ],
                [f"."]
                + [
                    f"{season} CPM - {model}"
                    for model in wet_day_stats["model"].data
                    if model != "CPM"
                ],
            ]
        )

    axd = fig.subplot_mosaic(spec, subplot_kw={"projection": cp_model_rotated_pole})

    if style == "raw":
        plot_map_kwargs = {"vmin": 20, "vmax": 80, "style": None}
    elif style == "change":
        plot_map_kwargs = {"vmin": -20, "center": 0, "style": None, "cmap": "BrBG"}

    for season in seasons:
        ax = axd[f"{season} CPM"]
        ax.text(
            -0.1,
            0,
            season,
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize="large",
            fontweight="bold",
        )

        for label, model_wet_prop in wet_day_stats.sel(season=season).groupby(
            "model", squeeze=False
        ):
            ax = axd[f"{season} {label}"]
            plot_map(
                model_wet_prop.squeeze("model"),
                ax,
                title=f"{label}",
                add_colorbar=True,
                **plot_map_kwargs,
            )
            ax.set_title(label, fontsize="medium")

            if label != "CPM":
                label = f"CPM - {label}"
                ax = axd[f"{season} {label}"]
                plot_map(
                    (
                        wet_day_stats.sel(season=season, model="CPM")
                        - model_wet_prop.squeeze("model")
                    ).rename("Difference"),
                    ax,
                    title=f"{label}",
                    style=None,
                    add_colorbar=True,
                    vmin=-20,
                    center=0,
                    cmap="RdBu",
                )
                ax.set_title(label, fontsize="medium")

    return fig, axd

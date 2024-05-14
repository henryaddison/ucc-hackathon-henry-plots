import cftime
import math
import matplotlib

from mlde_utils import cp_model_rotated_pole

from . import (
    plot_map,
    precip_clevs,
    precip_cmap,
    precip_norm,
    sorted_em_time_by_mean_pr,
)


def em_timestamps(ds, seasons, percentiles, overrides={}):
    em_ts = {}
    for season in seasons:
        seasonal_ds = ds.sel(time=ds["time"]["time.season"] == season)

        mean_sorted_em_time = sorted_em_time_by_mean_pr(seasonal_ds)

        for sample_percentile in percentiles:
            em_ts[f"{season} {sample_percentile['label']}"] = mean_sorted_em_time[
                math.ceil(
                    len(mean_sorted_em_time) * (1 - sample_percentile["percentile"])
                )
            ]

    for key, value in overrides.items():
        em_ts[key] = (
            value[0],
            cftime._cftime.Datetime360Day.strptime(
                value[1], "%Y-%m-%d %H:%M:%S", calendar="360_day"
            ),
        )

    return em_ts


def show_samples(
    ds,
    em_ts,
    models,
    fig,
    sim_title,
):
    n_samples = 2

    inputs = ["vorticity850"]

    det_models = [
        mlabel for mlabel, mconfig in models.items() if mconfig["deterministic"]
    ]

    stoch_models = [
        mlabel for mlabel, mconfig in models.items() if not mconfig["deterministic"]
    ][:1]

    axes = fig.subplots(
        nrows=len(em_ts),
        ncols=1 + len(inputs) + len(det_models) + len(stoch_models) * n_samples,
        subplot_kw={"projection": cp_model_rotated_pole},
    )

    bilinear_present = any(map(lambda x: "Bilinear" in x, det_models))

    for tsi, (desc, ts) in enumerate(em_ts.items()):
        print(f"Precip requested for EM{ts[0]} on {ts[1]}")

        ts_ds = ds.sel(ensemble_member=ts[0]).sel(time=ts[1], method="nearest")
        print(
            f"Precip actually for EM{ts_ds['ensemble_member'].data.item()} on {ts_ds['time'].data.item()}"
        )

        ax = axes[tsi][0]
        plot_map(
            ts_ds.isel(model=0)["target_pr"],
            ax,
            cmap=precip_cmap,
            norm=precip_norm,
            add_colorbar=False,
        )
        # label column
        if tsi == 0:
            ax.set_title(sim_title, fontsize="small")

        # label row
        ax.text(
            -0.1,
            0.5,
            desc,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize="small",
            rotation=90,
        )

        for input_idx, var in enumerate(inputs):
            ax = axes[tsi][1 + bilinear_present + input_idx]
            plot_map(
                ts_ds[var],
                ax,
                style=None,
                cmap="Greys",
                add_colorbar=False,
            )
            # label column
            if tsi == 0:
                ax.set_title(f"Example coarse\ninput", fontsize="small")

        for mi, model in enumerate(stoch_models):
            for sample_idx in range(n_samples):
                ax = axes[tsi][
                    1 + bilinear_present + len(inputs) + mi * n_samples + sample_idx
                ]
                plot_map(
                    ts_ds.sel(model=model).isel(sample_id=sample_idx)["pred_pr"],
                    ax,
                    cmap=precip_cmap,
                    norm=precip_norm,
                    add_colorbar=False,
                )

                if tsi == 0:
                    ax.set_title(f"Sample {sample_idx+1}", fontsize="small")

                    if sample_idx == 0:
                        fig.text(
                            1,
                            1.25,
                            f"{model}",
                            ha="center",
                            fontsize="small",
                            transform=ax.transAxes,
                        )

        det_model_offset = 0
        for mi, model in enumerate(det_models):
            if "Bilinear" in model:
                icol = 1
                det_model_offset = -1
            else:
                icol = (
                    1
                    + bilinear_present
                    + len(inputs)
                    + len(stoch_models) * n_samples
                    + mi
                    + det_model_offset
                )

            ax = axes[tsi][icol]
            plot_map(
                ts_ds.sel(model=model).isel(sample_id=0)["pred_pr"],
                ax,
                cmap=precip_cmap,
                norm=precip_norm,
                add_colorbar=False,
            )
            if tsi == 0:
                ax.set_title(f"{model}", fontsize="small")

    ax = fig.add_axes([0.05, -0.05, 0.95, 0.05])
    cb = matplotlib.colorbar.Colorbar(
        ax, cmap=precip_cmap, norm=precip_norm, orientation="horizontal"
    )
    cb.ax.set_xticks(precip_clevs)
    cb.ax.set_xticklabels(precip_clevs, fontsize="small")
    cb.ax.tick_params(axis="both", which="major")
    cb.ax.set_xlabel("Precip (mm day-1)", fontsize="small")

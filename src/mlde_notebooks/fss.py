import matplotlib.pyplot as plt
import numpy as np
import pysteps
from typing import Iterable
import xarray as xr


def compute_fss(ds, threshold, fss_windows, grid_box_size=8.8):
    return xr.concat(
        [
            xr.apply_ufunc(
                pysteps.verification.spatialscores.fss,  # first the function
                ds["pred_pr"].mean(
                    dim="sample_id"
                ),  # now arguments in the order expected by 'fss'
                ds["target_pr"],
                threshold,
                fss_window,
                input_core_dims=[
                    ["grid_latitude", "grid_longitude"],
                    ["grid_latitude", "grid_longitude"],
                    [],
                    [],
                ],  # list with one entry per arg
                output_core_dims=[[]],
                # exclude_dims=set(("grid_latitude", "grid_longitude",)),  # dimensions allowed to change size. Must be set!
                vectorize=True,
            ).expand_dims(dict(fss_window=[fss_window * grid_box_size]))
            for fss_window in fss_windows
        ],
        dim="fss_window",
    ).mean(dim=["ensemble_member", "time"])


def plot_fss(
    fig: plt.Figure,
    ds: xr.Dataset,
    thresholds: list[float],
    grid_box_size: float = 8.8,
    fss_windows: Iterable[int] = range(4, 33, 4),
):
    axd = fig.subplot_mosaic(np.array([thresholds]).reshape(-1, 2))

    for i, threshold in enumerate(thresholds):
        ax = axd[threshold]
        fss_scores = compute_fss(
            ds, threshold, fss_windows, grid_box_size=grid_box_size
        )

        for group_label, group_da in fss_scores.groupby("model", squeeze=True):
            group_da.plot.line(
                x="fss_window",
                ax=ax,
                label=f"{group_label}",
                add_legend=False,
            )
        ax.set_title(f"Threshold: {threshold:.2f}mm/day", fontsize="small")
        ax.set_ylabel("FSS")
        ax.set_xlabel("FSS window (km)")
        if i == 0:
            ax.legend()

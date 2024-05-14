import math
from matplotlib import pyplot as plt
import numpy as np
import scipy

from mlde_notebooks import plot_map


def mean_bias(sample_pr, cpm_pr, normalize=False):
    sample_dims = set(["ensemble_member", "sample_id", "time"]) & set(sample_pr.dims)

    sample_summary = sample_pr.mean(dim=sample_dims)

    truth_dims = set(["ensemble_member", "sample_id", "time"]) & set(cpm_pr.dims)

    cpm_summary = cpm_pr.mean(dim=truth_dims)

    raw_bias = sample_summary - cpm_summary

    if normalize:
        return 100 * raw_bias / cpm_summary
    else:
        return raw_bias


def std_bias(sample_pr, cpm_pr, normalize=False):
    sample_dims = set(["ensemble_member", "sample_id", "time"]) & set(sample_pr.dims)
    sample_summary = sample_pr.std(dim=sample_dims)

    truth_dims = set(["ensemble_member", "sample_id", "time"]) & set(cpm_pr.dims)
    cpm_summary = cpm_pr.std(dim=truth_dims)

    raw_bias = sample_summary - cpm_summary

    if normalize:
        return 100 * raw_bias / cpm_summary
    else:
        return raw_bias


def rms_mean_bias(sample_pr, cpm_pr, normalize=False):
    return np.sqrt((mean_bias(sample_pr, cpm_pr, normalize=normalize) ** 2).mean())


def rms_std_bias(sample_pr, cpm_pr, normalize=False):
    return np.sqrt((std_bias(sample_pr, cpm_pr, normalize=normalize) ** 2).mean())


def normalized_mean_bias(sample_pr, cpm_pr):
    return mean_bias(sample_pr, cpm_pr, normalize=True)


def normalized_std_bias(sample_pr, cpm_pr):
    return std_bias(sample_pr, cpm_pr, normalize=True)


def plot_freq_density(
    hist_data,
    ax,
    target_da=None,
    target_label="CPM",
    title="",
    legend=True,
    linestyle="-",
    alpha=0.95,
    linewidth=2,
    **kwargs,
):

    hrange = (
        0,
        250,  # max(*[d["data"].max().values for d in hist_data], target_da.max().values),
    )

    if target_da is not None:
        print(f"Target max: {target_da.max().values}")
    for d in hist_data:
        print(f"{d['label']} max: {d['data'].max().values}")

    bins = np.histogram_bin_edges([], bins=50, range=hrange)

    if target_da is not None:
        min_density = 1 / np.product(target_da.shape)
        print(min_density)
        ymin = 10 ** (math.floor(math.log10(min_density))) / 2
        print(ymin)
        counts, bins = np.histogram(target_da, bins=bins, range=hrange, density=True)
        ax.stairs(
            counts,
            bins,
            fill=True,
            color="black",
            alpha=0.2,
            label=target_label,
        )
    else:
        ymin = None

    for pred in hist_data:
        counts, bins = np.histogram(pred["data"], bins=bins, range=hrange, density=True)
        ax.stairs(
            counts,
            bins,
            fill=False,
            color=pred["color"],
            alpha=pred.get("alpha", alpha),
            linestyle=pred.get("linestyle", linestyle),
            linewidth=linewidth,
            label=f"{pred['label']}",
            **kwargs,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Precip (mm/day)")
    ax.set_ylabel("Freq. density")
    ax.set_ylim(ymin, None)
    ax.tick_params(axis="both", which="major")
    if legend:
        ax.legend(fontsize="small")
    ax.set_title(title)


def compute_fractional_contribution(pr_da, bins, range):
    density, bins = np.histogram(pr_da, bins=bins, range=range, density=True)

    bin_mean, _, _ = scipy.stats.binned_statistic(
        pr_da.data.reshape(-1), pr_da.data.reshape(-1), bins=bins, range=range
    )
    # don't allow NaNs in bin_means - if no values for bin then frac contrib will be 0
    bin_mean[np.isnan(bin_mean)] = 0

    return density * bin_mean


def plot_fractional_contribution(
    hist_data,
    ax,
    target_da=None,
    target_label="CPM",
    title="",
    legend=True,
    linestyle="-",
    alpha=0.95,
    linewidth=2,
    **kwargs,
):

    hrange = (
        0,
        250,  # max(*[d["data"].max().values for d in hist_data], target_da.max().values),
    )

    bins = np.histogram_bin_edges([], bins=50, range=hrange)

    if target_da is not None:
        frac_contrib = compute_fractional_contribution(target_da, bins, hrange)

        ax.stairs(
            frac_contrib,
            bins,
            fill=True,
            color="black",
            alpha=0.2,
            label=target_label,
        )
    # else:
    #     ymin = None

    for pred in hist_data:
        frac_contrib = compute_fractional_contribution(pred["data"], bins, hrange)

        ax.stairs(
            frac_contrib,
            bins,
            fill=False,
            color=pred["color"],
            alpha=pred.get("alpha", alpha),
            linestyle=pred.get("linestyle", linestyle),
            linewidth=linewidth,
            label=f"{pred['label']}",
            **kwargs,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Precip (mm/day)")
    ax.set_ylabel("Fractional contrib.\n(mm/day)")
    # ax.set_ylim(ymin, None)
    ax.tick_params(axis="both", which="major")
    if legend:
        ax.legend(fontsize="small")
    ax.set_title(title)


def plot_mean_biases(mean_biases, axd, colorbar=False, plot_map_kwargs={}):
    meanb_axes = []
    for i, bias in enumerate(mean_biases):
        label = bias["label"]
        bias_ratio = bias["data"]
        ax = axd[f"meanb {label}"]
        meanb_axes.append(ax)
        pcm = plot_map(
            bias_ratio,
            ax,
            title=f"{label}",
            style="discreteBias",
            add_colorbar=False,
            **plot_map_kwargs,
        )
        ax.set_title(label, fontsize="medium")
        if i == 0:
            ax.text(
                -0.1,
                0.5,
                "Mean",
                transform=ax.transAxes,
                ha="right",
                va="center",
                # fontsize="small",
                rotation=90,
            )

    if colorbar:
        cb = plt.colorbar(
            pcm,
            ax=meanb_axes,
            location="bottom",
            shrink=0.8,
            extend="both",
            aspect=40,
        )
        cb.set_label("Relative bias (percent)")
    return meanb_axes


def plot_std_biases(std_biases, axd, colorbar=True, plot_map_kwargs={}):
    stddevb_axes = []
    # meanb_axes = []
    for i, bias in enumerate(std_biases):
        label = bias["label"]
        bias_ratio = bias["data"]
        ax = axd[f"stddevb {label}"]
        stddevb_axes.append(ax)
        # meanb_axes.append(axd[f"meanb {label}"])
        pcm = plot_map(
            bias_ratio - 1,
            ax,
            title=f"{label}",
            style="discreteBias",
            add_colorbar=False,
            **plot_map_kwargs,
        )
        ax.set_title(label, fontsize="medium")
        if i == 0:
            ax.text(
                -0.1,
                0.5,
                "Std. dev.",
                transform=ax.transAxes,
                ha="right",
                va="center",
                # fontsize="small",
                rotation=90,
            )
    if colorbar:
        cb = plt.colorbar(
            pcm,
            ax=stddevb_axes,
            location="bottom",
            shrink=0.8,
            extend="both",
            aspect=40,
        )
        cb.set_label("Relative bias (percent)")

    return stddevb_axes

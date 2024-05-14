import numpy as np
import scipy


def compute_rmss_rmse_bins(pred_pr, target_pr, nbins=100):
    """
    For an "ensemble" of predicted rainfall and the coresponding "truth" value,
    this computes bins for spread and error for a spread-error plot

    Sources:

    * https://journals.ametsoc.org/view/journals/hydr/15/4/jhm-d-14-0008_1.xml?tab_body=fulltext-display
    * https://journals.ametsoc.org/view/journals/aies/2/2/AIES-D-22-0061.1.xml
    * https://www.sciencedirect.com/science/article/pii/S0021999107000812
    """

    # Need to correct for the finite ensemble (or num samples runs) size of samples
    # Equation 7 from # Leutbecher, M., & Palmer, T. N. (2008). Ensemble forecasting. Journal of Computational Physics, 227(7), 3515-3539. doi:10.1016/j.jcp.2007.02.014
    ensemble_size = len(pred_pr["sample_id"])
    variance_correction_term = (ensemble_size + 1) / (ensemble_size - 1)

    ensemble_mean = pred_pr.mean(dim=["sample_id"])
    ensemble_variance = (
        variance_correction_term
        * np.power(pred_pr - ensemble_mean, 2).mean(dim="sample_id").values.flatten()
    )

    squared_error = np.power(ensemble_mean - target_pr, 2).values.flatten()

    bin_edges = np.concatenate(
        [[0.0], np.quantile(ensemble_variance, np.linspace(0, 1, nbins + 1))]
    )
    # remove bin edges too near each other
    bin_edges = np.delete(bin_edges, np.argwhere(np.ediff1d(bin_edges) <= 1e-6) + 1)

    spread_binned_mse, _, abinnumbers = scipy.stats.binned_statistic(
        ensemble_variance, squared_error, statistic="mean", bins=bin_edges
    )
    spread_binned_rmse = np.sqrt(spread_binned_mse)

    spread_binned_variance, _, bbinnumbers = scipy.stats.binned_statistic(
        ensemble_variance, ensemble_variance, statistic="mean", bins=bin_edges
    )
    spread_binned_rmss = np.sqrt(spread_binned_variance)

    assert (abinnumbers == bbinnumbers).all()

    return spread_binned_rmss, spread_binned_rmse


def plot_spread_error(ds, ax, line_props):
    rmss_rmse_max = 0

    for model, model_ds in ds.groupby("model"):
        pred_pr = model_ds["pred_pr"]
        target_pr = model_ds["target_pr"]

        binned_rmss, binned_rmse = compute_rmss_rmse_bins(pred_pr, target_pr)

        rmss_rmse_max = max(rmss_rmse_max, np.max(binned_rmss), np.max(binned_rmse))

        ax.scatter(
            binned_rmss, binned_rmse, label=f"{model}", color=line_props[model]["color"]
        )

    ax.plot(
        [0, rmss_rmse_max],
        [0, rmss_rmse_max],
        label="ideal",
        color="black",
        alpha=0.5,
        linestyle="--",
    )
    # ax.legend()
    ax.set_xlabel("RMSS (mm/day)", fontsize="small")
    ax.set_ylabel("RMSE (mm/day)", fontsize="small")
    ax.set_title("CPM Diffusion\nSpread-Error", fontsize="medium")
    ax.set_aspect(aspect=1)

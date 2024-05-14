import scipy
import xarray as xr


def significance_test(ds: xr.Dataset):
    """Perform significance test on the samples of the diffusion model.

    Args:
        ds (xr.Dataset): Dataset containing the samples of the diffusion model.
    """

    dec_adjusted_year = ds["time.year"] + (ds["time.month"] == 12)
    season = ds["time.season"]
    season_year = season.data + dec_adjusted_year.astype("str").data
    # season_year_grouped = ds.assign_coords(season_year=("time", season_year)).groupby("season_year")

    season_year_sample_mean = (
        ds["pred_pr"]
        .assign_coords(season_year=("time", season_year))
        .groupby("season_year")
        .mean(dim=["time", "ensemble_member", "sample_id"])
    )
    season_year_sim_mean = (
        ds["target_pr"]
        .assign_coords(season_year=("time", season_year))
        .groupby("season_year")
        .mean(dim=["time", "ensemble_member"])
    )

    return scipy.stats.ttest_rel(season_year_sample_mean, season_year_sim_mean, axis=0)

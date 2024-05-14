import numpy as np
import pysteps
import scipy
import xarray as xr


def psd(batch, npix):
    assert len(batch.shape) == 3, batch.shape
    # npix = batch.shape[1]
    fourier = np.fft.fftshift(np.fft.fftn(batch, axes=(1, 2)), axes=(1, 2))
    amps = np.abs(fourier) ** 2 / npix**2
    return amps


def _rapsd(precip_np, npix, pixel_size):
    fourier_amplitudes = psd(precip_np, npix)

    kfreq = np.fft.fftshift(np.fft.fftfreq(npix, d=pixel_size))
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    kbins = (
        np.arange(0.5, npix // 2 + 1, 1.0) / npix / pixel_size
    )  # bands of k-norms for radially averaging over
    # kvals = 0.5 * (kbins[1:] + kbins[:-1]) # mid-point of each bin

    # radially average the means for each example
    # take mean of amplitudes of each example once grouped into tori
    # kbins defined equal-width (though not equal area) tori of Fourier plane with radius between start and end of each bin point
    # knrm is size of k at each point in the plane (basically Euclidean distance in Fourier plane from centre point of the 64x64 array) so can determine which torus each member of fourier amplitudes belongs
    Abins, _, _ = scipy.stats.binned_statistic(
        knrm.flatten(),
        fourier_amplitudes.reshape(-1, npix * npix),
        statistic="mean",
        bins=kbins,
    )

    return Abins


def rapsd(pr_da, pixel_size=8.8):
    npix = max(pr_da[0].shape)

    kbins = (
        np.arange(0.5, npix // 2 + 1, 1.0) / npix / pixel_size
    )  # bands of k-norms for radially averaging over
    kvals = 0.5 * (kbins[1:] + kbins[:-1])  # mid-point of each bin

    rapsd_da = xr.apply_ufunc(
        _rapsd,  # first the function
        pr_da,  # now arguments in the order expected by function
        npix,
        pixel_size,
        input_core_dims=[
            ["grid_latitude", "grid_longitude"],
            [],
            [],
        ],  # list with one entry per arg
        output_core_dims=[
            ["freq"],
        ],
        exclude_dims=set(
            (
                "grid_latitude",
                "grid_longitude",
            )
        ),  # dimensions allowed to change size. Must be set!
        # vectorize=True,
    )
    rapsd_da["freq"] = kvals
    return rapsd_da


def _urapsd(example, pixel_size):
    return pysteps.utils.spectral.rapsd(
        example, d=pixel_size, return_freq=False, fft_method=np.fft
    )


def pysteps_rapsd(pr_da, pixel_size):
    npix = max(pr_da[0].shape)
    freqs = np.fft.fftfreq(npix, d=pixel_size)
    if npix % 2 == 1:
        r_range = np.arange(0, int(npix / 2) + 1)
    else:
        r_range = np.arange(0, int(npix / 2))
    freqs = freqs[r_range]
    rapsd_da = xr.apply_ufunc(
        _urapsd,  # first the function
        pr_da,  # now arguments in the order expected by function
        pixel_size,
        input_core_dims=[
            ["grid_latitude", "grid_longitude"],
            [],
        ],  # list with one entry per arg
        output_core_dims=[
            ["freq"],
        ],
        exclude_dims=set(
            (
                "grid_latitude",
                "grid_longitude",
            )
        ),  # dimensions allowed to change size. Must be set!
        vectorize=True,
    )
    rapsd_da["freq"] = freqs
    return rapsd_da


def plot_psd(target_rapsd, pred_rapsds, ax, legend_kwargs={}):
    ax.loglog(
        target_rapsd["freq"].data,
        target_rapsd.data,
        label="CPM",
        color="black",
        linewidth=3,
        linestyle=":",
        alpha=0.5,
    )

    for pred_psd in pred_rapsds:
        da = pred_psd["data"]
        ax.loglog(
            da["freq"].data, da.data, label=pred_psd["label"], color=pred_psd["color"]
        )

    ax.set_xlabel("Spatial Frequency ($km^{-1}$)")
    ax.set_ylabel("PSD")
    ax.legend(ncols=2, **legend_kwargs)
    ax.tick_params(axis="both", which="minor")
    # ax.set_xlim(1e-3, 1e-1)

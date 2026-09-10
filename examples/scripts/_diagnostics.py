"""Shared presentation code for the scripts and notebooks; models live in lyaforecast."""
import argparse
import configparser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lyaforecast.analytic_p1d_PD2013 import P1D_z_kms_PD2013
from lyaforecast.fisher import Fisher
from lyaforecast.forecast_new import NewForecast
from lyaforecast.utils import get_file


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "desi2/lya_qso_2x2pt.ini"


def load_forecast(config_path, output_dir):
    """Use the supplied scientific configuration, overriding only its output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(get_file(config_path))
    config["output"]["filename"] = str(output_dir.resolve() / "forecast")
    effective = output_dir / "effective.ini"
    with effective.open("w") as stream:
        config.write(stream)
    return NewForecast(effective)


def style():
    """Use one plotting style for maintained diagnostics."""
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .2,
                         "figure.figsize": (8, 5), "figure.constrained_layout.use": True})


def plot_p1d(forecast):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    k = np.logspace(-4, -1, 200)
    for z in (2., 3., 4.):
        axes[0].loglog(k, P1D_z_kms_PD2013(z, k), label=f"z={z:g}")
    axes[0].set(xlabel="k [s/km]", ylabel="P1D [km/s]", title="Analytic P1D")
    kh = np.logspace(-2, 0, 200)
    for label, resolution, pixel in (("Unsmoothed", None, None),
                                     ("Gaussian resolution", 1., None),
                                     ("Top-hat pixel", None, 1.)):
        power = forecast.power_spectrum.compute_p1d_hmpc(3., kh, resolution, pixel)
        axes[1].loglog(kh, power, label=label)
    axes[1].set(xlabel="k [h/Mpc]", ylabel="P1D [Mpc/h]", title="Resolution and pixel smoothing")
    for ax in axes:
        ax.legend()
    return {"p1d": fig}


def plot_cosmo(forecast):
    cosmo = forecast.cosmo
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    k = forecast.power_spectrum.k
    axes[0].loglog(k, cosmo.get_pk_lin(k))
    axes[0].set(xlabel="k [h/Mpc]", ylabel="P linear [(Mpc/h)^3]",
                title=f"CAMB at z={cosmo.z_ref:g}")
    z = forecast.survey.z_bin_centres
    axes[1].plot(z, cosmo.growth_factor_ratios, "o-", label="sigma8(z) / sigma8(z_ref)")
    axes[1].plot(z, cosmo.growth_rate_zbins, "s-", label="Growth rate f(z)")
    axes[1].set(xlabel="Redshift", ylabel="Growth", title="Configured redshift bins")
    axes[1].legend()
    return {"cosmology": fig}


def plot_p3d(forecast):
    spectrum = forecast.power_spectrum
    k = spectrum.k
    z = forecast.survey.z_bin_centres[0]
    corr = next(name for name, (a, b) in forecast.correlations.items() if a is b)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mu in (0., 1.):
        power = spectrum.compute_p3d_hmpc(z, k, mu, corr)
        axes[0].loglog(k, power, label=f"mu={mu:g}")
    # Reuse the exact peak extraction used by Fisher; no pasted reference arrays.
    fisher = Fisher(spectrum, forecast.cosmo, np.ones_like(k))
    peak = fisher._get_p_pk(power[None, :])[0]
    for label, values in (("Full", power), ("Smooth", power-peak), ("Peak", peak)):
        axes[1].plot(k, k*values, label=label)
    axes[0].set(xlabel="k [h/Mpc]", ylabel="P3D [(Mpc/h)^3]", title=f"{corr}, z={z:g}")
    axes[1].set(xlabel="k [h/Mpc]", ylabel="k P3D [(Mpc/h)^2]", title="Fisher BAO decomposition")
    for ax in axes:
        ax.legend()
    return {"p3d": fig}


def plot_fisher(forecast, data=None):
    if data is None:
        data = forecast.new_run_forecast()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax in zip(("sigma_ap", "sigma_at"), axes):
        for corr in forecast.correlations_to_compute:
            ax.plot(data["redshifts"], data[corr][key], label=corr)
        ax.plot(data["redshifts"], data[key], "k-o", label="Combined")
        ax.set(xlabel="Redshift", ylabel=key, yscale="log")
        ax.legend(fontsize=8)

    # Construct diagnostic power/covariance arrays using current model methods.
    # These private covariance objects are used only in examples, not a new API.
    z = forecast.survey.z_bin_centres[0]
    mu = forecast.power_spectrum.mu[-1]
    k = forecast.power_spectrum.k
    measured = {}
    for corr, covariance in forecast._covariance.items():
        covariance(*forecast.survey.z_bin_edges[:, 0])
        covariance.compute_eff_density_and_noise(corr)
        measured[corr] = covariance.compute_total_power(k, mu, corr)
    corr = forecast.correlations_to_compute[0]
    covariance = forecast._covariance[corr]
    signal = forecast.power_spectrum.compute_p3d_hmpc_smooth(
        z, k, mu, covariance.pix_width_kms, covariance.pix_res_kms, corr)
    fisher = Fisher(forecast.power_spectrum, forecast.cosmo, covariance.num_modes)
    matrix, _ = fisher.gaussian_covariance_array_func(measured, [corr])
    diagnostic, ax = plt.subplots()
    ax.plot(k, signal/np.sqrt(matrix[0, 0]))
    ax.set(xlabel="k [h/Mpc]", ylabel="Signal / standard deviation",
           title=f"{corr}, z={z:g}, mu={mu:g}")
    return {"bao": fig, "power_signal_to_noise": diagnostic}


def plot_noise(forecast):
    corr = next(name for name, (a, b) in forecast.correlations.items()
                if a is b and a.type == "continuous")
    spec = forecast.spectrograph[corr]
    tracer = forecast.correlations[corr][0]
    wavelengths = np.linspace(max(3600., spec._lambda_obs_m[0]),
                              min(6000., spec._lambda_obs_m[-1]), 150)
    fig, ax = plt.subplots()
    for z in (2.5, 3.):
        for mag in (22., 22.5):
            noise = [np.asarray(spec.get_pixel_rms_noise(
                mag, z, wave, tracer.pix_ang, tracer.num_exp)).item() for wave in wavelengths]
            ax.plot(wavelengths, noise, label=f"z={z:g}, r={mag:g}")
    ax.set(xlabel="Observed wavelength [Angstrom]", ylabel="Pixel noise RMS", ylim=(0, 3))
    ax.legend()
    return {"noise": fig}


def plot_density(forecast):
    tracers = [t for t in forecast.tracers.values() if t.type == "discrete"]
    fig, ax = plt.subplots()
    for tracer in tracers:
        mags = np.linspace(tracer._mmin, tracer._mmax, 150)
        for z in (2.5, 3.):
            density = tracer.get_dn_dzdm(z, mags)
            ax.semilogy(mags, density, label=f"{tracer.name}, z={z:g}")
    ax.set(xlabel="r magnitude", ylabel="dn / dz / dm [deg^-2]", ylim=(1e-3, None))
    ax.legend()
    return {"tracer_density": fig}


def plot_bias(forecast):
    bias = forecast.power_spectrum.bias
    z = np.linspace(2., 4., 80)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in dict.fromkeys(t.simple_name for t in forecast.tracers.values()):
        axes[0].plot(z, [bias._get_density_bias(zi, name) for zi in z], label=name)
        axes[1].plot(z, [bias._get_beta_rsd(zi, name) for zi in z], label=name)
    old_z, old_beta, old_bias = np.loadtxt(get_file("z_beta_bias.txt"), unpack=True)
    axes[0].plot(old_z, -old_bias, "k:", label="Historical Lyα (McDonald 2003)")
    axes[1].plot(old_z, old_beta, "k:", label="Historical Lyα (McDonald 2003)")
    for ax, label in zip(axes, ("Density bias", "RSD beta")):
        ax.set(xlabel="Redshift", ylabel=label)
        ax.legend(fontsize=8)
    return {"bias": fig}


PLOTS = {"p1d": plot_p1d, "p3d": plot_p3d, "cosmo": plot_cosmo,
         "fisher": plot_fisher, "noise": plot_noise, "density": plot_density, "bias": plot_bias}


def main(kind):
    parser = argparse.ArgumentParser(description=f"Plot {kind} using current lyaforecast models.")
    parser.add_argument("-i", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs") / kind)
    args = parser.parse_args()
    style()
    forecast = load_forecast(args.config, args.output_dir)
    for name, figure in PLOTS[kind](forecast).items():
        figure.savefig(args.output_dir / f"{name}.png")
        plt.close(figure)

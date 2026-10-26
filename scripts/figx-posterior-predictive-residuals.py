"""Plot native-resolution posterior-predictive residual maps for RACS."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from catsim import Racs, batch_simulate, prng_key
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from numpy.typing import NDArray

from ciri.artifact_io import (
    load_final_posterior,
    load_model_config,
    load_reference_observation,
)
from ciri.plotting import smooth_map
from ciri.palette import RACS_SURVEY_COLOURS
from ciri.plot_style import configure_plot_style, configure_sky_graticule


POSTERIOR_PARAMETERS = (
    "log10_n_initial_samples",
    "observer_speed",
    "dipole_longitude",
    "dipole_latitude",
    "temp_beta",
    "lambda_clus",
)
NATIVE_NSIDE = 64
DEFAULT_N_SAMPLES = 500
DEFAULT_SEED = 42
SMOOTH_SCALE_STERADIANS = 1
Backend = Literal["numpy", "jax-cuda"]


configure_plot_style()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low3-artifact", type=Path, required=True)
    parser.add_argument("--mid1-artifact", type=Path, required=True)
    parser.add_argument("--low3-catalogue-path", type=Path, required=True)
    parser.add_argument("--mid1-catalogue-path", type=Path, required=True)
    parser.add_argument("--paf-dir-path", type=Path, required=True)
    parser.add_argument("--noisemap-path", type=Path, required=True)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of posterior draws (default: %(default)s).",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "jax-cuda"),
        default="numpy",
        help="Simulation backend (default: %(default)s).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="NumPy worker processes; ignored by jax-cuda (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="JAX simulations per device batch; ignored by numpy (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Posterior and simulation seed (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figx-posterior-predictive-residuals.pdf"),
    )
    args = parser.parse_args()
    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")
    if args.n_workers <= 0:
        parser.error("--n-workers must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def draw_posterior_parameters(
    artifact_path: Path,
    n_samples: int,
    seed: int,
) -> dict[str, NDArray[np.float64]]:
    """Draw an equal-weight Monte Carlo sample from a weighted posterior."""
    posterior = load_final_posterior(artifact_path)
    weights = np.array(posterior.get_weights(), dtype=np.float64, copy=True)
    if weights.shape != (len(posterior),):
        raise ValueError(f"Posterior weights have unexpected shape {weights.shape}")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Posterior weights must be finite and non-negative")
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("Posterior weights sum to zero")

    parameter_matrix = np.column_stack(
        [np.asarray(posterior[name], dtype=np.float64) for name in POSTERIOR_PARAMETERS]
    )
    invalid_rows = ~np.all(np.isfinite(parameter_matrix), axis=1)
    if np.any(invalid_rows & (weights > 0.0)):
        raise ValueError("Posterior contains non-finite parameters with positive weight")
    weights[invalid_rows] = 0.0
    weights /= weights.sum()

    # Replacement is intentional: this converts the weighted nested-sampling
    # posterior into an ordinary equal-weight posterior-predictive draw.
    indices = np.random.default_rng(seed).choice(
        len(posterior),
        size=n_samples,
        replace=True,
        p=weights,
    )
    return {
        name: parameter_matrix[indices, column]
        for column, name in enumerate(POSTERIOR_PARAMETERS)
    }


def prepare_model_config(
    artifact_path: Path,
    catalogue_path: Path,
    paf_dir_path: Path,
    noisemap_path: Path,
    expected_product: str,
) -> tuple[object, NDArray[np.float32], NDArray[np.bool_]]:
    config = load_model_config(artifact_path)
    if config.product.key != expected_product:
        raise ValueError(
            f"Expected a {expected_product} artifact; got {config.product.key}"
        )
    config.catalogue_path = str(catalogue_path.expanduser())
    config.paf_temperature_data_dir = str(paf_dir_path.expanduser())
    # The inference configuration is downscaled. Posterior predictive maps and
    # the archived native reference must instead remain at NSIDE 64.
    config.downscale_nside = None
    config.store_final_samples = False
    config.noisemap_data_dir = noisemap_path
    reference, native_mask = load_reference_observation(
        artifact_path,
        load_native=True,
    )
    expected_shape = (hp.nside2npix(NATIVE_NSIDE),)
    if reference.shape != expected_shape or native_mask.shape != expected_shape:
        raise ValueError(
            "Native reference and mask must both have shape "
            f"{expected_shape}; got {reference.shape} and {native_mask.shape}"
        )
    if np.any(~np.isfinite(reference[native_mask])):
        raise ValueError("Native reference contains non-finite unmasked pixels")
    config.mask_map = native_mask
    return config, reference, native_mask


def simulate_numpy(
    config,
    theta: dict[str, NDArray[np.float64]],
    n_workers: int,
    seed: int,
) -> tuple[NDArray, NDArray]:
    model = Racs(config)
    model.initialise_data()
    if len(theta[POSTERIOR_PARAMETERS[0]]) == 1:
        parameters = {name: values[0] for name, values in theta.items()}
        simulation, mask = model.generate_dipole(
            **parameters,
            rng_key=prng_key(seed),
        )
        return simulation[None, :], mask[None, :]
    return batch_simulate(
        theta,
        model.generate_dipole,
        n_workers=n_workers,
        rng_key=prng_key(seed),
    )


def simulate_jax_cuda(
    config,
    theta: dict[str, NDArray[np.float64]],
    batch_size: int,
    seed: int,
) -> tuple[NDArray, NDArray]:
    try:
        import jax
        from catsim import RacsJax
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "jax-cuda was requested, but CatSim's JAX dependencies are not "
            "installed. Install CatSim's optional JAX extra first."
        ) from exc

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError as exc:
        raise RuntimeError(
            "jax-cuda was requested, but JAX could not initialise a GPU backend"
        ) from exc
    if not gpu_devices:
        raise RuntimeError("jax-cuda was requested, but JAX found no GPU devices")

    with jax.default_device(gpu_devices[0]):
        model = RacsJax(config)
        model.initialise_data()
        return model.batch_generate_dipole(
            theta,
            jax.random.PRNGKey(seed),
            batch_size=batch_size,
            show_progress=True,
        )


def posterior_predictive_residual(
    artifact_path: Path,
    catalogue_path: Path,
    paf_dir_path: Path,
    noisemap_path: Path,
    expected_product: str,
    n_samples: int,
    backend: Backend,
    n_workers: int,
    batch_size: int,
    posterior_seed: int,
    simulation_seed: int,
) -> NDArray[np.floating]:
    config, reference, native_mask = prepare_model_config(
        artifact_path,
        catalogue_path,
        paf_dir_path,
        noisemap_path,
        expected_product,
    )
    theta = draw_posterior_parameters(artifact_path, n_samples, posterior_seed)
    if backend == "jax-cuda":
        simulations, masks = simulate_jax_cuda(
            config, theta, batch_size, simulation_seed
        )
    else:
        simulations, masks = simulate_numpy(
            config, theta, n_workers, simulation_seed
        )

    expected_shape = (n_samples, hp.nside2npix(NATIVE_NSIDE))
    simulations = np.asarray(simulations)
    masks = np.asarray(masks, dtype=np.bool_)
    if simulations.shape != expected_shape or masks.shape != expected_shape:
        raise ValueError(
            f"Simulations and masks must have shape {expected_shape}; got "
            f"{simulations.shape} and {masks.shape}"
        )
    if not np.all(masks == native_mask[None, :]):
        raise ValueError("Simulation masks do not match the native reference mask")
    if np.any(~np.isfinite(simulations[:, native_mask])):
        raise ValueError("Simulations contain non-finite unmasked pixels")

    simulations_smooth = smooth_map(
        simulations,
        angle_scale=SMOOTH_SCALE_STERADIANS,
        only_return_data=True
    )
    reference_smooth = smooth_map(
        reference,
        angle_scale=SMOOTH_SCALE_STERADIANS,
        only_return_data=True
    )
    mean_simulation = np.full(reference.shape, np.nan, dtype=np.float64)
    std_simulation = np.full(reference.shape, np.nan, dtype=np.float64)
    mean_simulation[native_mask] = simulations_smooth[:, native_mask].mean(axis=0)
    std_simulation[native_mask] = simulations_smooth[:, native_mask].std(axis=0)
    residual = np.full(reference.shape, np.nan, dtype=np.float64)
    valid_std = native_mask & np.isfinite(std_simulation) & (std_simulation > 0.0)
    np.divide(
        reference_smooth - mean_simulation,
        std_simulation,
        out=residual,
        where=valid_std,
    )

    return residual


def add_map_panel(
    sample_map: NDArray[np.floating], sub: int, cmap, norm, latitude_label_side: str,
):
    hp.projview(
        sample_map,
        cmap=cmap,
        norm=norm,
        nest=True,
        cbar=False,
        sub=sub,
        dpi=300,
    )
    axes = plt.gca()
    configure_sky_graticule(axes, latitude_label_side, colour='dark')
    return axes


def add_map_colorbar(figure, cmap, norm, limit: float) -> None:
    colorbar_axes = figure.add_axes([0.30, 0.06, 0.40, 0.035])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_axes,
        orientation="horizontal",
        ticks=[-limit, 0.0, limit],
        format="%.3g",
        label=r"$z$-scored residuals (data $-$ simulations)",
    )
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.set_xlim(-limit, limit)


def main() -> None:
    args = parse_args()
    child_seeds = np.random.SeedSequence(args.seed).spawn(4)
    seeds = [int(seed.generate_state(1, dtype=np.uint32)[0]) for seed in child_seeds]
    low3_map = posterior_predictive_residual(
        args.low3_artifact,
        args.low3_catalogue_path,
        args.paf_dir_path,
        args.noisemap_path,
        "low3",
        args.n_samples,
        args.backend,
        args.n_workers,
        args.batch_size,
        seeds[0],
        seeds[1],
    )
    mid1_map = posterior_predictive_residual(
        args.mid1_artifact,
        args.mid1_catalogue_path,
        args.paf_dir_path,
        args.noisemap_path,
        "mid1",
        args.n_samples,
        args.backend,
        args.n_workers,
        args.batch_size,
        seeds[2],
        seeds[3],
    )

    cmap = plt.get_cmap("RdBu_r")
    limit = max(
        float(np.nanmax(np.abs(low3_map))),
        float(np.nanmax(np.abs(mid1_map))),
    )
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("Residual maps must contain finite, non-zero values")
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    left_axes = add_map_panel(low3_map, 121, cmap, norm, 'right')
    right_axes = add_map_panel(mid1_map, 122, cmap, norm, 'left')
    left_axes.text(
        0.02, -0.03, "RACS-low3", transform=left_axes.transAxes,
        ha="left", va="bottom", color=RACS_SURVEY_COLOURS["low3"],
    )
    right_axes.text(
        0.98, -0.03, "RACS-mid1", transform=right_axes.transAxes,
        ha="right", va="bottom", color=RACS_SURVEY_COLOURS["mid1"],
    )

    figure = plt.gcf()
    add_map_colorbar(figure, cmap, norm, limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

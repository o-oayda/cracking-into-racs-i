import healpy as hp
from numpy.typing import NDArray
import numpy as np
from functools import lru_cache
import matplotlib
import scipy as sp


def smooth_map(
        healpy_map: NDArray,
        weights: NDArray | None = None,
        angle_scale: float = 1.,
        only_return_data: bool = False,
        fig: matplotlib.figure.Figure | None = None,
        **kwargs
    ) -> NDArray | None:
    smoothed_map_to_plot = average_smooth_map(
        healpy_map,
        weights=weights,
        angle_scale=angle_scale
    )

    if only_return_data:
        return smoothed_map_to_plot

    if smoothed_map_to_plot.ndim != 1:
        raise ValueError("Batched maps can only be used with only_return_data=True.")

    hp.projview(
        smoothed_map_to_plot,
        nest=True,
        fig=fig.number if fig is not None else None,
        **kwargs
    )
    return None


def average_smooth_map(
        healpy_map: NDArray[np.floating],
        weights: NDArray[np.floating] | None = None, 
        angle_scale: float = 1.
    ) -> NDArray:
    '''
    Smooth one or more HEALPix maps using a moving average.

    A one-dimensional input is treated as a single map. For a two-dimensional
    input, axis 0 is the batch axis and smoothing is applied along the pixel
    axis 1.
    '''
    healpy_map = np.asarray(healpy_map)
    if healpy_map.ndim not in (1, 2):
        raise ValueError("healpy_map must have shape (npix,) or (batch, npix).")

    if healpy_map.ndim == 2:
        n_batch, n_pixels = healpy_map.shape
        nside = hp.npix2nside(n_pixels)
        smoothed_map = np.full(
            healpy_map.shape,
            np.nan,
            dtype=np.result_type(healpy_map.dtype, np.float32),
        )

        if weights is None:
            batch_weights = np.ones_like(healpy_map)
        else:
            weights = np.asarray(weights)
            if weights.shape == (n_pixels,):
                batch_weights = np.broadcast_to(weights, healpy_map.shape)
            elif weights.shape == healpy_map.shape:
                batch_weights = weights
            else:
                raise ValueError(
                    "weights must have shape (npix,) or match the batched map shape."
                )

        smoothing_radius = omega_to_theta(angle_scale)
        neighbourhoods = _smoothing_neighbourhood_matrix(
            nside,
            float(smoothing_radius),
        )
        weighted_maps = healpy_map * batch_weights
        finite_values = np.isfinite(weighted_maps)
        sums = np.asarray(
            neighbourhoods @ np.where(finite_values, weighted_maps, 0.0).T
        ).T

        if np.all(finite_values == finite_values[:1]):
            shared_counts = np.asarray(
                neighbourhoods @ finite_values[0].astype(np.float32)
            ).reshape(-1)
            np.divide(
                sums,
                shared_counts[None, :],
                out=smoothed_map,
                where=shared_counts[None, :] > 0,
            )
        else:
            counts = np.asarray(
                neighbourhoods @ finite_values.astype(np.float32).T
            ).T
            np.divide(
                sums,
                counts,
                out=smoothed_map,
                where=counts > 0,
            )

        smoothed_map[~np.isfinite(healpy_map)] = np.nan
        assert smoothed_map.shape == (n_batch, n_pixels)
        return smoothed_map

    included_pixels = np.where(~np.isnan(healpy_map))[0]
    smoothed_map = np.nan * np.empty_like(healpy_map)
    nside = hp.get_nside(healpy_map)
    
    if weights is None:
        weights = np.ones_like(healpy_map)

    smoothing_radius = omega_to_theta(angle_scale)
    for p_index in included_pixels:
        vec = hp.pix2vec(nside, p_index, nest=True)
        disc = hp.query_disc(nside, vec, smoothing_radius, nest=True)
        smoothed_map[p_index] = np.nanmean(healpy_map[disc] * weights[disc])

    return smoothed_map


def omega_to_theta(omega: float) -> np.float64:
    '''
    Convert solid angle in steradins to theta in radians for
    a cone section of a sphere.

    :param omega: solid angle in steradians.
    '''
    return np.arccos( 1 - omega / (2 * np.pi) )


@lru_cache(maxsize=4)
def _smoothing_neighbourhood_matrix(
        nside: int,
        smoothing_radius: float,
    ) -> sp.sparse.csr_matrix:
    """Build and cache the top-hat neighbourhood operator for one geometry."""
    n_pixels = hp.nside2npix(nside)
    indptr = np.empty(n_pixels + 1, dtype=np.int64)
    indptr[0] = 0
    neighbourhoods: list[NDArray[np.int64]] = []
    for p_index in range(n_pixels):
        vec = hp.pix2vec(nside, p_index, nest=True)
        disc = np.asarray(
            hp.query_disc(nside, vec, smoothing_radius, nest=True),
            dtype=np.int64,
        )
        neighbourhoods.append(disc)
        indptr[p_index + 1] = indptr[p_index] + disc.size

    indices = np.concatenate(neighbourhoods)
    data = np.ones(indices.size, dtype=np.float32)
    return sp.sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n_pixels, n_pixels),
    )

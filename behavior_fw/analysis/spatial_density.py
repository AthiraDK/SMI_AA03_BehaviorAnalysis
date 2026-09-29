# behavior_fw/analysis/spatial_density.py

import numpy as np


def convert_to_polar(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Converts Cartesian coordinates (x, y) to polar coordinates (r, theta).

    Assumes origin (0,0) is centered. Returns r in arena units and theta in
    radians [-pi, pi].
    """
    r = np.hypot(x, y)  # Faster and numerically stable sqrt(x^2 + y^2)
    theta = np.arctan2(y, x)  # Standard NumPy arctan2 (y first, then x)
    return r, theta


def compute_2d_cartesian_histogram(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 91,
    arena_r: float = 45.0,
    density: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes a 2D spatial histogram in Cartesian coordinates.

    Returns transposed bin count matrix (H.T) formatted directly for
    matplotlib pcolormesh/imshow, along with x and y bin edges.
    """
    counts, edges_x, edges_y = np.histogram2d(
        x,
        y,
        bins=n_bins,
        range=[[-arena_r, arena_r], [-arena_r, arena_r]],
        density=density,
    )
    # Transpose so H[i, j] maps to x[i] and y[j] for standard meshgrid plotting
    return counts.T, edges_x, edges_y


def compute_2d_polar_histogram(
    r: np.ndarray,
    theta: np.ndarray,
    n_bins_r: int = 45,
    n_bins_theta: int = 180,
    arena_r: float = 45.0,
    density: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes an equal-area 2D spatial histogram in polar coordinates using r^2 binning.

    Binning r^2 directly guarantees that all radial concentric rings have equal surface area.
    """
    # r**2 binning ensures equal area bins
    counts, edges_r_sq, edges_theta = np.histogram2d(
        r**2,
        theta,
        bins=[n_bins_r, n_bins_theta],
        range=[[0, arena_r**2], [-np.pi, np.pi]],
        density=density,
    )

    edges_r = np.sqrt(edges_r_sq)

    # Transpose matrix for polar pcolormesh alignment (R along radius, Theta along angle)
    return counts.T, edges_r, edges_theta
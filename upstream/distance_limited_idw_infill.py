"""
distance_limited_idw_infill.py

Distance-limited infilling for 2D gridded data using local inverse-distance
weighting (IDW).

Overview
--------
This module provides a function for filling missing values in a 2D gridded
array, but only in locations that are close to existing observed data.
The intended use case is data with sparse, filamentary, or track-like spatial
coverage, such as aircraft trajectories gridded onto a latitude-longitude or
x-y raster.

The main design goal is to avoid "filling the world." In particular, this
routine is meant to bridge local gaps between nearby observed features while
preventing interpolation across very large empty regions.

Core idea
---------
The algorithm has two parts:

1. Eligibility test:
   A missing grid cell is only eligible to be filled if it is within
   `n_cells_max` grid-cell units of at least one valid observed cell.

2. Local interpolation:
   Each eligible missing cell is assigned a value by inverse-distance-weighted
   interpolation from its `k` nearest valid observed cells.

This is intentionally a *local* interpolator with a hard spatial cutoff.
That makes it much safer for sparse trajectory-style data than global splines,
biharmonic inpainting, unrestricted radial basis functions, or other methods
that can extrapolate into large unobserved domains.

Assumptions about the data
--------------------------
- The input is a 2D rectangular NumPy array.
- Missing values are represented by `np.nan` or other non-finite values.
- Grid spacing is assumed to be uniform in row/column coordinates, because
  distances are computed in units of grid cells.
- The algorithm uses Euclidean distance in array-index space, not great-circle
  distance or projected physical distance.
- The algorithm does not try to detect fronts, track orientation, coastlines,
  or barriers. It simply uses local distance geometry in the raster.

When this is appropriate
------------------------
This method is useful when:
- data coverage consists of quasi-linear or patchy sampled features,
- you want to bridge small gaps between nearby observed cells,
- you want a simple, robust method with easily interpreted controls.

When this may be less appropriate
---------------------------------
This method may be less appropriate when:
- your grid spacing is strongly anisotropic and row/column distance is not
  meaningful,
- the field contains sharp discontinuities that should not be smoothed,
- you need structure-aware infilling that respects feature orientation,
- you need physically constrained interpolation rather than a geometric fill.

Dependencies
------------
- numpy
- scipy.ndimage.distance_transform_edt
- scipy.spatial.cKDTree
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree


def infill_near_data_idw(
    gridded_data: np.ndarray,
    n_cells_max: float,
    k: int = 8,
    power: float = 2.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill missing cells in a 2D grid using local inverse-distance weighting (IDW),
    but only where the missing cell is sufficiently close to valid data.

    Parameters
    ----------
    gridded_data : np.ndarray
        Two-dimensional rectangular array containing the gridded field to be
        infilled.

        Expected format:
        - shape: (n_rows, n_cols)
        - dtype: numeric
        - missing values: represented by `np.nan` or any non-finite value

        The function converts the input to floating point internally, so integer
        arrays with NaN-like sentinels should be converted beforehand if needed.

    n_cells_max : float
        Maximum distance, in units of grid cells, from a missing cell to the
        nearest valid cell for that missing cell to be eligible for infill.

        Interpretation:
        - A value of 1 fills only cells immediately adjacent to valid data
          (including diagonals under Euclidean distance).
        - A value of 3 to 6 is often a reasonable starting range for sparse
          trajectory-like data.
        - Larger values allow the interpolation to spread farther from observed
          data and therefore increase the risk of filling broad empty regions.

    k : int, default=8
        Number of nearest valid observed cells used in the IDW interpolation.

        Interpretation:
        - Smaller `k` keeps the interpolation more local and less smoothed.
        - Larger `k` produces smoother fills but may blur across unrelated
          nearby structures.
        - The effective number of neighbors is automatically reduced if fewer
          than `k` valid cells exist in the whole grid.

    power : float, default=2.0
        Exponent used in inverse-distance weighting.

        The weight assigned to a neighbor at distance d is proportional to

            1 / d**power

        Interpretation:
        - `power = 1` gives relatively broad influence.
        - `power = 2` is a common default.
        - Larger values make the nearest observed cells dominate more strongly.

    eps : float, default=1e-12
        Small positive number used to avoid division by zero when computing
        inverse-distance weights.

    Returns
    -------
    filled : np.ndarray
        Copy of the input data, converted to floating point, with eligible
        missing cells infilled and all other cells unchanged.

    eligible_mask : np.ndarray of bool
        Boolean array with the same shape as `gridded_data`.
        True where a missing cell was considered eligible for infill based on
        the distance threshold.
        False elsewhere.

    Algorithm
    ---------
    The procedure is:

    1. Identify valid cells:
       Cells with finite values are treated as observed data.

    2. Compute distance-to-data:
       For every cell in the grid, compute the Euclidean distance in
       row/column index space to the nearest valid cell.

    3. Identify fillable missing cells:
       A missing cell is marked eligible if its distance to the nearest valid
       cell is less than or equal to `n_cells_max`.

    4. Build a nearest-neighbor search structure:
       The coordinates of all valid cells are inserted into a KD-tree.

    5. Interpolate each eligible missing cell:
       Query the `k` nearest valid cells and compute the weighted mean of their
       values using inverse-distance weights.

    6. Write results back:
       Replace only the eligible missing cells with their interpolated values.

    Important limitations
    ---------------------
    - Distances are measured in array-index space, not physical distance.
      If your grid spacing differs in the two dimensions, or if your axes are
      latitude and longitude with substantial meridional distortion, you may
      want to rescale coordinates before interpolation or use a different
      approach.
    - This method is local and geometric. It does not explicitly infer track
      direction or require that a missing cell lie "between" observations.
    - A cell can be filled even if all its nearby support comes from one side.
      For some trajectory applications, that may allow modest outward spread.
      A stricter "multi-direction support" criterion could be added if needed.

    Why this method is often a good default for aircraft-trajectory-like data
    -------------------------------------------------------------------------
    The hard distance cutoff prevents the interpolation from invading large
    empty regions, while the local IDW step can still bridge narrow gaps
    between nearby observed features. This balance is often preferable to
    global interpolators for sparse observational coverage.

    Examples
    --------
    Basic use:

    >>> filled, eligible = infill_near_data_idw(data, n_cells_max=5)

    More local behavior:

    >>> filled, eligible = infill_near_data_idw(
    ...     data,
    ...     n_cells_max=3,
    ...     k=4,
    ...     power=2.5,
    ... )

    Raises
    ------
    ValueError
        If `gridded_data` is not 2D, if `k` is less than 1, or if
        `n_cells_max` is negative.
    """
    # -----------------------------
    # Input validation
    # -----------------------------
    data = np.asarray(gridded_data, dtype=float)

    if data.ndim != 2:
        raise ValueError("gridded_data must be a 2D array.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if n_cells_max < 0:
        raise ValueError("n_cells_max must be non-negative.")

    # -----------------------------
    # Identify valid and missing cells
    # -----------------------------
    # Any finite value is considered observed data.
    valid_mask = np.isfinite(data)
    missing_mask = ~valid_mask

    # Start with a copy of the input so the original is never modified.
    filled = data.copy()

    # If there are no valid cells at all, nothing can be infilled.
    if not np.any(valid_mask):
        eligible_mask = np.zeros_like(data, dtype=bool)
        return filled, eligible_mask

    # -----------------------------
    # Compute distance to nearest valid cell
    # -----------------------------
    # distance_transform_edt computes distance to the nearest zero-valued cell.
    # By passing ~valid_mask:
    #   - valid cells become False (0)
    #   - missing cells become True  (1)
    # so the returned distance at each location is the Euclidean distance to
    # the nearest valid cell.
    dist_to_valid = distance_transform_edt(~valid_mask)

    # Only missing cells within the specified radius are eligible for infill.
    eligible_mask = missing_mask & (dist_to_valid <= n_cells_max)

    # If no missing cells pass the eligibility criterion, return unchanged data.
    if not np.any(eligible_mask):
        return filled, eligible_mask

    # -----------------------------
    # Extract coordinates of valid cells
    # -----------------------------
    # Each coordinate is stored as [row_index, col_index].
    valid_coords = np.column_stack(np.nonzero(valid_mask))
    valid_values = data[valid_mask]

    # Coordinates of the missing cells that we will attempt to fill.
    target_coords = np.column_stack(np.nonzero(eligible_mask))

    # -----------------------------
    # Build KD-tree for fast nearest-neighbor lookup
    # -----------------------------
    tree = cKDTree(valid_coords)

    # If there are fewer valid cells than k, use all available valid cells.
    k_eff = min(k, len(valid_coords))

    # Query the k nearest observed cells for every eligible target cell.
    dists, inds = tree.query(target_coords, k=k_eff)

    # If k_eff == 1, scipy returns 1D arrays; convert them to 2D arrays so that
    # the later code can treat both cases uniformly.
    if k_eff == 1:
        dists = dists[:, None]
        inds = inds[:, None]

    neighbor_vals = valid_values[inds]

    # -----------------------------
    # Compute inverse-distance-weighted interpolation
    # -----------------------------
    # Exact matches should not normally occur here because targets are missing
    # cells and neighbors are valid cells, but this safeguard keeps the function
    # numerically well-defined in edge cases.
    exact = dists < eps

    # Standard IDW weights: w = 1 / d**power
    # max(d, eps) avoids division by zero.
    weights = 1.0 / np.maximum(dists, eps) ** power

    # For exact matches, we temporarily zero the weights and handle those rows
    # explicitly below.
    weights[exact] = 0.0

    weighted_sum = np.sum(weights * neighbor_vals, axis=1)
    weight_total = np.sum(weights, axis=1)

    # Avoid division by zero with max(weight_total, eps).
    interp_vals = weighted_sum / np.maximum(weight_total, eps)

    # If an exact match exists in a row, overwrite with the exact observed value.
    if np.any(exact):
        row_has_exact = np.any(exact, axis=1)
        first_exact_col = np.argmax(exact, axis=1)
        interp_vals[row_has_exact] = neighbor_vals[
            np.arange(len(target_coords))[row_has_exact],
            first_exact_col[row_has_exact],
        ]

    # -----------------------------
    # Write interpolated values back into the output grid
    # -----------------------------
    filled[eligible_mask] = interp_vals

    return filled, eligible_mask


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Minimal self-contained example
    # ------------------------------------------------------------
    # Create a small array with missing values.
    data = np.array(
        [
            [np.nan, np.nan, 10.0,   np.nan, np.nan],
            [np.nan, 11.0,   np.nan, 13.0,   np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [20.0,   np.nan, 22.0,   np.nan, 24.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=float,
    )

    filled, eligible = infill_near_data_idw(
        data,
        n_cells_max=2.0,
        k=4,
        power=2.0,
    )

    np.set_printoptions(precision=3, suppress=True)

    print("Original data:")
    print(data)
    print("\nEligible missing cells (True means fillable):")
    print(eligible)
    print("\nFilled data:")
    print(filled)

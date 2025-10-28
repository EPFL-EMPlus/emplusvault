"""Lightweight co-ranking utilities to avoid the pycoranking build dependency.

The original project (https://github.com/samueljackson92/coranking) ships a
compiled extension which fails to build under Poetry's isolated environments
because it requires NumPy headers during the build step. The functions below
reimplement the small subset of behaviours we rely on (co-ranking matrix,
trustworthiness, continuity, LCMC) using NumPy only.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import distance


def coranking_matrix(high_data: np.ndarray, low_data: np.ndarray) -> np.ndarray:
    """Construct the co-ranking matrix between high and low dimensional data."""
    high = np.asarray(high_data)
    low = np.asarray(low_data)
    if high.shape[0] != low.shape[0]:
        raise ValueError("high_data and low_data must have the same length")

    # Pairwise distances then convert to rank matrices.
    high_dist = distance.squareform(distance.pdist(high))
    low_dist = distance.squareform(distance.pdist(low))
    high_rank = np.argsort(np.argsort(high_dist, axis=1), axis=1)
    low_rank = np.argsort(np.argsort(low_dist, axis=1), axis=1)

    # Histogram-based co-ranking matrix; discard self rankings.
    q_matrix, _, _ = np.histogram2d(
        high_rank.ravel(),
        low_rank.ravel(),
        bins=high.shape[0],
    )
    return q_matrix[1:, 1:].astype(np.int64, copy=False)


def trustworthiness(Q: np.ndarray, min_k: int = 1, max_k: int | None = None) -> np.ndarray:
    """Compute trustworthiness values for K in [min_k, max_k)."""
    matrix = _ensure_int_matrix(Q)
    n = matrix.shape[0]
    upper = n if max_k is None else max_k

    return np.array([_trustworthiness_k(matrix, k) for k in range(min_k, upper)])


def continuity(Q: np.ndarray, min_k: int = 1, max_k: int | None = None) -> np.ndarray:
    """Compute continuity values for K in [min_k, max_k)."""
    matrix = _ensure_int_matrix(Q)
    n = matrix.shape[0]
    upper = n if max_k is None else max_k

    return np.array([_continuity_k(matrix, k) for k in range(min_k, upper)])


def LCMC(Q: np.ndarray, min_k: int = 1, max_k: int | None = None) -> np.ndarray:  # noqa: N802
    """Compute Local Continuity Meta-Criteria values for K in [min_k, max_k)."""
    matrix = _ensure_int_matrix(Q)
    n = matrix.shape[0]
    upper = n if max_k is None else max_k

    return np.array([_lcmc_k(matrix, k) for k in range(min_k, upper)])


# Internal helpers -----------------------------------------------------------------


def _ensure_int_matrix(Q: np.ndarray) -> np.ndarray:
    matrix = np.asarray(Q)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Co-ranking matrix must be square.")
    if matrix.dtype != np.int64:
        matrix = matrix.astype(np.int64)
    return matrix


def _trustworthiness_k(matrix: np.ndarray, k: int) -> float:
    n = matrix.shape[0]
    if not 0 <= k < n:
        raise ValueError("k must be between 0 and the matrix size.")
    if k == 0:
        return 1.0

    norm_weight = _tc_normalisation_weight(k, n + 1)
    if norm_weight == 0:
        return 1.0

    weights = np.arange(k, n, dtype=np.float64) + 1 - k
    row_sums = matrix[k:, :k].sum(axis=1, dtype=np.float64)
    penalty = (2.0 / norm_weight) * np.dot(weights, row_sums)
    return 1.0 - penalty


def _continuity_k(matrix: np.ndarray, k: int) -> float:
    n = matrix.shape[0]
    if not 0 <= k < n:
        raise ValueError("k must be between 0 and the matrix size.")
    if k == 0:
        return 1.0

    norm_weight = _tc_normalisation_weight(k, n + 1)
    if norm_weight == 0:
        return 1.0

    weights = np.arange(k, n, dtype=np.float64) + 1 - k
    penalty = (2.0 / norm_weight) * (matrix[:k, k:] * weights).sum(dtype=np.float64)
    return 1.0 - penalty


def _tc_normalisation_weight(k: int, n: int) -> float:
    if k < n / 2:
        return n * k * (2 * n - 3 * k - 1)
    return n * (n - k) * (n - k)


def _lcmc_k(matrix: np.ndarray, k: int) -> float:
    n = matrix.shape[0]
    if not 0 <= k < n:
        raise ValueError("k must be between 0 and the matrix size.")
    if k == 0:
        return 0.0

    leading_block = matrix[:k, :k].sum(dtype=np.float64)
    return (k / (1.0 - n)) + (leading_block / (n * k))


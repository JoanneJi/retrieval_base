"""
Normalization utilities for spectral data. Including:
    - simplistic: using the medium of each order in each detector
    - TBD
"""

import numpy as np

def simplistic_normalization(flux, err=None, axis=-1):
    """
    Simplistic normalization using the median value. Each spectrum (e.g., each order in each detector) is divided by its median flux along the given axis, usually the pixel axis.
    * We pre-normalize each spectral segment by its median value and allow for an additional linear scaling parameter in the likelihood.

    Parameters:
        flux (np.ndarray): flux array to be normalized, shape (len(orders), len(dets), n_pixels)
        err (np.ndarray or None): error array associated with flux, shape (len(orders), len(dets), n_pixels)
        axis (int): axis along which to compute the median (default: last axis).

    Returns:
        flux_norm (np.ndarray): median-normalized flux, shape same as input flux
        err_norm (np.ndarray or None): median-normalized error (only if err is provided), shape same as input err
    """
    flux = np.asarray(flux).copy()

    if err is not None:
        err = np.asarray(err).copy()

    median = np.nanmedian(flux, axis=axis, keepdims=True)

    # protect against zero or NaN median
    bad = ~np.isfinite(median) | (median == 0)
    if np.any(bad):
        raise ValueError("Median normalization failed: zero or non-finite median encountered.")

    flux /= median

    if err is not None:
        err /= median
        return flux, err

    return flux

"""
Normalization utilities for spectral data. Including:
    - simplistic: using the medium of each order in each detector
    - low-resolution: normalize by low-resolution continuum
    - median_highpass: normalize using median high-pass filter
"""

import numpy as np
from scipy.ndimage import median_filter
from utils.spectral import convolve_to_resolution
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

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


def low_resolution_normalization(wave, flux, err=None, out_res=100):
    """
    Normalize flux by dividing by low-resolution continuum.
    
    This function convolves the flux to low resolution (out_res) to get the continuum,
    then divides the original flux by this continuum.
    
    Parameters:
        wave (np.ndarray): wavelength array [nm], shape (n_pixels,)
        flux (np.ndarray): flux array to be normalized, shape (n_pixels,)
        err (np.ndarray or None): error array associated with flux, shape (n_pixels,)
        out_res (float): target output resolution for continuum estimation (default: 100)
    
    Returns:
        flux_norm (np.ndarray): normalized flux (flux / continuum), shape (n_pixels,)
        err_norm (np.ndarray or None): normalized error (only if err is provided), shape (n_pixels,)
    """
    flux = np.asarray(flux).copy()
    wave = np.asarray(wave)
    
    if err is not None:
        err = np.asarray(err).copy()
    
    # Get continuum by convolving to low resolution
    continuum = convolve_to_resolution(wave, flux, out_res=out_res)
    
    # Protect against zero or NaN continuum
    bad = ~np.isfinite(continuum) | (continuum == 0)
    if np.any(bad):
        # Replace bad values with median of good values
        good_continuum = continuum[~bad]
        if len(good_continuum) > 0:
            continuum[bad] = np.nanmedian(good_continuum)
        else:
            # Fallback to simplistic normalization
            continuum = np.full_like(continuum, np.nanmedian(flux))
    
    # Normalize: divide by continuum
    flux_norm = flux / continuum
    
    if err is not None:
        err_norm = err / continuum
        return flux_norm, err_norm
    
    return flux_norm, continuum


def median_highpass_normalization(wave, flux, err=None, window=600):
    """
    Normalize flux using median high-pass filtering.
    
    This follows the KPIC-style normalization:
    - estimate low-frequency component with a median filter
    - normalize by dividing flux by the low-frequency component
    
    Parameters:
        wave (np.ndarray): wavelength array [nm], shape (n_pixels,)
        flux (np.ndarray): flux array to be normalized, shape (n_pixels,)
        err (np.ndarray or None): error array associated with flux, shape (n_pixels,)
        window (int): window size for median filter (default: 600 for CRIRES+ K band, or equivalent for 5nm)
    
    Returns:
        flux_norm (np.ndarray): normalized flux (flux / low_freq), shape (n_pixels,)
        err_norm (np.ndarray or None): normalized error (only if err is provided), shape (n_pixels,)
    """
    flux = np.asarray(flux).copy()
    wave = np.asarray(wave)
    
    if err is not None:
        err = np.asarray(err).copy()
    
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    # Estimate low-frequency component with median filter
    low_freq = median_filter(flux, size=window, mode="reflect")
    
    # Protect against zero or NaN low_freq
    bad = ~np.isfinite(low_freq) | (low_freq == 0)
    if np.any(bad):
        # Replace bad values with median of good values
        good_low_freq = low_freq[~bad]
        if len(good_low_freq) > 0:
            low_freq[bad] = np.nanmedian(good_low_freq)
        else:
            # Fallback to simplistic normalization
            low_freq = np.full_like(low_freq, np.nanmedian(flux))
    
    # Normalize: divide by low-frequency component
    flux_norm = flux / low_freq
    
    if err is not None:
        err_norm = err / low_freq
        return flux_norm, err_norm
    
    return flux_norm

def gaussian_lfp(wave, flux, err=None, sigma_px=200):
    """
    Gaussian low-frequency pattern extraction
    sigma_px : Gaussian sigma in *pixel space*
    """

    flux = np.asarray(flux).copy()
    wave = np.asarray(wave)
    
    flux_lfp = gaussian_filter1d(flux, sigma=sigma_px, mode="nearest")

    flux_norm = flux / flux_lfp

    if err is not None:
        err_norm = err / flux_lfp
        return flux_norm, err_norm

    return flux_norm

def savgol_lfp(wave, flux, err=None, window_length=1301, polyorder=2):
    """
    Savitzky-Golay low-frequency pattern extraction

    window_length : SG window length in *pixel space* (must be odd)
    polyorder     : polynomial order for local fitting
    """

    flux = np.asarray(flux).copy()
    wave = np.asarray(wave)

    # low-frequency pattern via local polynomial reconstruction
    flux_lfp = savgol_filter(
        flux,
        window_length=window_length,
        polyorder=polyorder,
        mode="interp"
    )

    flux_norm = flux / flux_lfp

    if err is not None:
        err_norm = err / flux_lfp
        return flux_norm, err_norm

    return flux_norm


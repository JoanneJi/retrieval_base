"""
Spectral processing utilities for retrieval framework.

Provides functions for:
- Instrumental broadening
- Resolution convolution
- Spectral manipulation
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from astropy import units as u


def convolve_to_resolution(wave, flux, out_res, in_res=None):
    """
    Convolve spectrum to a target resolution, accounting for input resolution.
    
    This function handles the case where the input spectrum already has finite
    resolution. It uses error propagation to compute the additional broadening
    needed: 1/R_out² = 1/R_in² + 1/R_add²
    
    Args:
        wave (np.ndarray): Wavelength array [nm]
        flux (np.ndarray): Flux array
        out_res (float): Target output resolution (R = λ/Δλ)
        in_res (float, optional): Input resolution. If None, automatically computed
                                 from wavelength spacing.
    
    Returns:
        np.ndarray: Convolved flux array
    """
    # Handle astropy Quantity if needed
    if isinstance(wave, u.Quantity):
        wave = wave.to(u.Unit("nm")).value
    
    # Auto-compute input resolution if not provided
    if in_res is None:
        # Resolution from wavelength spacing: R = λ / Δλ
        in_res = np.mean(wave[:-1] / np.diff(wave))
    
    # Compute additional broadening needed
    # Using error propagation: 1/R_out² = 1/R_in² + 1/R_add²
    # So: R_add = 1/√(1/R_out² - 1/R_in²)
    # And sigma_LSF = λ/(R_add × 2√(2ln2))
    sigma_LSF = np.sqrt(1.0 / out_res**2 - 1.0 / in_res**2) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    # Compute wavelength spacing (log-spaced grid)
    spacing = np.mean(2.0 * np.diff(wave) / (wave[1:] + wave[:-1]))
    
    # Convert to pixels for gaussian filter
    sigma_LSF_gauss_filter = sigma_LSF / spacing
    
    # Handle NaN values
    out_flux = np.full_like(flux, np.nan)
    nans = np.isnan(flux)
    
    # Apply gaussian filter only to non-NaN values
    if not nans.all():
        out_flux[~nans] = gaussian_filter(
            flux[~nans],
            sigma=sigma_LSF_gauss_filter,
            mode='reflect'
        )
    
    return out_flux


def instr_broadening(wave, flux, out_res=1e6, in_res=1e6):
    """
    Apply instrumental broadening from input resolution to output resolution.
    
    This is a simpler interface that explicitly takes both input and output
    resolutions. It's equivalent to convolve_to_resolution but with explicit
    in_res parameter.
    
    Args:
        wave (np.ndarray): Wavelength array [nm]
        flux (np.ndarray): Flux array
        out_res (float): Target output resolution (R = λ/Δλ). Default: 1e6
        in_res (float): Input resolution (R = λ/Δλ). Default: 1e6
    
    Returns:
        np.ndarray: Broadened flux array
    """
    # Handle astropy Quantity if needed
    if isinstance(wave, u.Quantity):
        wave = wave.to(u.Unit("nm")).value
    
    # Check if out_res >= in_res (cannot broaden to higher resolution)
    if out_res >= in_res:
        import warnings
        warnings.warn(
            f"out_res ({out_res}) >= in_res ({in_res}). "
            "Cannot broaden to higher resolution. Returning original flux.",
            UserWarning
        )
        return flux.copy()
    
    # Compute additional broadening needed
    # Delta lambda of resolution element is FWHM of the LSF's standard deviation
    # Using error propagation: 1/R_out² = 1/R_in² + 1/R_add²
    diff = 1.0 / out_res**2 - 1.0 / in_res**2
    if diff <= 0:
        import warnings
        warnings.warn(
            f"Cannot compute broadening: 1/out_res² - 1/in_res² = {diff:.2e} <= 0. "
            "Returning original flux.",
            UserWarning
        )
        return flux.copy()
    
    sigma_LSF = np.sqrt(diff) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    # Compute wavelength spacing (log-spaced grid)
    # This formula works for both linear and log-spaced grids
    spacing = np.mean(2.0 * np.diff(wave) / (wave[1:] + wave[:-1]))
    
    # Calculate the sigma to be used in the gauss filter in pixels
    sigma_LSF_gauss_filter = sigma_LSF / spacing
    
    # Check if sigma is too small to have any effect
    if sigma_LSF_gauss_filter < 0.1:
        import warnings
        warnings.warn(
            f"sigma_LSF_gauss_filter ({sigma_LSF_gauss_filter:.6f} pixels) is too small. "
            f"Gaussian filter will have negligible effect. "
            f"out_res={out_res:.0f}, in_res={in_res:.0f}. "
            "Consider using a larger resolution difference.",
            UserWarning
        )
    
    # Apply gaussian filter to broaden with the spectral resolution
    flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter, mode='nearest')
    
    return flux_LSF


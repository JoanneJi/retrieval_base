"""
Preprocessing module for the input.
    - Choose the range of wavelengths to be used in the retrieval.
    - Flatten the spectrum.
    - normalization
"""

import numpy as np
from utils.normalization import (
    simplistic_normalization,
    low_resolution_normalization,
    median_highpass_normalization,
    gaussian_lfp,
    savgol_lfp
)

def select_order_and_flatten(wave, flux, err, orders, dets, normalize=True):
    """
    Select specific orders and detectors you want to use, then flatten the arrays.

    Parameters:
        wave (np.ndarray): wavelength [nm], shape (n_orders, n_dets, n_pixels)
        flux (np.ndarray): flux [original], shape (n_orders, n_dets, n_pixels)
        err (np.ndarray): error [original], shape (n_orders, n_dets, n_pixels)
        orders (list of int): index of orders to select, e.g., [0, 1, 2]
        dets (list of int): index of detectors to select, e.g., [0, 1]
        normalize (bool): whether to normalize the spectrum

    Returns:
        wave (np.ndarray): wavelength [nm], shape (n_selected_pixels,)
        flux (np.ndarray): flux [normalized if selected], shape (n_selected_pixels,)
        err (np.ndarray): error [normalized if selected], shape (n_selected_pixels,)
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    err  = np.asarray(err)

    wave = wave[np.ix_(orders, dets)]
    flux = flux[np.ix_(orders, dets)]
    err  = err[np.ix_(orders, dets)]

    # Remove bad pixels BEFORE normalization (per order, det independently)
    n_orders, n_dets, n_pixels = wave.shape
    wave_clean = []
    flux_clean = []
    err_clean = []
    
    for i in range(n_orders):
        for j in range(n_dets):
            # Remove bad pixels for this (order, det) combination
            # firstly remove the NaN/inf, and negative flux
            mask = ~np.isfinite(flux[i, j]) | (flux[i, j] < 0)
            # then remove the pixels with error larger than 5 times the median error
            mask = mask | (err[i, j] > 5 * np.nanmedian(err[i, j]))
            # then remove bad pixels, the flux is larger/smaller than 1*median flux
            mask = mask | (np.abs(flux[i, j] - np.nanmedian(flux[i, j])) > 1 * np.nanmedian(flux[i, j]))
            
            wave_clean.append(wave[i, j][~mask])
            flux_clean.append(flux[i, j][~mask])
            err_clean.append(err[i, j][~mask])
    
    # Flatten all clean data
    wave = np.concatenate(wave_clean)
    flux = np.concatenate(flux_clean)
    err  = np.concatenate(err_clean)

    if normalize:
        # Normalize the cleaned data
        # Since data is already flattened, normalize the entire array by its median
        median = np.nanmedian(flux)
        if median > 0 and np.isfinite(median):
            flux = flux / median
            err = err / median
        else:
            raise ValueError("Normalization failed: zero or non-finite median encountered after cleaning.")

    return wave, flux, err

def select_orders_chips(wave, flux, err, orders, dets, normalize=True, normalize_method='simplistic_normalization', remove_metal_lines=False):
    """
    Select specific orders and detectors, keeping the chip structure (not flattened).
    Each (order, det) combination becomes a chip.
    For star A, remove the metal line regions. (2178.5-2179.5 nm; 2261-2262 nm; 2262.5-2264 nm; 2265.5-2266.5 nm)

    Parameters:
        wave (np.ndarray): wavelength [nm], shape (n_orders, n_dets, n_pixels)
        flux (np.ndarray): flux [original], shape (n_orders, n_dets, n_pixels)
        err (np.ndarray): error [original], shape (n_orders, n_dets, n_pixels)
        orders (list of int): index of orders to select, e.g., [5, 6]
        dets (list of int): index of detectors to select, e.g., [0, 1, 2]
        normalize (bool): whether to normalize the spectrum
        normalize_method (str): normalization method to use. Options:
            - 'simplistic_normalization': divide by median (default)
            - 'low-resolution': divide by low-resolution continuum (out_res=100)
            - 'median_highpass': divide by median-filtered low-frequency component
            - 'gaussian_lfp': divide by Gaussian-filtered low-frequency component
            - 'savgol_lfp': divide by Savgol-filtered low-frequency component
        remove_metal_lines (bool): whether to remove the metal line regions. Default is False.
    Returns:
        wave_chips (list): list of wavelength arrays, one per chip, shape (n_chips, n_pixels_per_chip)
        flux_chips (list): list of flux arrays, one per chip, shape (n_chips, n_pixels_per_chip)
        err_chips (list): list of error arrays, one per chip, shape (n_chips, n_pixels_per_chip)
        wave_ranges_chips (np.ndarray): wavelength ranges for each chip, shape (n_chips, 2) [min, max]
    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    err  = np.asarray(err)

    # Select orders and detectors
    wave = wave[np.ix_(orders, dets)]
    flux = flux[np.ix_(orders, dets)]
    err  = err[np.ix_(orders, dets)]

    n_orders, n_dets, n_pixels = wave.shape
    n_chips = n_orders * n_dets
    
    # Remove bad pixels BEFORE normalization (per order, det independently)
    wave_chips = []
    flux_chips = []
    err_chips = []
    
    for i in range(n_orders):
        for j in range(n_dets):
            # Remove bad pixels for this (order, det) combination
            # firstly remove the NaN/inf, and negative flux
            mask = ~np.isfinite(flux[i, j]) | (flux[i, j] < 0)
            # then remove the pixels with error larger than 5 times the median error
            mask = mask | (err[i, j] > 5 * np.nanmedian(err[i, j]))
            # then remove bad pixels, the flux is larger/smaller than 1*median flux
            mask = mask | (np.abs(flux[i, j] - np.nanmedian(flux[i, j])) > 1 * np.nanmedian(flux[i, j]))
            
            wave_chip = wave[i, j][~mask]
            flux_chip = flux[i, j][~mask]
            err_chip = err[i, j][~mask]
            
            if remove_metal_lines:
                # Remove the metal line regions. (2178.5-2179.5 nm; 2261-2262 nm; 2262.5-2264 nm; 2265.5-2266.5 nm)
                mask = ((wave_chip > 2178.5) & (wave_chip < 2179.5)) | ((wave_chip > 2261) & (wave_chip < 2262)) | ((wave_chip > 2262.5) & (wave_chip < 2264)) | ((wave_chip > 2265.5) & (wave_chip < 2266.5))
                wave_chip = wave_chip[~mask]
                flux_chip = flux_chip[~mask]
                err_chip = err_chip[~mask]
            
            # Only keep chips with valid data
            if len(wave_chip) > 0:
                # Apply normalization to cleaned data if requested
                if normalize:
                    if normalize_method == 'simplistic_normalization':
                        # For 1D array, normalize by median
                        median = np.nanmedian(flux_chip)
                        if median > 0 and np.isfinite(median):
                            flux_chip = flux_chip / median
                            err_chip = err_chip / median
                        else:
                            raise ValueError(f"Normalization failed for chip ({i}, {j}): zero or non-finite median encountered.")
                    elif normalize_method == 'low-resolution':
                        flux_chip, err_chip = low_resolution_normalization(
                            wave_chip, flux_chip, err_chip, out_res=100
                        )
                    elif normalize_method == 'median_highpass':
                        flux_chip, err_chip = median_highpass_normalization(
                            wave_chip, flux_chip, err_chip, window=600  # default window size is 600 for CRIRES+ K band, or equivalent for 5nm
                        )
                    elif normalize_method == 'gaussian_lfp':
                        flux_chip, err_chip = gaussian_lfp(
                            wave_chip, flux_chip, err_chip, sigma_px=200
                        )
                    elif normalize_method == 'savgol_lfp':
                        flux_chip, err_chip = savgol_lfp(
                            wave_chip, flux_chip, err_chip, window_length=1301, polyorder=2
                        )
                    else:
                        raise ValueError(f"Unknown normalize_method: {normalize_method}. "
                                       f"Must be one of: 'simplistic_normalization', 'low-resolution', 'median_highpass', 'gaussian_lfp'")
                
                wave_chips.append(wave_chip)
                flux_chips.append(flux_chip)
                err_chips.append(err_chip)
    
    # Create wavelength ranges for each chip
    wave_ranges_chips = []
    for wave_chip in wave_chips:
        wave_ranges_chips.append([wave_chip.min(), wave_chip.max()])
    
    wave_ranges_chips = np.array(wave_ranges_chips) if len(wave_ranges_chips) > 0 else np.array([]).reshape(0, 2)
    
    return wave_chips, flux_chips, err_chips, wave_ranges_chips

"""
Preprocessing module for the input.
    - Choose the range of wavelengths to be used in the retrieval.
    - Flatten the spectrum.
    - normalization
"""

import numpy as np
from utils.normalization import simplistic_normalization

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

    if normalize:
        # normalize each (order, detector) independently
        flux, err = simplistic_normalization(flux, err, axis=-1)

    wave = wave.reshape(-1)
    flux = flux.reshape(-1)
    err  = err.reshape(-1)

    # remove bad pixels that exceed 4 sigma, as well as NaN/inf
    mask = np.abs(flux - np.nanmedian(flux)) > 5 * np.nanstd(flux)
    mask = mask | ~np.isfinite(flux)  # Also remove NaN/inf
    wave = wave[~mask]
    flux = flux[~mask]
    err  = err[~mask]

    return wave, flux, err

def select_orders_chips(wave, flux, err, orders, dets, normalize=True):
    """
    Select specific orders and detectors, keeping the chip structure (not flattened).
    Each (order, det) combination becomes a chip.

    Parameters:
        wave (np.ndarray): wavelength [nm], shape (n_orders, n_dets, n_pixels)
        flux (np.ndarray): flux [original], shape (n_orders, n_dets, n_pixels)
        err (np.ndarray): error [original], shape (n_orders, n_dets, n_pixels)
        orders (list of int): index of orders to select, e.g., [5, 6]
        dets (list of int): index of detectors to select, e.g., [0, 1, 2]
        normalize (bool): whether to normalize the spectrum

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

    if normalize:
        # normalize each (order, detector) independently
        flux, err = simplistic_normalization(flux, err, axis=-1)

    # Reshape to (n_chips, n_pixels) where n_chips = n_orders * n_dets
    n_orders, n_dets, n_pixels = wave.shape
    n_chips = n_orders * n_dets
    
    wave_2d = wave.reshape(n_chips, n_pixels)
    flux_2d = flux.reshape(n_chips, n_pixels)
    err_2d  = err.reshape(n_chips, n_pixels)

    # Remove bad pixels that exceed 4 sigma (per chip)
    wave_chips = []
    flux_chips = []
    err_chips = []
    wave_ranges_chips = []
    
    for i in range(n_chips):
        # !!! this part can be adjusted according to the data!!!
        # firstly remove the NaN/inf, and negative flux
        mask = ~np.isfinite(flux_2d[i]) | (flux_2d[i] < 0)
        # then remove the pixels with error larger than 3 times the median error
        mask = mask | (err_2d[i] > 5 * np.nanmedian(err_2d[i]))
        # then remove bad pixels, the flux is larger/smaller than 1.5*median flux
        mask = mask | (np.abs(flux_2d[i] - np.nanmedian(flux_2d[i])) > 1.5 * np.nanmedian(flux_2d[i]))
        
        wave_chip = wave_2d[i][~mask]
        flux_chip = flux_2d[i][~mask]
        err_chip = err_2d[i][~mask]
        
        # Only keep chips with valid data
        if len(wave_chip) > 0:
            wave_chips.append(wave_chip)
            flux_chips.append(flux_chip)
            err_chips.append(err_chip)
            wave_ranges_chips.append([wave_chip.min(), wave_chip.max()])
    
    wave_ranges_chips = np.array(wave_ranges_chips) if len(wave_ranges_chips) > 0 else np.array([]).reshape(0, 2)
    
    return wave_chips, flux_chips, err_chips, wave_ranges_chips

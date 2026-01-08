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

    return wave, flux, err

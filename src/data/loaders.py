"""
Loaders for data input. The output should always be three arrays with the same size:
    - wavelength [nm]
    - flux [original]
    - error [original]
"""

import os
import numpy as np

# data directory
from core.paths import DATA_DIR

def load_crires_dat(target, night, savename, n_orders=7, n_dets=3, n_pixels=2048, base_dir=DATA_DIR):
    """
    Loading data from CRIRES+, using the default output 1D spectrum file from excalibuhr

    Parameters:
        target (str): target name, e.g., "CD-35_2722"
        night (str): yyyy-mm-dd
        savename (str): same as in excalibuhr, single word without spaces/hyphen/underscore, e.g., "starA" (not "star A"/"star-A"/"star_A")
        n_orders (int): default as 7 for K2166 
        n_dets (int): default as 3 for K2166
        n_pixels (int): default as 2048 for K2166
        base_dir (str): the output directory
    
    Outputs:
        wave (np.ndarray): wavelength [nm], shape (n_orders, n_dets, n_pixels)
        flux (np.ndarray): flux [original], shape (n_orders, n_dets, n_pixels)
        err (np.ndarray): error [original], shape (n_orders, n_dets, n_pixels)
    """
    filename = f"SPEC_CD-352722_{savename}1D_TELLURIC_CORR_MOLECFIT.dat"
    path = os.path.join(base_dir, target, night, filename)

    data = np.genfromtxt(path, comments="#")

    wave = data[:, 0].reshape(n_orders, n_dets, n_pixels) * 1e3  # μm → nm
    flux = data[:, 1].reshape(n_orders, n_dets, n_pixels)
    err  = data[:, 2].reshape(n_orders, n_dets, n_pixels)

    return wave, flux, err

"""
Atmosphere module for retrieval framework.

Provides:
- Chemistry models (free, equilibrium)
- Temperature-pressure profiles
- Radtrans atmosphere setup utilities
"""

import numpy as np
import pandas as pd
import pathlib
import pickle
import warnings

from petitRADTRANS.radtrans import Radtrans
from core.paths import SRC_DIR


def get_species_from_params(param_dict, species_info_path=None):
    """
    Extract pRT species names from parameter dictionary.
    
    Looks for parameters with 'log_' prefix (excluding 'log_g'),
    then maps them to pRT species names using species_info.csv.
    
    This is a utility function for setting up Radtrans atmosphere objects.
    The actual chemistry calculation is handled by Chemistry classes.
    
    Args:
        param_dict (dict): Parameter dictionary containing log_* parameters
        species_info_path (str, optional): Path to species_info.csv. 
                                          If None, uses default location:
                                          SRC_DIR / "atmosphere" / "species_info.csv"
    
    Returns:
        list: List of pRT species names (e.g., ['12C-16O', '1H2-16O'])
    """
    # Load species_info.csv with default path: SRC_DIR / "atmosphere" / "species_info.csv"
    if species_info_path is None:
        species_info_path = SRC_DIR / "atmosphere" / "species_info.csv"
    
    species_info_path = pathlib.Path(species_info_path)
    if not species_info_path.exists():
        raise FileNotFoundError(
            f"species_info.csv not found at {species_info_path}. "
            "Please ensure the file exists."
        )
    
    species_info = pd.read_csv(species_info_path, index_col=0)
    
    # Extract all log_* parameters (excluding log_g)
    chem_species = []
    for par in param_dict:
        if 'log_' in par and par != 'log_g':
            chem_species.append(par)
    
    # Map to pRT species names
    species = []
    for chemspec in chem_species:
        # Remove 'log_' prefix to get species name
        species_name = chemspec[4:]  # e.g., 'log_12CO' -> '12CO'
        if species_name not in species_info.index:
            warnings.warn(
                f"Species '{species_name}' from parameter '{chemspec}' "
                f"not found in species_info.csv. Skipping."
            )
            continue
        prt_name = species_info.loc[species_name, 'pRT_name']
        species.append(prt_name)
    
    return species


def setup_radtrans_atmosphere(
    species,
    target_wavelengths,
    pressure,
    lbl_opacity_sampling=3,
    wl_pad=7,
    cache_file=None,
    redo=False
):
    """
    Create and optionally cache Radtrans atmosphere objects.
    
    This function handles the setup of Radtrans objects for atmospheric retrieval.
    It determines wavelength boundaries from target data and creates a properly
    configured Radtrans object.
    
    Args:
        species (list): List of pRT species names (e.g., ['12C-16O', '1H2-16O'])
        target_wavelengths (np.ndarray): Target wavelength array [nm]
        pressure (np.ndarray): Pressure grid [bar]
        lbl_opacity_sampling (int): Line-by-line opacity sampling rate
        wl_pad (float): Wavelength padding [nm] for atmosphere object
        cache_file (str, optional): Path to cache file. If None, uses default.
        redo (bool): If True, recreate atmosphere objects even if cached.
    
    Returns:
        Radtrans: Configured Radtrans atmosphere object
    """
    # Default cache file path
    if cache_file is None:
        cache_file = pathlib.Path('atmosphere_objects.pickle')
    else:
        cache_file = pathlib.Path(cache_file)
    
    # Try to load from cache if redo=False
    if cache_file.exists() and not redo:
        try:
            with open(cache_file, 'rb') as f:
                atmosphere_objects = pickle.load(f)
                print(f"Loaded atmosphere objects from cache: {cache_file}")
                return atmosphere_objects
        except Exception as e:
            warnings.warn(f"Failed to load atmosphere objects from cache: {e}. Recreating...")
    
    # Create new atmosphere objects
    print('Creating new atmosphere objects...')
    
    # Determine wavelength range
    wlmin = np.min(target_wavelengths) - wl_pad
    wlmax = np.max(target_wavelengths) + wl_pad
    
    # Convert nm to cm, then to micron for pRT
    wlen_range = np.array([wlmin, wlmax]) * 1e-7  # nm to cm
    boundary = wlen_range * 1e4  # cm to micron
    
    print(f'Wavelength range for atmosphere object: {wlmin:.2f} - {wlmax:.2f} nm')
    print(f'Line species: {species}')
    print(f'Pressure levels: {len(pressure)} layers, range: {pressure.min():.2e} - {pressure.max():.2e} bar')
    print(f'Wavelength boundaries: {boundary[0]:.4f} - {boundary[1]:.4f} micron')
    print(f'Line-by-line opacity sampling: {lbl_opacity_sampling}')
    
    # Create Radtrans object
    atmosphere_objects = Radtrans(
        line_species=species,
        rayleigh_species=['H2', 'He'],
        gas_continuum_contributors=['H2-H2', 'H2-He'],
        wavelength_boundaries=boundary,
        line_opacity_mode='lbl',
        line_by_line_opacity_sampling=lbl_opacity_sampling,
        pressures=pressure
    )
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(atmosphere_objects, f)
        print(f"Saved atmosphere objects to cache: {cache_file}")
    except Exception as e:
        warnings.warn(f"Failed to save atmosphere objects to cache: {e}")
    
    return atmosphere_objects


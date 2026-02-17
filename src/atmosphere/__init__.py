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
    species_colors = {}
    for par in param_dict:
        if 'log_' in par and par != 'log_g':
            chem_species.append(par)
    
    # Map to pRT species names
    species = []
    for chemspec in chem_species:
        # Remove 'log_' prefix to get species name
        species_name = chemspec[4:]  # e.g., 'log_12CO' -> '12CO'
        # but skip if it starts with 'log_P_quench_<>' or 'log_Kzz_chem'
        if species_name.startswith('P_quench_') or species_name.startswith('Kzz_chem') or species_name.startswith('P_base_gray') or species_name.startswith('opa_base_gray'):
            continue
        if species_name not in species_info.index:
            warnings.warn(
                f"Species '{species_name}' from parameter '{chemspec}' "
                f"not found in species_info.csv. Skipping."
            )
            continue
        prt_name = species_info.loc[species_name, 'pRT_name']
        species_colors[prt_name] = str(species_info.loc[species_name, 'color'])
        species.append(prt_name)
    
    return species, species_colors


def setup_radtrans_atmosphere(
    species,
    target_wavelengths,
    pressure,
    lbl_opacity_sampling=3,
    wl_pad=7,
    cache_file=None,
    redo=False,
    chips_mode=False,
    wave_ranges_chips=None,
    star_mode=False
):
    """
    Create and optionally cache Radtrans atmosphere objects.
    
    This function handles the setup of Radtrans objects for atmospheric retrieval.
    It determines wavelength boundaries from target data and creates a properly
    configured Radtrans object. Supports both single continuous spectrum and
    multi-chip (discontinuous) spectra.
    
    Args:
        species (list): List of pRT species names (e.g., ['12C-16O', '1H2-16O'])
        target_wavelengths: Target wavelength(s). Can be:
            - np.ndarray: 1D array for continuous spectrum
            - list of np.ndarray: list of arrays for multi-chip spectrum
        pressure (np.ndarray): Pressure grid [bar]
        lbl_opacity_sampling (int): Line-by-line opacity sampling rate
        wl_pad (float): Wavelength padding [nm] for atmosphere object
        cache_file (str, optional): Path to cache file. If None, uses default.
        redo (bool): If True, recreate atmosphere objects even if cached.
        chips_mode (bool): If True, treat as multi-chip spectrum
        wave_ranges_chips (np.ndarray, optional): Wavelength ranges for each chip,
            shape (n_chips, 2) with [min, max] per chip. Required if chips_mode=True.
        star_mode (bool): If True, add 'H-' to gas_continuum_contributors. Defaults to False.
    
    Returns:
        Radtrans or list of Radtrans: Configured Radtrans atmosphere object(s)
            - Single Radtrans object if chips_mode=False
            - List of Radtrans objects (one per chip) if chips_mode=True
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
    
    if chips_mode:
        # Multi-chip mode: create ONE Radtrans object covering all chips
        # This avoids loading opacity data multiple times
        if wave_ranges_chips is None:
            raise ValueError("wave_ranges_chips must be provided when chips_mode=True")
        
        n_chips = len(wave_ranges_chips)
        
        # Determine overall wavelength range covering all chips (with padding)
        wlmin_all = np.min(wave_ranges_chips[:, 0]) - wl_pad
        wlmax_all = np.max(wave_ranges_chips[:, 1]) + wl_pad
        
        # Convert nm to cm, then to micron for pRT
        wlen_range = np.array([wlmin_all, wlmax_all]) * 1e-7  # nm to cm
        boundary = wlen_range * 1e4  # cm to micron
        
        print(f'Multi-chip mode: Creating single Radtrans object covering all chips')
        print(f'  Overall wavelength range: {wlmin_all:.2f} - {wlmax_all:.2f} nm')
        print(f'  Number of chips: {n_chips}')
        for i, (wlmin_chip, wlmax_chip) in enumerate(wave_ranges_chips):
            print(f'    Chip {i+1}: {wlmin_chip:.2f} - {wlmax_chip:.2f} nm')
        print(f'  Wavelength boundaries: {boundary[0]:.4f} - {boundary[1]:.4f} micron')
        print(f'Line species: {species}')
        print(f'Pressure levels: {len(pressure)} layers, range: {pressure.min():.2e} - {pressure.max():.2e} bar')
        print(f'Line-by-line opacity sampling: {lbl_opacity_sampling}')
        
        # Create single Radtrans object covering all chips
        # Set gas_continuum_contributors based on star_mode
        gas_continuum = ['H2-H2', 'H2-He']
        if star_mode:
            gas_continuum.append('H-')
        
        atmosphere_objects = Radtrans(
            line_species=species,
            rayleigh_species=['H2', 'He'],
            gas_continuum_contributors=gas_continuum,
            wavelength_boundaries=boundary,
            line_opacity_mode='lbl',
            line_by_line_opacity_sampling=lbl_opacity_sampling,
            pressures=pressure
        )
        
        # Store wave_ranges_chips for later use in make_spectrum
        atmosphere_objects.wave_ranges_chips = wave_ranges_chips
        atmosphere_objects.wl_pad = wl_pad

    else:
        # Single continuous spectrum mode (original behavior)
        # Determine wavelength range
        if isinstance(target_wavelengths, list):
            # Flatten if list
            target_wavelengths = np.concatenate(target_wavelengths)
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
        # Set gas_continuum_contributors based on star_mode
        gas_continuum = ['H2-H2', 'H2-He']
        if star_mode:
            gas_continuum.append('H-')
        
        atmosphere_objects = Radtrans(
            line_species=species,
            rayleigh_species=['H2', 'He'],
            gas_continuum_contributors=gas_continuum,
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


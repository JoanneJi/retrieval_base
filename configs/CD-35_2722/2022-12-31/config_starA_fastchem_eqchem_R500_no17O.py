"""
Configuration file for equilibrium chemistry retrieval.

This configuration uses pRT v3 equilibrium chemistry tables.
Equilibrium chemistry calculates species abundances from elemental ratios (C/O, Fe/H, etc.) and temperature-pressure conditions.

"""
import numpy as np


# ========== File paths ==========



# ========== Priors ==========

# ---------- Sampling options (optional) ----------
# inverse_flag: If True (default), temperature knots are sampled from prior only.
#   If False, sampling enforces T_4 < T_3 < T_2 < T_1 < T_0 (monotonic TP with altitude).
# Omit this variable to use the default (True) or the value passed to Parameters().
inverse_flag = False

# ---------- Constant parameters ----------
constant_params = {
    # ----- radial velocity and surface gravity -----
    # 'rv': 32,      # radial velocity [km/s] - used for Doppler shifting the spectrum
    # 'log_g': 4.5,  # log10 of surface gravity [cm/s^2] - used for pRT spectrum calculation
    
    # ----- base temperature for gradient mode -----
    # Required when using TP_mode='gradient' (one of the following):
    # 'T_phot': 1500.,  # photospheric temperature [K]
    # 'T_0': 1500.,  # base temperature [K] (alternative to T_phot)
    # 'P_phot': 1.0,  # photospheric pressure [bar] (optional, defaults to max pressure)

    # ----- pressure knots, choose one of the below -----
    # # case 1: tighter at the bottom/middle, and more sparse at the top
    # 'log_P_knots': [-6, -3, -1.25, -0.25, 0.5, 1, 1.5, 2],
    # # case 2: default equally-spaced knots (set n_knots in TP_knots, no params needed here)
    # # case 3: defining P_phot (the most sensitive part) -> [-5, 0, 2]
    # 'log_P_phot': 0.0,
    # # case 4: defining separations around bottom P (care about the bottom atmosphere) -> [-5, -0.5, 1, 2]
    # 'd_log_P_0+1': 1.0,  # log_P_base=2., so knot at 2. - 1. = 1.
    # 'd_log_P_0+2': 2.5,  # log_P_base=2., so knot at 2. - 2.5 = -0.5
    # # case 5: defining separations around P_phot (care about the photosphere) -> [-5, -2, -1, 0, 1.5, 2]
    # 'log_P_phot': 0.0,  # must given if using d_log_P_phot+{i}
    # 'd_log_P_phot+1': 1.0,  # log_P_base=0., so knot at 0. - 1. = -1.
    # 'd_log_P_phot+2': 2.0,  # log_P_base=0., so knot at 0. - 2. = -2.
    # 'd_log_P_phot-1': 1.5,  # log_P_base=0., so knot at 0. + 1.5 = 1.5

    
}


# ---------- Free parameters ----------
free_params = {
    # ----- full format of the input -----
    # 'T0' : {'bounds': [0, 5000], 'label': r'$T_0$', 'type': 'uniform'},
    # ... but make sure all the parameters are using the same format

    # ----- radial velocity & surface gravity -----
    'rv': ([0, 40], r'$v_{\rm rad}$', 'uniform'),  # radial velocity [km/s]
    'log_g': ([3.5, 5.5], r'log $g$', 'uniform'),  # log10 of surface gravity [cm/s^2]
    # 'log_g': ([4.3, 0.3], r'log $g$', 'normal'),  # log10 of surface gravity [cm/s^2]

    # ----- temperature profile -----
    # case 1: 5 temperature knots -> T0, T1, T2, T3, T4
    'T_0' : ([3000,15000], r'$T_0$', 'uniform'), # bottom of the atmosphere (usually hotter)
    'T_1' : ([3000,5000], r'$T_1$', 'uniform'),
    'T_2' : ([1000,5000], r'$T_2$', 'uniform'),
    'T_3' : ([1000,5000], r'$T_3$', 'uniform'),
    'T_4' : ([1000,5000], r'$T_4$', 'uniform'), # top of atmosphere (usually cooler)
    # # case 2: 5 temperature gradients -> dlnT_dlnP_{i} (for TP_mode='gradient')
    # # Note: Also need T_phot or T_0 in constant_params as base temperature
    # 'dlnT_dlnP_0' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_0$', 'uniform'), # bottom of the atmosphere (usually hotter)
    # 'dlnT_dlnP_1' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_1$', 'uniform'),
    # 'dlnT_dlnP_2' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_2$', 'uniform'),
    # 'dlnT_dlnP_3' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_3$', 'uniform'),
    # 'dlnT_dlnP_4' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_4$', 'uniform'), # top of atmosphere (usually cooler)

    # ----- stellar/planetary parameters -----
    'vsini': ([0, 40], r'$v\sin i$', 'uniform'),  # projected rotational velocity [km/s] - used for rotational broadening
    
    # ----- equilibrium chemistry parameters -----
    # Required parameters for equilibrium chemistry:
    'C/O': ([0.1, 1.6], r'C/O', 'uniform'),  # Carbon to oxygen ratio (required)
    'Fe/H': ([-1.5, 1.5], r'Fe/H', 'uniform'),  # Metallicity in log10 units (required)
    
    # Optional parameters for equilibrium chemistry:
    # 'N/O': ([0.05, 0.5], r'N/O', 'uniform'),  # Nitrogen to oxygen ratio (optional)
    
    # Optional isotope ratios (for isotope calculations):
    '12C/13C': ([30, 250], r'$^{12}$C/$^{13}$C', 'uniform'),  # 12C/13C isotope ratio (optional), or '12/13C_ratio' for backward compatibility
    # '16/17O_ratio': ([200, 1000], r'$^{16}$O/$^{17}$O', 'uniform'),  # 16O/17O isotope ratio (optional)
    '16/18O_ratio': ([200, 1000], r'$^{16}$O/$^{18}$O', 'uniform'),  # 16O/18O isotope ratio (optional)
    # 'H/D_ratio': ([200, 1000], r'H/D', 'uniform'),  # H/D isotope ratio (optional)
    
    # Optional: Override specific species with free chemistry (if needed) if in fastchem calculation, otherwise add a new species here
    'log_Sc':([-12,-1], r'log Sc', 'uniform'),  # Add Sc abundance as free chemistry
}



# ========== Physical model keyword-arguments ==========
TP_kwargs = dict[str, tuple[float, float] | int | str](
    # ...

    # ----- pressure grid, must be given -----
    log_P_range = (-5., 2.),  # pressure range
    n_atm_layers = 70,  # number of atmospheric layers (interpolate to)
    # pressure = np.logspace(-5., 2., 70),  # or personalized (will be set automatically from atmosphere object)

    # ----- temperature profile -----
    # Option 1: Interpolation mode (uses T_0, T_1, T_2, ... from free_params)
    TP_mode = 'interpolation',  # default
    n_knots = 5,  # number of temperature knots
    interp_mode = 'cubic',  # 'cubic', 'quadratic', or 'linear'
    
    # Option 2: Gradient mode (uses dlnT_dlnP_0, dlnT_dlnP_1, ... from free_params)
    # TP_mode = 'gradient',
    # n_knots = 5,  # number of gradient knots
    # interp_mode = 'linear',  # interpolation mode for gradients
    # Also need T_phot or T_0 in constant_params for base temperature
    
    # temperature = 1500.,  # Constant temperature profile (overrides above)
)

chemistry_kwargs = dict(
    # ----- chemistry mode -----
    chem_mode = 'fastchem_live',  # 'free', 'equilibrium', 'fastchem_grid', 'fastchem_live'

    # ----- fastchem input files -----
    abundance_file = '/home/chenyangji/ESO/analysis/retrieval/retrieval_base/data/asplund_2020.dat',
    # gas_data_file = '/home/chenyangji/ESO/analysis/retrieval/retrieval_base/data/logK_simplified.txt',
    gas_data_file = '/home/chenyangji/ESO/analysis/retrieval/retrieval_base/data/_logK.dat',  # full version
    # cond_data_file = '/home/chenyangji/ESO/analysis/retrieval/retrieval_base/data/logK_condensates_simplified.txt',  # do not need to provide at this time
    # use_eq_cond = True,  # use equilibrium condensation by default
    # use_rainout_cond = False,
    min_temperature = 500.0,  # minimum temperature for FastChem convergence by default
    
    # ----- optional arguments -----
    # species_info_path: Path to custom species_info.csv file.
    #   - If None or not specified: uses default path (SRC_DIR / "atmosphere" / "species_info.csv")
    #   - If specified: uses the provided path (must exist, otherwise falls back to default)
    species_info_path = '/home/chenyangji/ESO/analysis/retrieval/retrieval_base/src/atmosphere/species_info_Sam.csv',  # Example: species_info_path = "/path/to/custom/species_info.csv"
    
    # line_species: List of species names to include in equilibrium chemistry.
    #   - If None or not specified: automatically extracts all available species from pRT eq-chem table
    #   - If specified: can be either:
    #     a) species_info names (e.g., ['H2O', '12CO', 'CH4']) - recommended, matches free chemistry config
    #     b) pRT names (e.g., ['1H2-16O', '12C-16O', '12C-1H4'])
    #   - The code will automatically convert species_info names to pRT names if needed
    #   - To match free chemistry config (config_starB_7mol_free.py), use species_info names:
    line_species = ['Na', 'Ca', 'HF', '12CO', '13CO', 'H2O', 'OH', 'CN', 'TiO', 'Sc', 'Fe', 'Ti', 'C18O', 'H2(18)O'],  # modified version under higher temperature, without He and H2
    # species I want to save into the VMR file -- name in species_info.csv
    save_species = ['Na', 'Ca', 'HF', '12CO', '13CO', 'H2O', 'OH', 'CN', 'TiO', 'Sc', 'e-', 'H', 'H1-', 'Fe', 'Ti', 'He', 'H2', 'C18O', 'H2(18)O']  # should include H2 and He
    # save_species = ['Na', 'Ca', 'HF', '12CO', '13CO', 'H2O', 'OH', 'CN', 'TiO', 'Sc', 'e-', 'H', 'H1-', 'Fe', 'Ti']
    
    # LineOpacity: Custom line opacity objects (list of opacity objects).
    #   - If None or not specified: no custom line opacities are used
    #   - If specified: should be a list of opacity objects with 'line_species' attribute
    # LineOpacity = None,  # Example: LineOpacity = [custom_opacity_object1, custom_opacity_object2]
)

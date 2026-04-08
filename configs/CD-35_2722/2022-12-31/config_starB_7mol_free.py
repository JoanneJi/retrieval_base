"""
Example configuration file for the simple retrieval framework.

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
    # ...

    # ----- full format of the input -----
    # 'T0' : {'bounds': [0, 5000], 'label': r'$T_0$', 'type': 'uniform'},
    # ... but make sure all the parameters are using the same format

    # ----- radial velocity & surface gravity -----
    'rv': ([0, 40], r'$v_{\rm rad}$', 'uniform'),  # radial velocity [km/s]
    'log_g': ([3, 5.5], r'log $g$', 'uniform'),  # log10 of surface gravity [cm/s^2]

    # ----- temperature profile -----
    # case 1: 5 temperature knots -> T0, T1, T2, T3, T4
    'T_0' : ([1000, 10000], r'$T_0$', 'uniform'), # bottom of the atmosphere (usually hotter)
    'T_1' : ([1000, 5000], r'$T_1$', 'uniform'),
    'T_2' : ([300, 4000], r'$T_2$', 'uniform'),
    'T_3' : ([300, 4000], r'$T_3$', 'uniform'),
    'T_4' : ([300, 4000], r'$T_4$', 'uniform'), # top of atmosphere (usually cooler)
    # # case 2: 5 temperature gradients -> dlnT_dlnP_{i} (for TP_mode='gradient')
    # # Note: Also need T_phot or T_0 in constant_params as base temperature
    # 'dlnT_dlnP_0' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_0$', 'uniform'), # bottom of the atmosphere (usually hotter)
    # 'dlnT_dlnP_1' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_1$', 'uniform'),
    # 'dlnT_dlnP_2' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_2$', 'uniform'),
    # 'dlnT_dlnP_3' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_3$', 'uniform'),
    # 'dlnT_dlnP_4' : ([0,0.1], r'$\frac{d\ln T}{d\ln P}_4$', 'uniform'), # top of atmosphere (usually cooler)

    # ----- stellar/planetary parameters -----
    'vsini': ([0, 40], r'$v\sin i$', 'uniform'),  # projected rotational velocity [km/s] - used for rotational broadening
    
    # ----- chemistry parameters -----
    'log_H2O':([-12,-1], r'log H$_2$O', 'uniform'), # if equilibrium chemistry, define VMRs
    'log_12CO':([-12,-1], r'log $^{12}$CO', 'uniform'),
    'log_13CO':([-12,-1], r'log $^{13}$CO', 'uniform'),
    'log_CH4':([-12,-1], r'log CH$_4$', 'uniform'),
    'log_H2S':([-12,-1], r'log H$_2$S', 'uniform'),
    'log_NH3':([-12,-1], r'log NH$_3$', 'uniform'),
    'log_HF':([-12,-1], r'log HF', 'uniform'),
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

chemistry_kwargs = dict[str, str](
    # ----- chemistry mode -----
    chem_mode = 'free',  # 'free' or 'equilibrium'
    
    # ----- optional arguments -----
    # species_info_path: Path to custom species_info.csv file.
    #   - If None or not specified: uses default path (SRC_DIR / "atmosphere" / "species_info.csv")
    #   - If specified: uses the provided path (must exist, otherwise falls back to default)
    species_info_path = '/home/chenyangji/ESO/analysis/retrieval/retrieval_base/src/atmosphere/species_info_Sam.csv',  # Example: species_info_path = "/path/to/custom/species_info.csv"
    
    # LineOpacity: Custom line opacity objects (list of opacity objects).
    #   - If None or not specified: no custom line opacities are used
    #   - If specified: should be a list of opacity objects with 'line_species' attribute
    # LineOpacity = None,  # Example: LineOpacity = [custom_opacity_object1, custom_opacity_object2]
)




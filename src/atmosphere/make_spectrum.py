from typing import Any


import numpy as np
import warnings

from astropy.coordinates import SkyCoord
from astropy import constants as const

from PyAstronomy.pyasl import helcorr
from PyAstronomy.pyasl import fastRotBroad
from atmosphere.chemistry import get_class as get_chemistry_class
from atmosphere.tp import get_class as get_tp_class
from utils.spectral import convolve_to_resolution, instr_broadening


class pRT_spectrum:
    """
    Forward model: physical parameters → synthetic spectrum.

    This class:
    - reads parameters from Parameters.params
    - builds P-T profile using TP_profile classes
    - sets chemistry using Chemistry classes
    - runs petitRADTRANS
    - applies Doppler shift, rotation, instrumental broadening
    - interpolates to data wavelength grid
    """

    def __init__(self, parameters, target, atmosphere, spectral_resolution=100_000, 
                 lbl_opacity_sampling=3, contribution=False, debug=False):
        self.parameters = parameters
        self.params = parameters.params  # shorthand
        self.target = target
        self.atmosphere = atmosphere

        self.data_wave = target.wl.flatten()
        self.coords = SkyCoord(ra=target.ra, dec=target.dec, frame="icrs")

        self.spectral_resolution = spectral_resolution
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.contribution = contribution
        self.debug = debug

        # Atmosphere setup
        # Radtrans stores pressure in CGS units (dyn/cm²), but we need bar for TP_profile and Chemistry
        # Convert from CGS to bar: 1 bar = 1e6 dyn/cm²
        self.pressure = atmosphere.pressures * 1e-6  # CGS to bar
        self.n_atm_layers = len(self.pressure)
        
        if self.debug:
            print(f"[DEBUG make_spectrum] Pressure from Radtrans (CGS): {atmosphere.pressures.min():.2e} - {atmosphere.pressures.max():.2e} dyn/cm²")
            print(f"[DEBUG make_spectrum] Pressure converted to bar: {self.pressure.min():.2e} - {self.pressure.max():.2e} bar")

        # ===== Get TP_kwargs from parameters object (loaded from config file) =====
        TP_kwargs = parameters.TP_kwargs.copy() if parameters.TP_kwargs else {}
        
        # Set default TP_kwargs if empty
        if not TP_kwargs:
            TP_kwargs = {
                'TP_mode': 'interpolation',
                'n_knots': 5,
                'interp_mode': 'cubic',
            }
            if debug:
                print("[atmosphere/make_spectrum.py/pRT_spectrum] No TP_kwargs provided. Using default TP_kwargs: 'TP_mode': 'interpolation', 'n_knots': 5, 'interp_mode': 'cubic'.")
        else:
            if debug:
                print("[atmosphere/make_spectrum.py/pRT_spectrum] TP_kwargs provided: 'TP_mode': {}, 'n_knots': {}, 'interp_mode': {}.".format(TP_kwargs.get('TP_mode'), TP_kwargs.get('n_knots'), TP_kwargs.get('interp_mode')))
        
        # Pass pressure to TP_profile (same as Chemistry class)
        self.tp_profile = get_tp_class(pressure=self.pressure, **TP_kwargs)

        # ===== Get chemistry_kwargs from parameters object (loaded from config file) =====
        chemistry_kwargs = parameters.chemistry_kwargs.copy() if parameters.chemistry_kwargs else {}
        
        # Set default chemistry_kwargs if empty
        if not chemistry_kwargs:
            chemistry_kwargs = {
                'chem_mode': 'free',
            }
            if debug:
                print("[atmosphere/make_spectrum.py/pRT_spectrum] No chemistry_kwargs provided. Using default chemistry_kwargs: 'chem_mode': 'free'.")
        else:
            if debug:
                print("[atmosphere/make_spectrum.py/pRT_spectrum] chemistry_kwargs provided: 'chem_mode': {}.".format(chemistry_kwargs['chem_mode']))
        
        # Get line species from parameters (for Chemistry initialization)
        # Get species_info_path from chemistry_kwargs (will be None if not specified, which uses default)
        from atmosphere import get_species_from_params
        species_info_path = chemistry_kwargs.get('species_info_path')
        line_species = get_species_from_params(param_dict=self.params, species_info_path=species_info_path)
        
        self.chemistry = get_chemistry_class(
            pressure=self.pressure,
            line_species=line_species,
            **chemistry_kwargs
        )

        # ===== Build atmosphere state using TP_profile and Chemistry classes =====
        # Generate temperature profile using TP_profile -- result will be -np.inf if failed
        result = self.tp_profile(self.params)
        self.temperature = self.tp_profile.temperature
        # Check if temperature generation failed
        if result == -np.inf or self.temperature is None or (isinstance(self.temperature, np.ndarray) and (self.temperature < 0).any()):
            raise ValueError("[atmosphere/make_spectrum.py/pRT_spectrum] Failed to create temperature profile")
        
        self.gravity = 10 ** self.params["log_g"]

        # Generate chemistry using Chemistry class -- result will be -np.inf if failed
        self.mass_fractions = self.chemistry(self.params, self.temperature)
        
        # Debug: Print mass fractions and other diagnostics
        if self.debug:
            print("\n[DEBUG make_spectrum] ===== Chemistry Results =====")
            print(f"[DEBUG make_spectrum] Mass fractions type: {type(self.mass_fractions)}")
            if isinstance(self.mass_fractions, dict):
                print(f"[DEBUG make_spectrum] Mass fractions keys: {list(self.mass_fractions.keys())}")
                for key, value in self.mass_fractions.items():
                    if isinstance(value, np.ndarray):
                        print(f"[DEBUG make_spectrum]   {key}: shape={value.shape}, min={value.min():.2e}, max={value.max():.2e}, mean={value.mean():.2e}")
                    else:
                        print(f"[DEBUG make_spectrum]   {key}: {value}")
            print("[DEBUG make_spectrum] ================================\n")
        
        if self.mass_fractions == -np.inf or not isinstance(self.mass_fractions, dict):
            raise ValueError("[atmosphere/make_spectrum.py/pRT_spectrum] Failed to create chemistry model with mass fractions: {}".format(self.mass_fractions))
        
        # Extract diagnostics from chemistry object
        self.CO = self.chemistry.CO
        self.FeH = self.chemistry.FeH
        self.MMW = self.mass_fractions["MMW"]

    # ========== Spectrum synthesis ==========
    def make_spectrum(self):
        # --- pRT forward model, which returns a pRT atmosphere object ---
        wl, flux, _ = self.atmosphere.calculate_flux(  # in cm
            temperatures=self.temperature,
            mass_fractions=self.mass_fractions,
            reference_gravity=self.gravity,
            mean_molar_masses=self.MMW,
            return_contribution=self.contribution,
            frequencies_to_wavelengths=True,
        )

        wl *= 1e7  # cm → nm
        flux /= np.nanmedian(flux)

        # --- Barycentric + RV shift ---
        # Extract RA and Dec values (type checker safety)
        ra_value = self.coords.ra.value if self.coords.ra is not None else 0.0
        dec_value = self.coords.dec.value if self.coords.dec is not None else 0.0
        v_bary, _ = helcorr(
            obs_long=-70.40,
            obs_lat=-24.62,
            obs_alt=2635,
            ra2000=ra_value,
            dec2000=dec_value,
            jd=self.target.JD,
        )

        wl_shifted = wl * (1 + (self.params["rv"] - v_bary) / getattr(const, 'c').to("km/s").value)

        # --- Regrid, evenly speced ---
        waves_even = np.linspace(wl.min(), wl.max(), wl.size)
        flux = np.interp(waves_even, wl_shifted, flux)

        # --- Rotation ---
        flux = fastRotBroad(waves_even, flux, 0.5, self.params["vsini"])

        # --- Instrumental broadening (double convolution) ---
        # Step 1: Convolve to spectral_resolution (e.g., 100,000)
        # This assumes input is infinite resolution
        flux = convolve_to_resolution(
            waves_even, flux, out_res=self.spectral_resolution
        )
        
        # Step 2: Additional broadening from lbl_opacity_sampling resolution to final resolution
        # The lbl opacity sampling creates a spectrum at resolution = 1e6 / lbl_opacity_sampling
        # We need to broaden from intermediate resolution to final resolution
        resolution = int(1e6 / self.lbl_opacity_sampling)
        # Calculate intermediate resolution based on lbl_opacity_sampling
        # For lbl_opacity_sampling=3, this gives ~500,000 (conservative estimate)
        intermediate_resolution = int(1e6 / max(1, self.lbl_opacity_sampling - 1))
        flux = instr_broadening(
            waves_even, flux, 
            out_res=resolution, 
            in_res=intermediate_resolution
        )

        # --- Interpolate to data grid ---
        model_flux = np.interp(self.data_wave, waves_even, flux)

        return model_flux

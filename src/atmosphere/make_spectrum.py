from typing import Any

import numpy as np
import warnings
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy import constants as const

from PyAstronomy.pyasl import helcorr
from PyAstronomy.pyasl import fastRotBroad
from atmosphere.chemistry import get_class as get_chemistry_class
from atmosphere.tp import get_class as get_tp_class
from atmosphere.cloud import get_class as get_cloud_class
from utils.spectral import convolve_to_resolution, instr_broadening
from utils.normalization import (
    simplistic_normalization,
    low_resolution_normalization,
    median_highpass_normalization,
    gaussian_lfp,
    savgol_lfp
)


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
                 lbl_opacity_sampling=3, normalize=True, normalize_method='simplistic_normalization',
                 contribution=False, debug=False):
        self.parameters = parameters
        self.params = parameters.params  # shorthand
        self.target = target
        self.atmosphere = atmosphere

        # Handle both chips_mode and single spectrum mode
        if target.chips_mode:
            # Multi-chip mode: data_wave is a list of arrays
            self.data_wave = target.wl  # list of arrays
            self.chips_mode = True
            self.n_chips = target.n_chips
        else:
            # Single spectrum mode (backward compatible)
            self.data_wave = target.wl.flatten() if hasattr(target.wl, 'flatten') else target.wl
            self.chips_mode = False
            self.n_chips = 1
        
        self.coords = SkyCoord(ra=target.ra, dec=target.dec, frame="icrs")

        self.spectral_resolution = spectral_resolution
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.contribution = contribution
        self.debug = debug
        # Optional: number of chips per order (e.g. 3 detectors per order)
        # Only relevant in chips_mode
        self.chips_per_order = getattr(target, 'chips_per_order', None)

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
        line_species_from_params = get_species_from_params(param_dict=self.params, species_info_path=species_info_path)  # from 'log_xxx'
        
        chem_mode = chemistry_kwargs.get('chem_mode', 'free')
        
        # Helper function to convert line_species to pRT names
        def convert_line_species_to_pRT(line_species_list, species_info_path, debug=False):
            """Convert line_species from species_info names to pRT names if needed."""
            if not line_species_list or not isinstance(line_species_list, list):
                return line_species_list
            
            # Quick check: if all are already pRT names, use directly
            all_pRT_names = all('-' in str(s) and any(c.isdigit() for c in str(s)) for s in line_species_list)
            if all_pRT_names:
                if debug:
                    print(f"[atmosphere/make_spectrum.py/pRT_spectrum] Using pre-converted pRT names: {line_species_list}")
                return line_species_list
            
            # Need to convert species_info names (e.g., 'H2O', '12CO') to pRT names (e.g., '1H2-16O', '12C-16O')
            import pathlib
            import pandas as pd
            from core.paths import SRC_DIR
            
            if species_info_path is None:
                species_info_path = SRC_DIR / "atmosphere" / "species_info.csv"
            else:
                species_info_path = pathlib.Path(species_info_path)
            
            if not species_info_path.exists():
                # Try default path
                species_info_path = SRC_DIR / "atmosphere" / "species_info.csv"
            
            if species_info_path.exists():
                species_info = pd.read_csv(species_info_path, index_col=0)
                line_species_pRT = []
                for species in line_species_list:
                    # Check if it's already a pRT name (contains '-' and numbers, e.g., '12C-16O')
                    # or if it's a species_info name (e.g., 'H2O', '12CO')
                    if '-' in str(species) and any(c.isdigit() for c in str(species)):
                        # Already a pRT name, use as is
                        line_species_pRT.append(species)
                    elif species in species_info.index:
                        # It's a species_info name, convert to pRT name
                        pRT_name = species_info.loc[species, 'pRT_name']
                        if pd.notna(pRT_name):
                            line_species_pRT.append(str(pRT_name))
                        else:
                            warnings.warn(f"Species '{species}' found in species_info but pRT_name is missing. Using as is.")
                            line_species_pRT.append(species)
                    else:
                        # Unknown format, use as is (might be a pRT name we don't recognize)
                        line_species_pRT.append(species)
                if debug:
                    print(f"[atmosphere/make_spectrum.py/pRT_spectrum] Converted line_species from chemistry_kwargs to pRT names: {line_species_pRT}")
                return line_species_pRT
            else:
                if debug:
                    print(f"[atmosphere/make_spectrum.py/pRT_spectrum] species_info.csv not found, using line_species as provided: {line_species_list}")
                return line_species_list
        
        # For fastchem_live and fastchem_grid: always use line_species from chemistry_kwargs if provided
        # (ignore line_species extracted from log_* params)
        if chem_mode in ['fastchem_live', 'fastchem_grid']:
            if 'line_species' in chemistry_kwargs:
                line_species = chemistry_kwargs['line_species']
                line_species = convert_line_species_to_pRT(line_species, species_info_path, debug)
                if debug:
                    print(f"[atmosphere/make_spectrum.py/pRT_spectrum] Using line_species from chemistry_kwargs for {chem_mode} mode: {line_species}")
            else:
                # Fallback to line_species from log_* params if no line_species in chemistry_kwargs
                line_species = line_species_from_params
                if debug:
                    print(f"[atmosphere/make_spectrum.py/pRT_spectrum] No line_species in chemistry_kwargs for {chem_mode} mode, using from log_* params: {line_species}")
        # For equilibrium chemistry, if line_species is empty (no log_* params),
        # try to get from chemistry_kwargs
        # Note: If still empty, EquilibriumChemistry.get_VMRs() will automatically
        # extract all available species from pRT interpolation table
        elif chem_mode == 'equilibrium':
            if len(line_species_from_params) == 0:
                # Check if line_species is explicitly provided in chemistry_kwargs
                if 'line_species' in chemistry_kwargs:
                    line_species = chemistry_kwargs['line_species']
                    line_species = convert_line_species_to_pRT(line_species, species_info_path, debug)
                else:
                    line_species = []
                    if debug:
                        print(f"[atmosphere/make_spectrum.py/pRT_spectrum] No line_species found. "
                              f"EquilibriumChemistry will automatically extract available species from pRT table.")
            else:
                line_species = line_species_from_params
        else:
            # For 'free' chemistry mode, use line_species from log_* params
            line_species = line_species_from_params
        
        # Remove line_species from chemistry_kwargs to avoid duplicate argument error
        # (line_species is already passed as a separate argument)
        chemistry_kwargs_for_init = chemistry_kwargs.copy()
        chemistry_kwargs_for_init.pop('line_species', None)
        
        self.chemistry = get_chemistry_class(
            pressure=self.pressure,
            line_species=line_species,
            **chemistry_kwargs_for_init
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
        
        # ===== Get cloud_kwargs from parameters object (loaded from config file) =====
        cloud_kwargs = parameters.cloud_kwargs.copy() if hasattr(parameters, 'cloud_kwargs') and parameters.cloud_kwargs else {}
        
        # Set default cloud_kwargs if empty (no cloud)
        if not cloud_kwargs:
            cloud_kwargs = {
                'cloud_mode': None,  # No cloud by default
            }
            if debug:
                print("[atmosphere/make_spectrum.py/pRT_spectrum] No cloud_kwargs provided. Using default: no cloud.")
        else:
            if debug:
                print("[atmosphere/make_spectrum.py/pRT_spectrum] cloud_kwargs provided: 'cloud_mode': {}.".format(cloud_kwargs.get('cloud_mode')))
        
        # Initialize cloud object
        self.cloud = get_cloud_class(pressure=self.pressure, **cloud_kwargs)

    # ========== Spectrum synthesis ==========
    def make_spectrum(self):
        if self.chips_mode:
            # Multi-chip mode: process each chip separately
            model_flux = self._make_spectrum_chips()
        else:
            # Single spectrum mode (original behavior, backward compatible)
            model_flux = self._make_spectrum_single()

        # Optional diagnostic plot: compare model flux and data flux on the same wavelength grid
        if self.debug:
            data_wl = self.target.wl_flat
            data_fl = self.target.fl_flat

            plt.figure(figsize=(10, 4))
            plt.plot(data_wl, data_fl, label="data", alpha=0.7, color='black')
            plt.plot(data_wl, model_flux, label="model", alpha=0.7, color='red')
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("Flux")
            plt.legend()
            plt.tight_layout()
            plt.savefig("created_atmosphere_object.pdf")

        return model_flux
    
    def _make_spectrum_single(self):
        """Original single spectrum processing (backward compatible)."""
        # --- Update cloud model with current parameters ---
        # Calculate mean wavelength for cloud opacity calculation (optional)
        mean_wave_micron = np.nanmean(self.data_wave) * 1e-3 if hasattr(self.data_wave, '__len__') else None
        self.cloud(self.params, mean_wave_micron=mean_wave_micron)
        
        # Get cloud opacity functions (if available)
        cloud_abs_opacity = getattr(self.cloud, 'abs_opacity', None)
        cloud_scat_opacity = getattr(self.cloud, 'scat_opacity', None)
        
        # remove MMW from mass_fractions
        mass_fractions_no_mmw = self.mass_fractions.copy()
        mass_fractions_no_mmw.pop('MMW')
        # --- pRT forward model, which returns a pRT atmosphere object ---
        wl, flux, _ = self.atmosphere.calculate_flux(  # in cm
            temperatures=self.temperature,
            mass_fractions=mass_fractions_no_mmw,
            reference_gravity=self.gravity,
            mean_molar_masses=self.MMW,
            additional_absorption_opacities_function=cloud_abs_opacity,
            additional_scattering_opacities_function=cloud_scat_opacity,
            return_contribution=self.contribution,
            frequencies_to_wavelengths=True,
        )

        # if debug, print continuum contribution from H- and CIA
        if self.debug:
            hminus = self.atmosphere._compute_h_minus_opacities(
                mass_fractions=mass_fractions_no_mmw,
                pressures=self.pressure,
                temperatures=self.temperature,
                frequencies=self.atmosphere._frequencies,
                frequency_bins_edges=self.atmosphere._frequency_bins_edges,
                mean_molar_masses=self.MMW,
            )
            print(f"[DEBUG make_spectrum] Continuum contribution from H- [cm^2/g]: mean={np.mean(hminus)}, min={hminus.min():.2e}, max={hminus.max():.2e}")
            cia_opacities = self.atmosphere._compute_cia_opacities(
                cia_dicts=self.atmosphere._cias_loaded_opacities,
                mass_fractions=mass_fractions_no_mmw,
                pressures=self.pressure,
                temperatures=self.temperature,
                frequencies=self.atmosphere._frequencies,
                mean_molar_masses=self.MMW,
            )
            print(f"[DEBUG make_spectrum] Continuum contribution from CIA [cm^2/g]: mean={np.mean(cia_opacities)}, min={cia_opacities.min():.2e}, max={cia_opacities.max():.2e}")

        wl *= 1e7  # cm → nm
        
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

        # --- Normalize on data wavelength grid (after all broadening and interpolation) ---
        if self.normalize:
            if self.normalize_method == 'simplistic_normalization':
                model_flux = simplistic_normalization(model_flux, err=None)
            elif self.normalize_method == 'low-resolution':
                # low_resolution_normalization returns (flux_norm, continuum) when err is None
                model_flux, _ = low_resolution_normalization(
                    self.data_wave, model_flux, err=None, out_res=150
                )
            elif self.normalize_method == 'median_highpass':
                model_flux = median_highpass_normalization(self.data_wave, model_flux, err=None, window=100)
            elif self.normalize_method == 'gaussian_lfp':
                model_flux = gaussian_lfp(self.data_wave, model_flux, err=None, sigma_px=100)
            elif self.normalize_method == 'savgol_lfp':
                model_flux = savgol_lfp(self.data_wave, model_flux, err=None, window_length=1301, polyorder=2)
            else:
                raise ValueError(
                    f"Unknown normalize_method: {self.normalize_method}. "
                    "Must be one of: 'simplistic_normalization', 'low-resolution', 'median_highpass'"
                )

        return model_flux
    
    def _make_spectrum_chips(self):
        """Multi-chip spectrum processing: use single Radtrans object, extract each chip."""
        # Optimized mode: single Radtrans object covering all chips
        # Calculate full spectrum once, then extract each chip's range
        
        # Get wave_ranges_chips from atmosphere object (stored during setup)
        wave_ranges_chips = getattr(self.atmosphere, 'wave_ranges_chips', None)
        wl_pad = getattr(self.atmosphere, 'wl_pad', 7)
        
        if wave_ranges_chips is None:
            # Fallback: estimate from data_wave
            wave_ranges_chips = np.array([[w.min(), w.max()] for w in self.data_wave])
        
        # --- Barycentric correction (same for all chips) ---
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
        
        # --- Update cloud model with current parameters ---
        # Calculate mean wavelength for cloud opacity calculation (optional)
        # Use mean of all chips' wavelengths
        all_wavelengths = np.concatenate(self.data_wave) if isinstance(self.data_wave, list) else self.data_wave
        mean_wave_micron = np.nanmean(all_wavelengths) * 1e-3 if len(all_wavelengths) > 0 else None
        self.cloud(self.params, mean_wave_micron=mean_wave_micron)
        
        # Get cloud opacity functions (if available)
        cloud_abs_opacity = getattr(self.cloud, 'abs_opacity', None)
        cloud_scat_opacity = getattr(self.cloud, 'scat_opacity', None)
        
        # remove MMW from mass_fractions
        mass_fractions_no_mmw = self.mass_fractions.copy()
        mass_fractions_no_mmw.pop('MMW')
        # --- Calculate full spectrum once (covering all chips) ---
        wl_full, flux_full, _ = self.atmosphere.calculate_flux(  # in cm
            temperatures=self.temperature,
            mass_fractions=mass_fractions_no_mmw,
            reference_gravity=self.gravity,
            mean_molar_masses=self.MMW,
            additional_absorption_opacities_function=cloud_abs_opacity,
            additional_scattering_opacities_function=cloud_scat_opacity,
            return_contribution=self.contribution,
            frequencies_to_wavelengths=True,
        )

        # if debug, print continuum contribution from H- and CIA
        if self.debug:
            hminus = self.atmosphere._compute_h_minus_opacities(
                mass_fractions=mass_fractions_no_mmw,
                pressures=self.pressure,
                temperatures=self.temperature,
                frequencies=self.atmosphere._frequencies,
                frequency_bins_edges=self.atmosphere._frequency_bins_edges,
                mean_molar_masses=self.MMW,
            )
            print(f"[DEBUG make_spectrum] Continuum contribution from H- [cm^2/g]: mean={np.mean(hminus)}, min={hminus.min():.2e}, max={hminus.max():.2e}")
            cia_opacities = self.atmosphere._compute_cia_opacities(
                cia_dicts=self.atmosphere._cias_loaded_opacities,
                mass_fractions=mass_fractions_no_mmw,
                pressures=self.pressure,
                temperatures=self.temperature,
                frequencies=self.atmosphere._frequencies,
                mean_molar_masses=self.MMW,
            )
            print(f"[DEBUG make_spectrum] Continuum contribution from CIA [cm^2/g]: mean={np.mean(cia_opacities)}, min={cia_opacities.min():.2e}, max={cia_opacities.max():.2e}")
        
        wl_full *= 1e7  # cm → nm
        
        # --- Process each chip: first extract chip segments, then normalize, then broaden ---
        n_chips = len(self.data_wave)
        chip_wl = []
        chip_flux = []
        
        # 1) Extract chip segments from full spectrum (no normalization yet)
        for i in range(n_chips):
            # Get wavelength range for this chip (with padding for extraction)
            wlmin_chip = wave_ranges_chips[i, 0] - wl_pad
            wlmax_chip = wave_ranges_chips[i, 1] + wl_pad
            
            # Extract the relevant portion from full spectrum (only chip region, skip gaps)
            mask_chip = (wl_full >= wlmin_chip) & (wl_full <= wlmax_chip)
            wl_chip = wl_full[mask_chip]
            flux_chip = flux_full[mask_chip]
            
            chip_wl.append(wl_chip)
            chip_flux.append(flux_chip)

        # 2) Apply RV shift, rotation, instrumental broadening and regrid per chip
        model_flux_chips = []
        for i, data_wave_i in enumerate(self.data_wave):
            wl_chip = chip_wl[i]
            flux_chip = chip_flux[i]

            # --- Barycentric + RV shift (apply to this chip) ---
            wl_shifted_chip = wl_chip * (1 + (self.params["rv"] - v_bary) / getattr(const, 'c').to("km/s").value)
            
            # --- Regrid, evenly spaced (this chip only) ---
            waves_even_chip = np.linspace(wl_chip.min(), wl_chip.max(), wl_chip.size)
            flux_chip = np.interp(waves_even_chip, wl_shifted_chip, flux_chip)
            
            # --- Rotation (this chip only) ---
            flux_chip = fastRotBroad(waves_even_chip, flux_chip, 0.5, self.params["vsini"])
            
            # --- Instrumental broadening (double convolution, this chip only) ---
            # Step 1: Convolve to spectral_resolution
            flux_chip = convolve_to_resolution(
                waves_even_chip, flux_chip, out_res=self.spectral_resolution
            )
            
            # Step 2: Additional broadening
            resolution = int(1e6 / self.lbl_opacity_sampling)
            intermediate_resolution = int(1e6 / max(1, self.lbl_opacity_sampling - 1))
            flux_chip = instr_broadening(
                waves_even_chip, flux_chip,
                out_res=resolution,
                in_res=intermediate_resolution
            )
            
            # --- Interpolate to data grid for this chip ---
            model_flux_chip = np.interp(data_wave_i, waves_even_chip, flux_chip)
            model_flux_chips.append(model_flux_chip)
        
        # Concatenate all chips into a single array (still pre-normalization)
        model_flux = np.concatenate(model_flux_chips)

        # 3) Normalize on data wavelength grid (after all broadening and interpolation)
        if self.normalize:
            if self.normalize_method == 'low-resolution' and self.chips_per_order is not None:
                # Order-level low-resolution normalization on data grid
                cpo = int(self.chips_per_order)

                # For diagnostic plotting across all orders
                plot_wl_all = []
                plot_flux_all = []
                plot_cont_all = []

                for start_idx in range(0, n_chips, cpo):
                    end_idx = min(start_idx + cpo, n_chips)
                    wl_all = np.concatenate(self.data_wave[start_idx:end_idx])
                    f_all = np.concatenate(model_flux_chips[start_idx:end_idx])

                    # low_resolution_normalization returns (flux_norm, continuum) when err is None
                    f_all_norm, continuum = low_resolution_normalization(
                        wl_all, f_all, err=None, out_res=150
                    )

                    # Collect for plotting (all orders together)
                    if self.debug:
                        plot_wl_all.append(wl_all)
                        plot_flux_all.append(f_all)
                        plot_cont_all.append(continuum)

                    # Split normalized flux back to each chip in this order
                    offset = 0
                    for idx in range(start_idx, end_idx):
                        n_pix = len(self.data_wave[idx])
                        model_flux_chips[idx] = f_all_norm[offset:offset + n_pix]
                        offset += n_pix

                # Plot flux and continuum for all orders (only when debugging)
                if self.debug and len(plot_wl_all) > 0:
                    wl_plot = np.concatenate(plot_wl_all)
                    flux_plot = np.concatenate(plot_flux_all)
                    cont_plot = np.concatenate(plot_cont_all)

                    plt.figure(figsize=(10, 4))
                    plt.plot(wl_plot, flux_plot, label="flux")
                    plt.plot(wl_plot, cont_plot, label="continuum")
                    plt.plot(wl_planck_nm, planck_flux * 20, color="violet", alpha=0.8, linewidth=3, label=f"Planck function: Teff = {teff} K")
                    plt.xlabel("Wavelength [nm]")
                    plt.ylabel("Flux")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig("flux_and_continuum.pdf")
                    import pdb; pdb.set_trace()

            else:
                # Per-chip normalization on data grid
                if self.normalize_method == 'low-resolution' and self.chips_per_order is None:
                    warnings.warn(
                        "[pRT_spectrum._make_spectrum_chips] chips_per_order is None; "
                        "applying low-resolution normalization per chip instead of per order."
                        "Check whether the data is also normalized per chip."
                    )
                for i in range(n_chips):
                    wl_chip_data = self.data_wave[i]
                    flux_chip_data = model_flux_chips[i]

                    if self.normalize_method == 'simplistic_normalization':
                        flux_chip_data = simplistic_normalization(flux_chip_data, err=None)
                    elif self.normalize_method == 'low-resolution':
                        # low_resolution_normalization returns (flux_norm, continuum) when err is None
                        flux_chip_data, _ = low_resolution_normalization(
                            wl_chip_data, flux_chip_data, err=None, out_res=150
                        )
                    elif self.normalize_method == 'median_highpass':
                        flux_chip_data = median_highpass_normalization(
                            wl_chip_data, flux_chip_data, err=None, window=100
                        )
                    elif self.normalize_method == 'gaussian_lfp':
                        flux_chip_data = gaussian_lfp(
                            wl_chip_data, flux_chip_data, err=None, sigma_px=100
                        )
                    elif self.normalize_method == 'savgol_lfp':
                        flux_chip_data = savgol_lfp(
                            wl_chip_data, flux_chip_data, err=None, window_length=1301, polyorder=2
                        )
                    else:
                        raise ValueError(
                            f"Unknown normalize_method: {self.normalize_method}. "
                            "Must be one of: 'simplistic_normalization', 'low-resolution', 'median_highpass'"
                        )

                    model_flux_chips[i] = flux_chip_data

            # Re-concatenate normalized chips
            model_flux = np.concatenate(model_flux_chips)
        
        return model_flux




# backup -- Planck function
import numpy as np


def planck_lambda(wavelength_nm, temperature_K):
    """Planck function B_λ(λ, T) in SI units.

    Parameters
    ----------
    wavelength_nm : array_like
        Wavelength in meters.
    temperature_K : float or array_like
        Temperature in Kelvin.

    Returns
    -------
    ndarray
        Spectral radiance B_λ in W m^-3 sr^-1.
    """
    h = 6.62607015e-34  # Planck constant [J s]
    c = 2.99792458e8    # Speed of light [m/s]
    k_B = 1.380649e-23  # Boltzmann constant [J/K]

    wavelength_m = wavelength_nm * 1e-9
    
    lam = np.asarray(wavelength_m, dtype=float)
    T = np.asarray(temperature_K, dtype=float)

    # Avoid division by zero
    lam = np.where(lam == 0, np.finfo(float).tiny, lam)

    exponent = (h * c) / (lam * k_B * T)
    # Use np.expm1 for numerical stability when exponent is small
    denom = np.expm1(exponent)

    prefactor = 2.0 * h * c**2 / lam**5
    B_lambda = prefactor / denom
    return B_lambda

wl_planck_nm = np.linspace(2050, 2500, 1000)
teff = 3600
planck_flux = planck_lambda(wl_planck_nm, teff)
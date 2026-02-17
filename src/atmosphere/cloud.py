"""
Select different cloud models for creating atmosphere model. Cloud models include:
    - no cloud
    - gray cloud

Authors
-------
Original version: Sam de Regt
Modified by: Chenyang Ji (2026-02-05)
"""

import numpy as np
import warnings


# ========== Factory function ==========

def get_class(pressure, cloud_mode=None, **kwargs):
    """
    Factory function to get the appropriate Cloud class based on the cloud_mode.
    
    Args:
        pressure (np.ndarray): Pressure grid [bar].
        cloud_mode (str, optional): Mode for the cloud model.
            - None or 'none': No cloud (default)
            - 'gray': Gray cloud model
        **kwargs: Additional keyword arguments passed to the Cloud class.
    
    Returns:
        Cloud: An instance of a Cloud subclass.
    """
    if cloud_mode in [None, 'none']:
        return Cloud(pressure, **kwargs)
    elif cloud_mode.lower() == 'gray':
        return Gray(pressure, **kwargs)
    else:
        raise ValueError(f'Cloud mode {cloud_mode} not recognized.')


# ========== Base Cloud class ==========

class Cloud:
    """
    Base class for handling cloud models (no cloud).
    """
    
    def __init__(self, pressure, **kwargs):
        """
        Initialize the Cloud class.
        
        Args:
            pressure (np.ndarray): Pressure grid [bar].
            **kwargs: Additional keyword arguments (ignored, kept for compatibility).
        """
        if pressure is None:
            raise ValueError('Pressure grid must be provided.')
        self.pressure = np.asarray(pressure)
        # self.pressure = np.sort(self.pressure)
        self.n_atm_layers = len(self.pressure)
    
    def __call__(self, ParamTable, **kwargs):
        """
        Evaluate the cloud model with given parameters.
        For the base Cloud class (no cloud), this returns None.
        
        Args:
            ParamTable (dict): Parameters for the model.
            **kwargs: Additional keyword arguments.
        """
        # No cloud - return None
        self.total_opacity = None
        return
    
    def abs_opacity(self, wave_micron, pressure):
        """
        Calculate the absorption opacity (no cloud case).
        
        Args:
            wave_micron (np.ndarray): Wavelengths in microns.
            pressure (np.ndarray): Pressure levels [bar].
        
        Returns:
            None: No cloud opacity.
        """
        return None
    
    def scat_opacity(self, wave_micron, pressure):
        """
        Calculate the scattering opacity (no cloud case).
        
        Args:
            wave_micron (np.ndarray): Wavelengths in microns.
            pressure (np.ndarray): Pressure levels [bar].
        
        Returns:
            None: No cloud opacity.
        """
        return None


# ========== Gray Cloud ==========

class Gray(Cloud):
    """
    Class for handling Gray cloud models.
    
    Gray cloud model with power-law opacity profile:
    - Opacity at cloud base: log_opa_base_gray (log10 of opacity at base)
    - Cloud base pressure: log_P_base_gray (log10 of pressure [bar])
    - Cloud decay power: f_sed_gray (power-law exponent)
    """
    
    def __init__(self, pressure, **kwargs):
        """
        Initialize the Gray class.
        
        Args:
            pressure (np.ndarray): Pressure grid [bar].
            **kwargs: Additional keyword arguments:
                - wave_cloud_0 (float, optional): Anchor point for non-gray power-law [micron]. Defaults to 1.0.
                - cloud_slope (float, optional): Non-gray cloud slope. Defaults to 0.0 (gray).
                - omega (float, optional): Single-scattering albedo. Defaults to 0.0 (pure absorption).
        """
        # Give arguments to parent class
        super().__init__(pressure, **kwargs)
        
        # Anchor point for non-gray power-law (1 um)
        self.wave_cloud_0 = kwargs.get('wave_cloud_0', 1.)
        # Non-gray cloud slope (default 0 = gray)
        self.cloud_slope = kwargs.get('cloud_slope', 0.)
        # Single-scattering albedo (default 0 = pure absorption)
        self.omega = kwargs.get('omega', 0.)
    
    def __call__(self, ParamTable, mean_wave_micron=None, **kwargs):
        """
        Evaluate the Gray cloud model with given parameters.
        
        Args:
            ParamTable (dict): Parameters for the model.
                Expected parameters:
                - log_P_base_gray (float): log10 of cloud base pressure [log10(bar)]
                - log_opa_base_gray (float): log10 of opacity at cloud base
                - f_sed_gray (float): Cloud decay power (power-law exponent)
                - cloud_slope (float, optional): Non-gray cloud slope. Defaults to 0.0.
                - omega (float, optional): Single-scattering albedo. Defaults to 0.0.
            mean_wave_micron (float, optional): Mean wavelength in microns. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        # Get parameters from ParamTable
        log_P_base = ParamTable.get('log_P_base_gray')
        log_opa_base = ParamTable.get('log_opa_base_gray')
        f_sed_gray = ParamTable.get('f_sed_gray')
        
        # Check if all required parameters are provided
        if None in [log_P_base, log_opa_base, f_sed_gray]:
            raise ValueError(
                "Gray cloud model requires 'log_P_base_gray', 'log_opa_base_gray', and 'f_sed_gray' "
                "parameters in ParamTable."
            )
        
        # Convert from log10 to linear values
        self.P_base = 10 ** log_P_base  # Base pressure [bar]
        self.opa_base = 10 ** log_opa_base  # Opacity at base
        
        # Store parameters
        self.f_sed_gray = f_sed_gray
        
        # Optional parameters (can override defaults from __init__)
        self.cloud_slope = ParamTable.get('cloud_slope', self.cloud_slope)
        self.omega = ParamTable.get('omega', self.omega)
        
        # Calculate total opacity if mean_wave_micron is provided
        if mean_wave_micron is not None:
            self.total_opacity = self.abs_opacity(mean_wave_micron, self.pressure) + \
                self.scat_opacity(mean_wave_micron, self.pressure)
            self.total_opacity = np.squeeze(self.total_opacity)  # to (n_atm_layers, )
        else:
            self.total_opacity = None
    
    def abs_opacity(self, wave_micron, pressure):
        """
        Calculate the absorption opacity.
        
        Args:
            wave_micron (np.ndarray): Wavelengths in microns.
            pressure (np.ndarray): Pressure levels [bar].
        
        Returns:
            np.ndarray: Absorption opacity.
        """
        wave_micron = np.atleast_1d(wave_micron)
        
        # Create cloud opacity
        opacity = np.zeros((len(wave_micron), len(pressure)), dtype=np.float64)
        
        # Determine pressure mask (pressures above the cloud base)
        mask_P = (pressure <= self.P_base)
        
        if not mask_P.any():
            # No layers above cloud base, return zero opacity
            return opacity * (1 - self.omega)
        
        # Pressure-dependent slope (power-law decay above cloud base)
        if self.f_sed_gray is None or self.f_sed_gray == 0:
            # No opacity change with altitude, assume deck below P_base
            slope_pressure = np.ones_like(pressure[mask_P])
        else:
            # Power-law decay above cloud base
            slope_pressure = (pressure[mask_P] / self.P_base) ** self.f_sed_gray
        
        # Wavelength-dependent slope (for non-gray clouds)
        if self.cloud_slope == 0:
            # Gray cloud: no wavelength dependence
            slope_wave = np.ones_like(wave_micron)
        else:
            # Non-gray cloud: wavelength-dependent opacity
            slope_wave = (wave_micron / self.wave_cloud_0) ** self.cloud_slope
        
        # Opacity decreases with power-law above the base
        opacity[:, mask_P] = self.opa_base * slope_wave[:, None] * slope_pressure
        
        return opacity * (1 - self.omega)
    
    def scat_opacity(self, wave_micron, pressure):
        """
        Calculate the scattering opacity.
        
        Args:
            wave_micron (np.ndarray): Wavelengths in microns.
            pressure (np.ndarray): Pressure levels [bar].
        
        Returns:
            np.ndarray: Scattering opacity.
        """
        # Total cloud opacity (absorption + scattering)
        total_opacity = 1 / (1 - self.omega) * self.abs_opacity(wave_micron, pressure)
        # Scattering component
        return total_opacity * self.omega

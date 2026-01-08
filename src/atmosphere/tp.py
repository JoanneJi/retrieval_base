"""
Select different TP profile for creating atmosphere model. TP profile model includes:
    - interpolation
    - ...

Authors
-------
Original version: Sam de Regt
Modified by: Chenyang Ji (2025-12-27)
"""

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import warnings
from typing import cast, Tuple


# ========== Factory function ==========

def get_class(pressure, **kwargs):
    """
    Factory function to get TP-profile class.
    
    Args:
        pressure (np.ndarray): Pressure grid [bar].
        TP_mode (str): Mode for the TP profile.
            - 'interpolation': Interpolate temperature knots (T_0, T_1, ...)
            - 'gradient': Integrate temperature gradients (dlnT_dlnP_0, dlnT_dlnP_1, ...)
        **kwargs: Additional keyword arguments passed to the TP profile class.
    """
    # get TP_mode from kwargs
    TP_mode = kwargs.get('TP_mode', None)
    if TP_mode is None:
        raise ValueError('TP_mode must be provided.')
    
    if TP_mode == 'interpolation':
        return TP_profile_interpolation(pressure, **kwargs)
    elif TP_mode == 'gradient':
        return TP_profile_gradient(pressure, **kwargs)
    # elif TP_mode in ['static', 'constant']:
    #     return TP_profile(pressure, **kwargs)
    else:
        raise ValueError(f'TP mode {TP_mode} not recognized.')


# ========== base TP profile class ==========

class TP_profile:
    """
    Base class for TP profiles.
    """

    def __init__(self, pressure, **kwargs):
        """
        Basic parameters for all TP profiles.

        Args:
            pressure (np.ndarray): Pressure grid [bar].
            **kwargs: Additional keyword arguments (ignored, kept for compatibility).
        """
        if pressure is None:
            raise ValueError('Pressure grid must be provided.')
        self.pressure = np.asarray(pressure)
        self.pressure = np.sort(self.pressure)
        self.n_atm_layers = len(self.pressure)
        self.log_pressure = np.log10(self.pressure)

    @staticmethod
    def get_dlnT_dlnP(temperature, pressure):
        """
        Get the temperature gradient with respect to pressure.

        Args:
            temperature (array): Temperature array.
            pressure (array): Pressure array.

        Returns:
            tuple: Temperature gradient and mean log pressure.
        """
        # Log pressure between layers
        log_pressure = np.log10(pressure)
        mean_log_pressure = 1/2*(log_pressure[1:] + log_pressure[:-1])

        # Temperature gradient
        dlnT_dlnP = np.diff(np.log(temperature)) / np.diff(np.log(pressure))

        return dlnT_dlnP, 10**mean_log_pressure

    def __call__(self, ParamTable):
        """
        Update the TP profile under different sub-classes.

        Returns:
            temperature (np.ndarray): Temperature profile [K].
        """
        if ParamTable.get('temperature') is not None:
            # Get a constant temperature profile
            self.temperature = ParamTable.get('temperature')

        if (self.temperature < 0).any():
            return -np.inf


# ========== interpolation TP profile ==========

class TP_profile_interpolation(TP_profile):
    """
    Interpolation TP profile using given temperature knots.
    """

    def __init__(self, pressure, interp_mode='cubic', symmetric_around_P_phot=False, **kwargs):
        # give arguments to the parent class
        super().__init__(pressure, **kwargs)

        # Read n_knots from kwargs (should be provided in TP_kwargs)
        n_knots = kwargs.get('n_knots', None)
        if n_knots is None:
            warnings.warn("n_knots not provided in TP_kwargs. Using default n_knots=5.")
            n_knots = 5
        self.n_knots = int(n_knots)
        
        self.interp_mode = interp_mode
        self.symmetric_around_P_phot = symmetric_around_P_phot

        # log_P_knots will be set in _set_pressure_knots() when __call__ is invoked
        # Do not set it here to avoid redundancy and ensure it's always set from ParamTable
        self.log_P_knots = None

    def __call__(self, ParamTable):
        self._set_pressure_knots(ParamTable)
        self._get_temperature(ParamTable)

        return super().__call__(ParamTable)

    # pressure knots
    def _set_pressure_knots(self, ParamTable):
        """
        Set the pressure knots for the PT profile.
        
        Note: Parameters should be placed in constant_params (not TP_kwargs),
              as they are read from ParamTable at runtime.
              TP_kwargs is only used for class initialization (n_knots, interp_mode, etc.).

        Args:
            ParamTable (dict): Parameter table (contains constant_params + free_params).
                Expected parameters:
                - log_P_knots (optional): Explicit pressure knots [log10(bar)]
                - log_P_phot (optional): Photospheric pressure [log10(bar)]
                - d_log_P_0+{i} (optional): Separations from bottom pressure
                - d_log_P_phot+{i} (optional): Separations above photosphere
                - d_log_P_phot-{i} (optional): Separations below photosphere
        """
        self.log_P_knots = ParamTable.get('log_P_knots')

        # case 1: constant knots as input --> return
        if self.log_P_knots is not None:
            self.log_P_knots = np.sort(self.log_P_knots)
            return

        # case 2: constant knots as default (equally-spaced knots)
        self.log_P_knots = np.linspace(
            self.log_pressure.min(), self.log_pressure.max(), self.n_knots
            )
        
        # case 3: user-defined important base (or photosphere): log_P_phot
        log_P_base = ParamTable.get('log_P_phot')
        if log_P_base is not None:
            # Relative to photospheric knot
            self.log_P_knots = [self.log_P_knots[0], log_P_base, self.log_P_knots[-1]]

        # case 4: or user-defined important separations (of sensitive photosphere): d_log_P_0+{i} relative to bottom or d_log_P_phot+{i} relative to photosphere
        if (log_P_base is None) and (ParamTable.get('d_log_P_0+1') is not None):
            log_P_base = self.log_pressure.max()
            self.log_P_knots = [self.log_P_knots[0], self.log_P_knots[-1]]

        # raise error if no photosphere is given, but d_log_P_phot+{i} is given
        if (log_P_base is None) and (ParamTable.get('d_log_P_phot+1') is not None):
            raise ValueError('log_P_phot must be given if d_log_P_phot+{i} is given.')

        # 'd_log_P_0+{i}' and 'd_log_P_phot+{i}' cannot be given at the same time
        if (ParamTable.get('log_P_phot')) is not None and (ParamTable.get('d_log_P_0+1') is not None):
            raise ValueError("Cannot mix 'log_P_phot' with 'd_log_P_0+{i}'. "
                             "Use d_log_P_phot+{i} instead.")

        # return if no base is given (or self.log_pressure is None)
        if log_P_base is None:
            return
        
        # get the user-defined separations
        for i in range(1, self.n_knots):
            # Upper and lower separations
            up_i = ParamTable.get(f'd_log_P_phot+{i}')
            if up_i is None:
                up_i = ParamTable.get(f'd_log_P_0+{i}')

            low_i = ParamTable.get(f'd_log_P_phot-{i}')
            if (up_i is None) and (low_i is None):
                break
            
            # if it is symmetric around the base point
            if self.symmetric_around_P_phot:
                low_i = up_i

            if up_i is not None:
                self.log_P_knots.append(log_P_base-up_i)
            if low_i is not None:
                self.log_P_knots.append(log_P_base+low_i)

        # Ascending pressure: sort
        self.log_P_knots = np.sort(np.array(self.log_P_knots))

    # get temperature profile by interpolating temperature knots
    def _get_temperature(self, ParamTable):
        """
        Get the temperature profile by interpolating temperature knots.
        
        Similar to pRT_spectrum.make_pt() in test_retrieval.ipynb:
        - Reads T_0, T_1, T_2, ... from ParamTable
        - Creates equally-spaced log_P_knots if not already set
        - Uses CubicSpline for interpolation in log10(pressure) space

        Args:
            ParamTable (dict): Parameter table containing T_0, T_1, T_2, ... temperature knots.
        """
        # Get temperature values at each knot from ParamTable
        # Look for T_0, T_1, T_2, ... up to n_knots-1
        T_knots = []
        for i in range(self.n_knots):
            T_i = ParamTable.get(f'T_{i}')
            if T_i is None:
                raise ValueError(f'Temperature knot T_{i} not found in ParamTable. '
                               f'Expected {self.n_knots} temperature knots (T_0 to T_{self.n_knots-1}).')
            T_knots.append(T_i)
        
        T_knots = np.array(T_knots[::-1])  # reverse to make sure: T_0 -> bottom, T_{n-1} -> top
        
        # If log_P_knots were not set in _set_pressure_knots, create equally-spaced knots
        # (similar to make_pt() which always creates equally-spaced knots)
        if not hasattr(self, 'log_P_knots') or self.log_P_knots is None:
            warnings.warn(f"log_P_knots were not set in config file, creating equally-spaced knots ranging from {self.log_pressure.min()} to {self.log_pressure.max()} with {len(T_knots)} knots.")
            self.log_P_knots = np.linspace(self.log_pressure.min(), self.log_pressure.max(), num=len(T_knots))

        # Ensure log_P_knots is a numpy array
        log_P_knots_array = np.array(self.log_P_knots)
        
        # Ensure the number of pressure knots matches the number of temperature knots
        if len(log_P_knots_array) != len(T_knots):
            # If mismatch, raise error for the user to check the config file
            raise ValueError(f"Number of pressure knots ({len(log_P_knots_array)}) does not match number of temperature knots ({len(T_knots)}), please check the config file.")
        
        # Sort knots by pressure (ascending) - similar to make_pt()
        sort_idx = np.argsort(log_P_knots_array)
        log_P_knots_sorted = log_P_knots_array[sort_idx]
        T_knots_sorted = T_knots[sort_idx]
        
        # Interpolate temperature onto all pressure levels
        # Use log10(pressure) for interpolation (consistent with make_pt())
        if self.interp_mode == 'cubic':
            # Use CubicSpline (same as make_pt())
            interp_func = CubicSpline(log_P_knots_sorted, T_knots_sorted)
            self.temperature = interp_func(self.log_pressure)
        elif self.interp_mode == 'quadratic':
            # Use quadratic interpolation
            interp_func = interp1d(
                log_P_knots_sorted, T_knots_sorted,
                kind='quadratic',
                bounds_error=False,
                fill_value='extrapolate'  # type: ignore[arg-type]
            )
            self.temperature = interp_func(self.log_pressure)
        elif self.interp_mode == 'linear':
            # Use linear interpolation
            interp_func = interp1d(
                log_P_knots_sorted, T_knots_sorted,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'  # type: ignore[arg-type]
            )
            self.temperature = interp_func(self.log_pressure)
        else:
            # Default to cubic spline if mode not recognized
            warnings.warn(f"Interpolation mode {self.interp_mode} not recognized, using cubic spline interpolation.")
            interp_func = CubicSpline(log_P_knots_sorted, T_knots_sorted)
            self.temperature = interp_func(self.log_pressure)
        
        # Ensure all temperatures are positive
        if (self.temperature < 0).any():
            raise ValueError('Interpolated temperature profile contains negative values.')


# ========== gradient TP profile ==========

class TP_profile_gradient(TP_profile):
    """
    Gradient-based TP profile using temperature gradients dlnT_dlnP.
    
    Similar to PT_profile_free_gradient in pt_profile.py:
    - Reads dlnT_dlnP_0, dlnT_dlnP_1, ... from ParamTable
    - Interpolates gradients onto all pressure levels
    - Integrates gradients from a base pressure/temperature to get temperature profile
    """
    
    def __init__(self, pressure, interp_mode='linear', symmetric_around_P_phot=False, **kwargs):
        """
        Initialize the TP_profile_gradient class.
        
        Args:
            pressure (np.ndarray): Pressure grid [bar].
            interp_mode (str): Interpolation mode for gradients ('linear', 'cubic', 'quadratic').
            symmetric_around_P_phot (bool): Flag for symmetry around photospheric pressure.
            **kwargs: Additional keyword arguments passed to parent class.
                Expected in kwargs (from TP_kwargs):
                - n_knots (int): Number of gradient knots.
        """
        # Give arguments to the parent class
        super().__init__(pressure, **kwargs)
        
        # Read n_knots from kwargs (should be provided in TP_kwargs)
        n_knots = kwargs.get('n_knots', None)
        if n_knots is None:
            warnings.warn("n_knots not provided in TP_kwargs. Using default n_knots=5.")
            n_knots = 5
        self.n_knots = int(n_knots)
        
        self.interp_mode = interp_mode
        self.symmetric_around_P_phot = symmetric_around_P_phot
        
        # Use natural log for pressure (needed for gradient integration)
        self.ln_pressure = np.log(self.pressure)  # type: ignore[assignment]
        
    def __call__(self, ParamTable):
        """
        Set the parameters for the TP profile.
        
        Args:
            ParamTable (dict): Parameter table.
        """
        self._set_pressure_knots(ParamTable)
        self._set_temperature_gradients(ParamTable)
        self._get_temperature(ParamTable)
        
        return super().__call__(ParamTable)
    
    def _set_pressure_knots(self, ParamTable):
        """
        Set the pressure knots for the gradient profile.
        
        Note: Parameters should be placed in constant_params (not TP_kwargs),
              as they are read from ParamTable at runtime.
              TP_kwargs is only used for class initialization (n_knots, interp_mode, etc.).
        
        Args:
            ParamTable (dict): Parameter table (contains constant_params + free_params).
                Expected parameters:
                - log_P_knots (optional): Explicit pressure knots [log10(bar)]
                - log_P_phot (optional): Photospheric pressure [log10(bar)]
                - d_log_P_0+{i} (optional): Separations from bottom pressure
                - d_log_P_phot+{i} (optional): Separations above photosphere
                - d_log_P_phot-{i} (optional): Separations below photosphere
        """
        self.log_P_knots = ParamTable.get('log_P_knots')
        
        # case 1: constant knots as input --> return
        if self.log_P_knots is not None:
            self.log_P_knots = np.sort(self.log_P_knots)
            return
        
        # case 2: constant knots as default (equally-spaced knots)
        self.log_P_knots = np.linspace(
            self.log_pressure.min(), self.log_pressure.max(), self.n_knots
        )
        
        # case 3: user-defined important base (or photosphere): log_P_phot
        log_P_base = ParamTable.get('log_P_phot')
        if log_P_base is not None:
            # Relative to photospheric knot
            self.log_P_knots = [self.log_P_knots[0], log_P_base, self.log_P_knots[-1]]
        
        # case 4: or user-defined important separations
        if (log_P_base is None) and (ParamTable.get('d_log_P_0+1') is not None):
            log_P_base = self.log_pressure.max()
            self.log_P_knots = [self.log_P_knots[0], self.log_P_knots[-1]]
        
        # raise error if no photosphere is given, but d_log_P_phot+{i} is given
        if (log_P_base is None) and (ParamTable.get('d_log_P_phot+1') is not None):
            raise ValueError('log_P_phot must be given if d_log_P_phot+{i} is given.')
        
        # 'd_log_P_0+{i}' and 'd_log_P_phot+{i}' cannot be given at the same time
        if (ParamTable.get('log_P_phot') is not None) and (ParamTable.get('d_log_P_0+1') is not None):
            raise ValueError("Cannot mix 'log_P_phot' with 'd_log_P_0+{i}'. "
                           "Use d_log_P_phot+{i} instead.")
        
        # return if no base is given
        if log_P_base is None:
            return
        
        # get the user-defined separations
        for i in range(1, self.n_knots):
            # Upper and lower separations
            up_i = ParamTable.get(f'd_log_P_phot+{i}')
            if up_i is None:
                up_i = ParamTable.get(f'd_log_P_0+{i}')
            
            low_i = ParamTable.get(f'd_log_P_phot-{i}')
            if (up_i is None) and (low_i is None):
                break
            
            # if it is symmetric around the base point
            if self.symmetric_around_P_phot:
                low_i = up_i
            
            if up_i is not None:
                self.log_P_knots.append(log_P_base - up_i)
            if low_i is not None:
                self.log_P_knots.append(log_P_base + low_i)
        
        # Ascending pressure: sort
        self.log_P_knots = np.sort(np.array(self.log_P_knots))
        self.P_knots = 10**self.log_P_knots
    
    def _set_temperature_gradients(self, ParamTable):
        """
        Get the temperature gradients for the TP profile.
        
        Similar to PT_profile_free_gradient._set_temperature_gradients():
        - Gets dlnT_dlnP_0, dlnT_dlnP_1, ... from ParamTable
        - Reverses the array to match ascending pressure order (dlnT_dlnP_0 -> top, dlnT_dlnP_{n-1} -> bottom)
        - Interpolates gradients onto all pressure levels in log10 space
        - Note: dlnT_dlnP is defined as d(ln T)/d(ln P), but we interpolate in log10 space
          This is consistent with the reference implementation.
        
        Args:
            ParamTable (dict): Parameter table containing dlnT_dlnP_0, dlnT_dlnP_1, ... gradient knots.
        """
        # Get the temperature gradients at each knot
        # dlnT_dlnP_0 corresponds to lowest pressure (top), dlnT_dlnP_{n-1} to highest (bottom)
        self.dlnT_dlnP_knots = np.array([
            ParamTable.get(f'dlnT_dlnP_{i}') for i in range(self.n_knots)
        ])
        
        # Check for None values
        if None in self.dlnT_dlnP_knots:
            missing = [i for i in range(self.n_knots) if self.dlnT_dlnP_knots[i] is None]
            raise ValueError(f'Temperature gradients not found in ParamTable: {[f"dlnT_dlnP_{i}" for i in missing]}. '
                           f'Expected {self.n_knots} gradient knots (dlnT_dlnP_0 to dlnT_dlnP_{self.n_knots-1}).')
        
        # CRITICAL: Reverse to match ascending pressure order
        # Before reversal: 'dlnT_dlnP_0' at the bottom, 'dlnT_dlnP_{n-1}' at the top
        #                 ascending pressure from top to bottom
        # After reversal: dlnT_dlnP_knots[0] corresponds to highest pressure (top)
        #                 dlnT_dlnP_knots[-1] corresponds to lowest pressure (bottom)
        # This matches the reference implementation (PT_profile_free_gradient line 204)
        self.dlnT_dlnP_knots = self.dlnT_dlnP_knots[::-1]
        
        # Ensure log_P_knots is set and sorted (ascending in pressure)
        if not hasattr(self, 'log_P_knots') or self.log_P_knots is None:
            warnings.warn(f"log_P_knots were not set, creating equally-spaced knots with {self.n_knots} knots.")
            self.log_P_knots = np.linspace(
                self.log_pressure.min(), self.log_pressure.max(), self.n_knots
            )
            self.P_knots = 10**self.log_P_knots
        
        # Ensure log_P_knots is sorted (ascending in pressure)
        self.log_P_knots = np.sort(np.array(self.log_P_knots))
        
        # Ensure the number of pressure knots matches the number of gradient knots
        if len(self.log_P_knots) != len(self.dlnT_dlnP_knots):
            raise ValueError(f"Number of pressure knots ({len(self.log_P_knots)}) does not match number of gradient knots ({len(self.dlnT_dlnP_knots)}), please check the config file.")
        
        # Interpolate gradients onto each pressure level
        # Note: We interpolate in log10 space, but dlnT_dlnP is defined as d(ln T)/d(ln P)
        # The conversion factor ln(10) is implicitly handled by the interpolation
        # This matches the reference implementation (PT_profile_free_gradient line 207-212)
        if self.interp_mode == 'cubic':
            kind = 'cubic'
        elif self.interp_mode == 'quadratic':
            kind = 'quadratic'
        elif self.interp_mode == 'linear':
            kind = 'linear'
        else:
            warnings.warn(f"Interpolation mode {self.interp_mode} not recognized, using linear interpolation.")
            kind = 'linear'
        
        interp_func = interp1d(
            self.log_P_knots, self.dlnT_dlnP_knots,
            kind=kind,
            bounds_error=False,
            fill_value='extrapolate'  # type: ignore[arg-type]
        )
        
        # Interpolate onto all pressure levels (ascending in pressure)
        self.dlnT_dlnP = interp_func(self.log_pressure)
    
    def _get_temperature(self, ParamTable):
        """
        Get the temperature profile by integrating temperature gradients.
        
        Similar to PT_profile_free_gradient._get_temperature():
        - Uses a base pressure and temperature (P_phot/T_phot or P_0/T_0)
        - Integrates gradients separately for above and below the base pressure
        - Formula: T_j = T_{j-1} * (P_j/P_{j-1})^dlnT_dlnP_j
        
        Note: T_phot, T_0, and P_phot should be placed in constant_params,
              as they are read from ParamTable at runtime.
        
        Args:
            ParamTable (dict): Parameter table (contains constant_params + free_params).
                Expected parameters:
                - T_phot (optional): Photospheric temperature [K]
                - T_0 (optional): Base temperature [K] (alternative to T_phot)
                - P_phot (optional): Photospheric pressure [bar] (defaults to max pressure)
        """
        # Compute the temperatures relative to a base pressure
        # Use P_knots if available, otherwise use pressure range
        if hasattr(self, 'P_knots') and self.P_knots is not None:
            P_knots_array = np.array(self.P_knots)
            P_base = ParamTable.get('P_phot', min(float(P_knots_array.max()), float(self.pressure.max()))) # type: ignore[arg-type]
        else:
            P_base = ParamTable.get('P_phot', float(self.pressure.max())) # type: ignore[arg-type]
        
        T_base = ParamTable.get('T_phot', ParamTable.get('T_0'))
        if T_base is None:
            raise ValueError('Either T_phot or T_0 must be provided in ParamTable for gradient integration.')
        
        # Mask for above and below the base pressure
        mask_above = (self.pressure <= P_base)
        mask_below = (self.pressure > P_base)
        
        self.temperature = np.zeros_like(self.pressure)
        
        for mask in [mask_above, mask_below]:
            if not mask.any():
                continue
            
            # Get gradients and pressures for this region
            dlnT_dlnP = self.dlnT_dlnP[mask]
            ln_P = self.ln_pressure[mask]
            
            # Sort relative to base pressure (closest first)
            idx = np.argsort(np.abs(ln_P - np.log(P_base)))
            sorted_ln_P = ln_P[idx]
            sorted_dlnT_dlnP = dlnT_dlnP[idx]
            
            ln_T = []
            for i, (ln_P_i, dlnT_dlnP_i) in enumerate(zip(sorted_ln_P, sorted_dlnT_dlnP)):
                if i == 0:
                    # Compare to the base pressure
                    dln_P_i = ln_P_i - np.log(P_base)
                    ln_T_previous = np.log(T_base)
                else:
                    # Compare to the previous pressure level
                    dln_P_i = ln_P_i - sorted_ln_P[i-1]
                    ln_T_previous = ln_T[-1]
                
                # T_j = T_{j-1} * (P_j/P_{j-1})^dlnT_dlnP_j
                # In log space: ln_T_j = ln_T_{j-1} + dln_P_j * dlnT_dlnP_j
                ln_T_i = ln_T_previous + dln_P_i * dlnT_dlnP_i
                ln_T.append(ln_T_i)
            
            # Sort by ascending pressure to restore original order
            idx_restore = np.argsort(sorted_ln_P)
            self.temperature[mask] = np.exp(np.array(ln_T)[idx_restore])
        
        # Ensure all temperatures are positive
        if (self.temperature < 0).any():
            raise ValueError('Integrated temperature profile contains negative values.')


# # ========== Test/Debug section ==========
# # Run "python -m atmosphere.tp" in src/ to test TP profile functionality

# if __name__ == "__main__":
#     import os
#     import sys
#     import matplotlib.pyplot as plt
    
#     # Add parent directory to path to import modules
#     sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
#     from retrieval.parameters import Parameters
#     from core.paths import SRC_DIR
    
#     print("=" * 60)
#     print("Testing TP Profile Classes")
#     print("=" * 60)
    
#     # Load parameters from config_example.py
#     config_file = os.path.join(SRC_DIR, "config/config_example.py")
#     if not os.path.exists(config_file):
#         print(f"Warning: Config file {config_file} not found.")
#         print("Using default test parameters...")
#         # Create test parameters manually
#         test_free_params = {
#             'T0': ([1000, 5000], r'$T_0$', 'uniform'),
#             'T1': ([500, 4500], r'$T_1$', 'uniform'),
#             'T2': ([500, 4000], r'$T_2$', 'uniform'),
#             'T3': ([500, 4000], r'$T_3$', 'uniform'),
#             'T4': ([500, 4000], r'$T_4$', 'uniform'),
#         }
#         test_constant_params = {
#             'n_knots': 5,
#             'log_P_range': (-5., 2.),
#             'n_atm_layers': 70,
#         }
#         test_TP_kwargs = {
#             'TP_mode': 'interpolation',
#             'n_knots': 5,
#             'interp_mode': 'cubic',
#             'log_P_range': (-5., 2.),
#             'n_atm_layers': 70,
#         }
#     else:
#         # Load from config file
#         parameters = Parameters(config_file=config_file)
        
#         # Extract TP_kwargs from config file
#         import importlib.util
#         spec = importlib.util.spec_from_file_location("config_module", config_file)
#         if spec is None or spec.loader is None:
#             raise ValueError(f"Could not load config file: {config_file}")
#         config_module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(config_module)
#         test_TP_kwargs = getattr(config_module, 'TP_kwargs', {})
        
#         # Create test parameter values (use midpoints of prior ranges)
#         test_params = {}
#         for key, info in parameters.free_params.items():
#             bounds = info["bounds"]
#             test_params[key] = (bounds[0] + bounds[1]) / 2.0
        
#         # Add constant params
#         test_params.update(parameters.constant_params)
        
#         test_free_params = parameters.free_params
#         test_constant_params = parameters.constant_params
    
#     # Test 1: Interpolation mode
#     print("\n" + "-" * 60)
#     print("Test 1: TP_profile_interpolation")
#     print("-" * 60)
    
#     try:
#         # Create TP profile instance
#         tp_kwargs_interp = test_TP_kwargs.copy()
#         tp_kwargs_interp['TP_mode'] = 'interpolation'
#         tp_interp = get_class(**tp_kwargs_interp)
        
#         # Create test ParamTable with temperature knots
#         if 'T0' in test_params or 'T0' in test_free_params:
#             param_table_interp = {}
#             for i in range(tp_interp.n_knots):
#                 key = f'T_{i}'
#                 if key in test_params:
#                     param_table_interp[key] = test_params[key]
#                 else:
#                     # Use default test values
#                     param_table_interp[key] = 2000 - i * 200  # Decreasing temperature
            
#             # Add constant params
#             param_table_interp.update(test_constant_params)
            
#             # Call TP profile
#             result = tp_interp(param_table_interp)
            
#             # Check results
#             print(f"✓ TP_profile_interpolation created successfully")
#             print(f"  - Number of layers: {len(tp_interp.pressure)}")
#             print(f"  - Pressure range: {tp_interp.pressure.min():.2e} - {tp_interp.pressure.max():.2e} bar")
#             print(f"  - Temperature range: {tp_interp.temperature.min():.1f} - {tp_interp.temperature.max():.1f} K")
#             print(f"  - All temperatures positive: {(tp_interp.temperature > 0).all()}")
#             print(f"  - Number of knots: {tp_interp.n_knots}")
#             print(f"  - Interpolation mode: {tp_interp.interp_mode}")
            
#             if hasattr(tp_interp, 'log_P_knots'):
#                 print(f"  - Pressure knots (log10): {tp_interp.log_P_knots}")
            
#         else:
#             print("⚠ Skipping interpolation test: T_0, T_1, ... not found in parameters")
    
#     except Exception as e:
#         print(f"✗ Error in interpolation test: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # Test 2: Gradient mode
#     print("\n" + "-" * 60)
#     print("Test 2: TP_profile_gradient")
#     print("-" * 60)
    
#     try:
#         # Create TP profile instance
#         tp_kwargs_grad = test_TP_kwargs.copy()
#         tp_kwargs_grad['TP_mode'] = 'gradient'
#         tp_grad = get_class(**tp_kwargs_grad)
        
#         # Create test ParamTable with gradients
#         if 'dlnT_dlnP_0' in test_params or 'dlnT_dlnP_0' in test_free_params:
#             param_table_grad = {}
#             for i in range(tp_grad.n_knots):
#                 key = f'dlnT_dlnP_{i}'
#                 if key in test_params:
#                     param_table_grad[key] = test_params[key]
#                 else:
#                     # Use default test values (small positive gradients)
#                     param_table_grad[key] = 0.05 + i * 0.01
            
#             # Add base temperature (required for gradient mode)
#             if 'T_phot' in test_constant_params:
#                 param_table_grad['T_phot'] = test_constant_params['T_phot']
#             elif 'T_0' in test_constant_params:
#                 param_table_grad['T_0'] = test_constant_params['T_0']
#             else:
#                 param_table_grad['T_phot'] = 1500.0  # Default test value
#                 print("  Using default T_phot = 1500 K")
            
#             # Add optional P_phot
#             if 'P_phot' in test_constant_params:
#                 param_table_grad['P_phot'] = test_constant_params['P_phot']
            
#             # Add constant params
#             param_table_grad.update(test_constant_params)
            
#             # Call TP profile
#             result = tp_grad(param_table_grad)
            
#             # Check results
#             print(f"✓ TP_profile_gradient created successfully")
#             print(f"  - Number of layers: {len(tp_grad.pressure)}")
#             print(f"  - Pressure range: {tp_grad.pressure.min():.2e} - {tp_grad.pressure.max():.2e} bar")
#             print(f"  - Temperature range: {tp_grad.temperature.min():.1f} - {tp_grad.temperature.max():.1f} K")
#             print(f"  - All temperatures positive: {(tp_grad.temperature > 0).all()}")
#             print(f"  - Number of knots: {tp_grad.n_knots}")
#             print(f"  - Interpolation mode: {tp_grad.interp_mode}")
            
#             if hasattr(tp_grad, 'log_P_knots'):
#                 print(f"  - Pressure knots (log10): {tp_grad.log_P_knots}")
#             if hasattr(tp_grad, 'dlnT_dlnP') and tp_grad.dlnT_dlnP is not None:
#                 dlnT_dlnP_array = np.array(tp_grad.dlnT_dlnP)
#                 print(f"  - Gradient range: {dlnT_dlnP_array.min():.4f} - {dlnT_dlnP_array.max():.4f}")
        
#         else:
#             print("⚠ Skipping gradient test: dlnT_dlnP_0, dlnT_dlnP_1, ... not found in parameters")
    
#     except Exception as e:
#         print(f"✗ Error in gradient test: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # Optional: Plot temperature profiles
#     try:
#         plot = os.environ.get('TP_TEST_PLOT', '0').lower() in ['1', 'true', 'yes']
#         if plot and 'tp_interp' in locals() and 'tp_grad' in locals():
#             print("\n" + "-" * 60)
#             print("Plotting temperature profiles...")
#             print("-" * 60)
            
#             fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
#             # Plot interpolation mode
#             ax = axes[0]
#             ax.plot(tp_interp.temperature, np.log10(tp_interp.pressure), 'b-', label='Interpolation')
#             if hasattr(tp_interp, 'log_P_knots'):
#                 T_knots_plot = [param_table_interp.get(f'T_{i}', 0) for i in range(tp_interp.n_knots)]
#                 P_knots_plot = 10**tp_interp.log_P_knots
#                 ax.scatter(T_knots_plot, np.log10(P_knots_plot), c='r', s=50, zorder=5, label='Knots')
#             ax.set_xlabel('Temperature [K]')
#             ax.set_ylabel('log10(Pressure [bar])')
#             ax.set_title('Interpolation Mode')
#             ax.legend()
#             ax.grid(True, alpha=0.3)
#             ax.invert_yaxis()
            
#             # Plot gradient mode
#             ax = axes[1]
#             ax.plot(tp_grad.temperature, np.log10(tp_grad.pressure), 'g-', label='Gradient')
#             if hasattr(tp_grad, 'log_P_knots'):
#                 P_knots_plot = 10**tp_grad.log_P_knots
#                 # Estimate T at knots from interpolated profile
#                 from scipy.interpolate import interp1d
#                 interp_T = interp1d(np.log10(tp_grad.pressure), tp_grad.temperature, 
#                                    bounds_error=False, fill_value='extrapolate')
#                 T_knots_plot = interp_T(tp_grad.log_P_knots)
#                 ax.scatter(T_knots_plot, np.log10(P_knots_plot), c='r', s=50, zorder=5, label='Knots')
#             ax.set_xlabel('Temperature [K]')
#             ax.set_ylabel('log10(Pressure [bar])')
#             ax.set_title('Gradient Mode')
#             ax.legend()
#             ax.grid(True, alpha=0.3)
#             ax.invert_yaxis()
            
#             plt.tight_layout()
#             output_file = os.path.join(SRC_DIR, "..", "output", "tp_profile_test.png")
#             os.makedirs(os.path.dirname(output_file), exist_ok=True)
#             plt.savefig(output_file, dpi=150, bbox_inches='tight')
#             print(f"✓ Plot saved to: {output_file}")
#             plt.close()
    
#     except Exception as e:
#         print(f"⚠ Could not create plot: {e}")
    
#     print("\n" + "=" * 60)
#     print("Test completed!")
#     print("=" * 60)
#     print("\nTo enable plotting, set environment variable:")
#     print("  export TP_TEST_PLOT=1")
#     print("\nTo run this test:")
#     print("  cd src/")
#     print("  python -m atmosphere.tp")
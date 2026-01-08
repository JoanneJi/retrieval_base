"""
Select different chemistry for creating atmosphere model. Chemistry includes:
    - free chemistry
    - equilibrium chemistry
    - ...

Authors
-------
Original version: Sam de Regt
Modified by: Chenyang Ji (2025-12-27)
"""

import numpy as np
import pandas as pd
import pathlib
import warnings
from typing import Any, Optional

# Get directory path for loading species info
directory_path = pathlib.Path(__file__).parent.resolve()
# /retrieval_base/src/atmosphere

from core.paths import SRC_DIR


# ========== Factory function ==========

def get_class(pressure, line_species, **kwargs):
    """
    Factory function to determine and return the appropriate Chemistry class based on chem_mode.
    
    Args:
        pressure (np.ndarray): Atmospheric pressure grid [bar].
        line_species (list): List of line species used in opacity calculations.
        chem_mode (str, optional): Chemistry mode.
            - 'free': Free chemistry (abundances are free parameters)
            - 'equilibrium': Equilibrium chemistry (placeholder for future implementation)
        **kwargs: Additional keyword arguments passed to the Chemistry class.
    
    Returns:
        Chemistry: An instance of a Chemistry subclass.
    """
    # get chem_mode from kwargs
    chem_mode = kwargs.get('chem_mode', None)
    if chem_mode is None:
        raise ValueError('chem_mode must be provided.')
    
    if chem_mode == 'free':
        return FreeChemistry(pressure, line_species, **kwargs)
    elif chem_mode == 'equilibrium':
        return EquilibriumChemistry(pressure, line_species, **kwargs)
    else:
        raise ValueError(f"Chemistry mode '{chem_mode}' not implemented. "
                        f"Available modes: 'free', 'equilibrium'.")


# ========== Base Chemistry class ==========

class Chemistry:
    """
    Base class for handling chemical species and their properties.
    """
    
    # Try to load species_info.csv from various possible locations
    # Default path: SRC_DIR / "atmosphere" / "species_info.csv"
    _default_species_info_path = SRC_DIR / "atmosphere" / "species_info.csv"
    _species_info_paths = [
        _default_species_info_path,
        directory_path / 'species_info.csv',
        pathlib.Path('species_info.csv'),  # Current working directory
    ]
    
    _species_info = None
    
    @classmethod
    def _load_species_info(cls):
        """Load species information from CSV file."""
        if cls._species_info is not None:
            return cls._species_info
        
        for path in cls._species_info_paths:
            if path.exists():
                try:
                    cls._species_info = pd.read_csv(path, index_col=0)
                    return cls._species_info
                except Exception as e:
                    warnings.warn(f"Could not load species_info from {path}: {e}")
                    continue
        
        # If no file found, create a minimal default
        warnings.warn("No species_info.csv found. Creating minimal default species info.")
        species_names = ['H2', 'He', '12CO', 'H2O', 'CH4', 'NH3', 'CO2', '13CO']
        cls._species_info = pd.DataFrame({
            'pRT_name': ['H2', 'He', '12CO', 'H2O', 'CH4', 'NH3', 'CO2', '13CO'],
            'mass': [2.016, 4.002602, 28.01, 18.010565, 16.0313, 17.026549, 43.989830, 28.99827],
            'C': [0, 0, 1, 0, 1, 0, 1, 1],
            'O': [0, 0, 1, 1, 0, 0, 2, 1],
            'H': [2, 0, 0, 2, 4, 3, 0, 0],
        })
        cls._species_info.index = species_names
        return cls._species_info
    
    def __init__(self, pressure, line_species, **kwargs):
        """
        Initialize the Chemistry class.
        
        Args:
            pressure (np.ndarray): Pressure levels [bar].
            line_species (list): List of line species.
            **kwargs: Additional keyword arguments:
                - LineOpacity (list, optional): Custom opacity objects. Defaults to None.
                - species_info_path (str, optional): Path to species_info.csv file. 
                  If None, uses default: SRC_DIR / "atmosphere" / "species_info.csv"
        """
        # Get LineOpacity and species_info_path from kwargs
        LineOpacity = kwargs.get('LineOpacity', None)
        species_info_path = kwargs.get('species_info_path', None)
        
        # Load species info
        # Priority: 1) user-specified path from config, 2) default path (SRC_DIR / "atmosphere" / "species_info.csv")
        if species_info_path is not None:
            species_info_path = pathlib.Path(species_info_path)
            if species_info_path.exists():
                self.species_info = pd.read_csv(species_info_path, index_col=0)
            else:
                warnings.warn(f"species_info_path {species_info_path} not found. Using default: {self._default_species_info_path}")
                # Try default path: SRC_DIR / "atmosphere" / "species_info.csv"
                if self._default_species_info_path.exists():
                    self.species_info = pd.read_csv(self._default_species_info_path, index_col=0)
                else:
                    # Fallback to _load_species_info which tries multiple paths
                    self.species_info = self._load_species_info()
        else:
            # No path specified, try default path first
            if self._default_species_info_path.exists():
                self.species_info = pd.read_csv(self._default_species_info_path, index_col=0)
            else:
                # Fallback to _load_species_info which tries multiple paths
                self.species_info = self._load_species_info()
        
        # Initialize line species list -- add H2 and He
        self.line_species = [*line_species, 'H2', 'He']
        
        # Custom line-opacities
        # LineOpacity can be specified in the config file via chemistry_kwargs
        if LineOpacity is not None:
            for LineOpacity_i in LineOpacity:
                if hasattr(LineOpacity_i, 'line_species'):
                    self.line_species.append(LineOpacity_i.line_species)
        
        # Store the species names that are in line_species
        self.species = []
        for species_i in self.species_info.index:
            line_species_i = self.read_species_info(species_i, 'pRT_name')
            
            if line_species_i not in self.line_species:
                continue
            
            if species_i == 'H2_lines':
                # Add H2 as a line_species separately
                self.add_H2_line_species = line_species_i
                self.line_species.remove(self.add_H2_line_species)
                continue
            
            self.species.append(species_i)
        
        self.pressure = pressure
        self.n_atm_layers = len(self.pressure)
        
        # Set to None initially, changed during evaluation
        self.mass_fractions = None
        self.VMRs = None
        self.MMW = None
        self.CO = None
        self.FeH = None
        self.temperature = None
    
    def read_species_info(self, species, info_key):
        """
        Read species information from the species_info DataFrame.
        
        Args:
            species (str): Species name.
            info_key (str): Key to retrieve specific information.
                - 'pRT_name': pRT line species name
                - 'mass': Molecular mass
                - 'C', 'O', 'H': Number of C, O, H atoms
                - 'COH': List of [C, O, H] counts
        
        Returns:
            Various: Information based on the info_key.
        """
        if species not in self.species_info.index:
            raise ValueError(f"Species '{species}' not found in species_info.")
        
        if info_key == 'pRT_name':
            return self.species_info.loc[species, 'pRT_name']
        elif info_key == 'mass':
            return self.species_info.loc[species, 'mass']
        elif info_key in ['C', 'O', 'H', 'N', 'S']:
            if info_key in self.species_info.columns:
                return self.species_info.loc[species, info_key]
            else:
                return 0  # Default to 0 if column doesn't exist
        elif info_key == 'COH':
            # modify here if you want to add N and S
            C = self.species_info.loc[species, 'C'] if 'C' in self.species_info.columns else 0
            O = self.species_info.loc[species, 'O'] if 'O' in self.species_info.columns else 0
            H = self.species_info.loc[species, 'H'] if 'H' in self.species_info.columns else 0
            return [C, O, H]
        elif info_key == 'label':
            return self.species_info.loc[species, 'mathtext_name']
        elif info_key == 'color':
            return self.species_info.loc[species, 'color']
        else:
            raise ValueError(f"Unknown info_key: {info_key}")
    
    def get_VMRs(self, ParamTable):
        """
        Placeholder method to get volume mixing ratios (VMRs).
        Should be implemented by child classes.
        
        Args:
            ParamTable (dict): Parameters for the model.
        """
        raise NotImplementedError("Subclasses should implement this method")
    
    def convert_to_MFs(self):
        """
        Convert volume mixing ratios (VMRs) to mass fractions (MFs).
        
        Since get_VMRs() ensures all required line species are already in VMRs dict
        """
        if self.MMW is None:
            raise ValueError("MMW must be set before converting to mass fractions.")
        if self.VMRs is None:
            raise ValueError("VMRs must be set before converting to mass fractions.")
        
        # Convert to mass-fractions using mass-ratio
        # VMRs dict uses species name as key (e.g., 'H2O', '12CO')
        # mass_fractions dict uses pRT_name as key (e.g., '1H2-16O', '12C-16O') for pRT compatibility
        self.mass_fractions = {'MMW': self.MMW}
        
        for species_i, VMR_i in self.VMRs.items():
            # species_i is already a species name (from species_info.index)
            # Check if it's in species_info
            if species_i not in self.species_info.index:
                # Try to find it by pRT_name (for backward compatibility)
                species_name = None
                for sp in self.species_info.index:
                    if self.read_species_info(sp, 'pRT_name') == species_i:
                        species_name = sp
                        break
                if species_name is None:
                    warnings.warn(f"Species {species_i} not found in species_info, skipping.")
                    continue
            else:
                species_name = species_i
            
            # Get pRT_name and mass for this species
            line_species_i = self.read_species_info(species_name, 'pRT_name')
            mass_i = self.read_species_info(species_name, 'mass')
            
            # Use pRT_name as key in mass_fractions (for pRT compatibility)
            self.mass_fractions[line_species_i] = VMR_i * mass_i / self.MMW
    
    def get_diagnostics(self):
        """
        Calculate diagnostics such as C/O and Fe/H ratios.
        """
        if self.VMRs is None:
            warnings.warn("VMRs are not set, skipping diagnostics.")
            return
        
        C = np.zeros(self.n_atm_layers)
        O = np.zeros(self.n_atm_layers)
        H = np.zeros(self.n_atm_layers)
        
        for species_i, VMR_i in self.VMRs.items():
            # species_i is already a species name (from species_info.index)
            # VMRs dict uses species name as key
            if species_i not in self.species_info.index:
                # Skip if not found (should not happen, but for safety)
                continue
            
            # modify here if you want to add N and S
            COH_i = self.read_species_info(species_i, 'COH')
            if isinstance(COH_i, (list, tuple, np.ndarray)) and len(COH_i) >= 3:
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i
        
        # Calculate C/O ratio
        if self.CO is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                CO_ratio = C / O
                self.CO = np.nanmean(CO_ratio[O > 0])
                if np.isnan(self.CO):
                    self.CO = 0.0
        
        # Calculate Fe/H (using C/H as proxy for metallicity)
        if self.FeH is None:
            log_CH_solar = 8.46 - 12  # Asplund et al. (2021)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_CH = np.log10(C / H)
                mask_valid = (C > 0) & (H > 0)
                if mask_valid.any():
                    self.FeH = np.nanmean(log_CH[mask_valid] - log_CH_solar)
                else:
                    self.FeH = 0.0
    
    def remove_species(self):
        """
        Remove the contribution of specified species by setting their mass fractions to zero.
        This is a placeholder - can be extended to support neglect_species dict.
        """
        # Placeholder for future implementation
        pass
    
    def _share_isotope_ratios(self, ParamTable):
        """
        Share isotope ratios between molecules.
        
        Args:
            ParamTable (dict): Parameters for the model.
                - Isotopologue-specific ratios: '13CO_ratio', 'C18O_ratio', 'HDO_ratio', etc.
                - Elemental isotope ratios: '12/13C_ratio', '16/18O_ratio', 'H/D_ratio', '14/15N_ratio'
        
        Returns:
            tuple: (all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios)
                Each is a dict mapping species names to their ratios relative to the main isotopologue.
                Example: all_CO_ratios = {'12CO': 1.0, '13CO': 0.01, 'C18O': 0.001, ...}
        """
        # First looks for isotopologue ratio, then for isotope ratio, otherwise sets abundance to 0
        all_CO_ratios = {
            '12CO': 1., 
            '13CO': 1./ParamTable.get('13CO_ratio', ParamTable.get('12/13C_ratio', np.inf)), 
            'C18O': 1./ParamTable.get('C18O_ratio', ParamTable.get('16/18O_ratio', np.inf)), 
            'C17O': 1./ParamTable.get('C17O_ratio', ParamTable.get('16/17O_ratio', np.inf)), 
        }
        all_H2O_ratios = {
            'H2O':     1., 
            'H2(18)O': 1./ParamTable.get('H2(18)O_ratio', ParamTable.get('16/18O_ratio', np.inf)), 
            'H2(17)O': 1./ParamTable.get('H2(17)O_ratio', ParamTable.get('16/17O_ratio', np.inf)), 
            'HDO':     1./ParamTable.get('HDO_ratio', ParamTable.get('H/D_ratio', np.inf)/2.), 
        }
        all_CH4_ratios = {
            'CH4':   1., 
            '13CH4': 1./ParamTable.get('13CH4_ratio', ParamTable.get('12/13C_ratio', np.inf)), 
            'CH3D':  1./ParamTable.get('CH3D_ratio', ParamTable.get('H/D_ratio', np.inf)/4.),
        }
        all_NH3_ratios = {
            'NH3':   1., 
            '15NH3': 1./ParamTable.get('15NH3_ratio', ParamTable.get('14/15N_ratio', np.inf)), 
        }
        all_CO2_ratios = {
            'CO2':     1., 
            '13CO2':   1./ParamTable.get('13CO2_ratio', ParamTable.get('12/13C_ratio', np.inf)), 
            'CO(18)O': 1./ParamTable.get('CO(18)O_ratio', ParamTable.get('16/18O_ratio', np.inf)/2.), 
            'CO(17)O': 1./ParamTable.get('CO(17)O_ratio', ParamTable.get('16/17O_ratio', np.inf)/2.), 
        }
        return all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios
    
    def get_isotope_VMRs(self, ParamTable):
        """
        Calculate the volume mixing ratios (VMRs) for isotopologues.

        If the calculation is under EquilibriumChemistry, the total VMR will be conserved.

        **  VMR_isotopologue = VMR_main x (minor_main_ratio) / (sum_of_ratios)  **
        
        e.g., for CO isotopologues:
            - VMR_main = VMR_12CO = 0.001 (0.1%)
            - all_CO_ratios = {'12CO': 1.0, '13CO': 0.01, 'C18O': 0.001}
            + EquilibriumChemistry
                - VMR_13CO = VMR_12CO x (13CO/12CO) / (12CO/12CO + 13CO/12CO + C18O/12CO)
                        = 0.001 x 0.01 / (1.0 + 0.01 + 0.001)
                - VMR_C18O = VMR_12CO x (C18O/12CO) / (12CO/12CO + 13CO/12CO + C18O/12CO)
                        = 0.001 x 0.001 / (1.0 + 0.01 + 0.001)
            + FreeChemistry
                - VMR_13CO = VMR_12CO x (13CO/12CO) / 1.0
                        = 0.001 x 0.01 / 1.0
                - VMR_C18O = VMR_12CO x (C18O/12CO) / 1.0
                        = 0.001 x 0.001 / 1.0

        Args:
            ParamTable (dict): Parameters including isotope ratios.
                See _share_isotope_ratios for parameter names.
        """
        # Step 1: Check if VMRs dictionary exists
        if self.VMRs is None or not isinstance(self.VMRs, dict):
            return
        
        # Step 2: Determine if we need to conserve total VMR
        # For EquilibriumChemistry, we need to normalize by sum of all ratios to conserve total
        # For FreeChemistry, we simply split the main isotopologue (sum_of_ratios = 1)
        conserve_tot_VMR = isinstance(self, EquilibriumChemistry)
        
        # Step 3: Get all isotope ratios from ParamTable
        # This returns dictionaries mapping each isotopologue to its ratio relative to main
        all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios = \
            self._share_isotope_ratios(ParamTable)
        
        VMRs_copy = self.VMRs.copy()
        
        # Step 4: Loop through all species in self.species
        for species_i in self.species:
            
            # Step 4a: Check if this species already has a VMR set or known isotopologue
            VMR_i = VMRs_copy.get(species_i, np.zeros_like(self.pressure))
            if (VMR_i != 0.).any():
                # Already set (e.g., from get_VMRs)
                continue
            
            if species_i not in [*all_CO_ratios, *all_H2O_ratios, *all_CH4_ratios, *all_NH3_ratios, *all_CO2_ratios]:
                # Not a CO, H2O, CH4, NH3, or CO2 isotopologue
                continue
            
            # Step 4b: Initialize variables to find the main isotopologue and ratio
            # We need to find:
            #   - minor_main_ratio_i: the ratio of this isotopologue relative to main
            #   - main_iso_VMR_i: the VMR of the main isotopologue
            #   - sum_of_ratios: normalization factor (1 for free chem, sum of all ratios for eq chem)
            minor_main_ratio_i = None
            main_iso_VMR_i = None
            sum_of_ratios = 1.
            

            iterables = zip(
                [all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios], 
                ['12CO', 'H2O', 'CH4', 'NH3', 'CO2']
            )

            for all_ratios, main_iso_i in iterables:
                minor_main_ratio_i = all_ratios.get(species_i)
                
                if minor_main_ratio_i is None:
                    # This species is not in this group, continue to next group
                    continue
                
                # Read the VMR of the main isotopologue
                # Try multiple keys: species name, pRT_name, or direct lookup
                main_iso_VMR_i = None
                
                # Try 1: direct species name lookup
                if main_iso_i in VMRs_copy:
                    main_iso_VMR_i = VMRs_copy[main_iso_i]
                else:
                    # Try 2: Look up the pRT_name (e.g., '12C-16O') and use that as key
                    main_iso_pRT_name = None
                    if main_iso_i in self.species_info.index:
                        main_iso_pRT_name = self.read_species_info(main_iso_i, 'pRT_name')
                        if main_iso_pRT_name in VMRs_copy:
                            main_iso_VMR_i = VMRs_copy[main_iso_pRT_name]
                    else:
                        # Try 3: Direct lookup using main_iso_i as pRT_name
                        if main_iso_i in VMRs_copy:
                            main_iso_VMR_i = VMRs_copy[main_iso_i]
                
                # calculate the sum of ratios
                sum_of_ratios = 1.
                if conserve_tot_VMR:
                    # To conserve the total VMR
                    sum_of_ratios = sum(all_ratios.values())
                
                # Matching isotope ratio found
                break
            
            # Step 5: Calculate the VMR for this isotopologue
            if main_iso_VMR_i is None:
                # Main isotopologue not set, cannot calculate isotopologue VMR
                continue
            
            if minor_main_ratio_i is None or np.isinf(minor_main_ratio_i):
                # No valid ratio found, skip this isotopologue
                continue
            
            # e.g. 13CO = 12CO * (13CO_ratio) / (sum of all CO ratios)
            # For free chemistry: sum_of_ratios = 1, so 13CO = 12CO * 13CO_ratio
            # For eq chemistry: sum_of_ratios includes all isotopologues, conserving total
            self.VMRs[species_i] = main_iso_VMR_i * minor_main_ratio_i / sum_of_ratios
    
    def _power_law_drop_off(self, VMR, P0, alpha):
        """
        Apply a power-law drop-off to the volume mixing ratio (VMR).
        
        Args:
            VMR (np.ndarray): Volume mixing ratio.
            P0 (float, optional): Reference pressure [bar] where upper regions will have drop-off. If None, no drop-off.
            alpha (float, optional): Power-law exponent. If None, instant drop-off.
        
        Returns:
            np.ndarray: Modified VMR.
        """
        if P0 is None:
            # No drop-off
            return VMR
        
        mask_TOA = (self.pressure < P0)  # Top-of-atmosphere
        if alpha is None:
            VMR[mask_TOA] = 0.  # Instant drop-off
            return VMR
        
        # Power-law drop-off
        VMR[mask_TOA] *= (self.pressure[mask_TOA] / P0) ** alpha
        return VMR
    
    def __call__(self, ParamTable, temperature):
        """
        Evaluate the chemistry model with given parameters and temperature.
        
        Args:
            ParamTable (dict): Parameters for the model.
            temperature (np.ndarray): Temperature profile [K].
        
        Returns:
            dict: Mass fractions dictionary with keys for each species and 'MMW'.
        """
        self.temperature = temperature
        
        # Initialize
        self.mass_fractions = {}
        self.VMRs = {}
        self.MMW = None
        self.CO = None
        self.FeH = None
        
        # Get volume-mixing ratios (implemented by subclasses)
        self.get_VMRs(ParamTable)
        if self.VMRs == -np.inf:
            # Some issue was raised
            self.mass_fractions = -np.inf
            warnings.warn("VMRs are -np.inf, setting mass_fractions to -np.inf. Check the chemistry model subclass.")
            return -np.inf
        
        # Get isotope abundances (split main isotopologues into isotopologues)
        self.get_isotope_VMRs(ParamTable)
        
        # Compute mean molecular weight (if needed), defined in subclass
        get_mmw_method = getattr(self, 'get_MMW', None)
        if get_mmw_method is not None and callable(get_mmw_method):
            try:
                get_mmw_method()
            except (AttributeError, ValueError):
                pass
        if self.MMW is None:
            # Calculate MMW from VMRs
            if self.VMRs is None:
                raise ValueError("VMRs must be set before calculating MMW.")
            MMW = np.zeros(self.n_atm_layers)
            for species_i, VMR_i in self.VMRs.items():
                species_name = None
                for sp in self.species_info.index:
                    if self.read_species_info(sp, 'pRT_name') == species_i:
                        species_name = sp
                        break
                if species_name is None:
                    if species_i in self.species_info.index:
                        species_name = species_i
                    else:
                        continue
                mass_i = self.read_species_info(species_name, 'mass')
                MMW += mass_i * VMR_i
            self.MMW = MMW
        
        if self.VMRs == -np.inf:
            # Some issue was raised
            self.mass_fractions = -np.inf
            warnings.warn("MMW is not set, setting mass_fractions to -np.inf. Check the chemistry model subclass.")
            return -np.inf
        
        # Convert to mass-fractions for pRT
        self.convert_to_MFs()
        
        # Get diagnostics (C/O, Fe/H)
        self.get_diagnostics()
        
        # Remove certain species (if needed)
        self.remove_species()
        
        # Handle H2 line opacity if needed
        if hasattr(self, 'add_H2_line_species') and self.VMRs is not None:
            if 'H2' in self.VMRs and self.mass_fractions is not None:
                self.VMRs['H2_lines'] = self.VMRs['H2'].copy()
                if 'H2' in self.mass_fractions:
                    self.mass_fractions[self.add_H2_line_species] = self.mass_fractions['H2'].copy()
        
        return self.mass_fractions


# ========== Free Chemistry ==========

class FreeChemistry(Chemistry):
    """
    Class for handling free chemistry models.
    In free chemistry, the volume mixing ratios (VMRs) of species are free parameters.
    """
    
    def __init__(self, pressure, line_species, **kwargs):
        """
        Initialize the FreeChemistry class.
        
        Args:
            pressure (np.ndarray): Pressure levels [bar].
            line_species (list): List of line species.
            **kwargs: Additional arguments (e.g., LineOpacity, species_info_path).
        """
        super().__init__(pressure, line_species, **kwargs)
    
    def get_VMRs(self, ParamTable):
        """
        Get volume mixing ratios (VMRs) for the free chemistry model.
        
        Args:
            ParamTable (dict): Parameters for the model.
                - For each species in self.species, can have:
                  - log_<species_name>: log10 of VMR value (e.g., 'log_H2O')
                  - <species_name>_P: Reference pressure for drop-off [bar] (optional)
                  - <species_name>_alpha: Power-law exponent for drop-off (optional)
                - 'C/O': C/O ratio (optional, for diagnostics)
                - 'Fe/H': Fe/H metallicity (optional, for diagnostics)
        """
        # Constant He abundance
        self.VMRs = {'He': 0.15 * np.ones(self.n_atm_layers)}
        
        for species_i in self.species_info.index:
            if species_i in ['H2', 'He']:
                continue  # Set by other VMRs
            
            # Get pRT name for this species
            line_species_i = str(self.read_species_info(species_i, 'pRT_name'))
            
            # Only process if this species (species_i, not pRT_name) is in self.species
            # self.species contains species names (e.g., 'H2O', '12CO'), not pRT_names
            if species_i not in self.species:
                continue
            
            # Read the fitted VMR parameter (use log_<species_name> format)
            # If parameter is missing, VMR will be set to 0
            param_VMR_i = ParamTable.get(f'log_{species_i}', None)
            is_log_value = True
            
            # Try linear value if log not found (for backward compatibility)
            if param_VMR_i is None:
                param_VMR_i = ParamTable.get(species_i, None)
                is_log_value = False
            
            # Convert log value to linear if needed
            if param_VMR_i is not None:
                if is_log_value:
                    param_VMR_i = 10 ** param_VMR_i
            else:
                # If parameter is missing, set VMR to 0
                # This ensures the species is still added to VMRs dict
                param_VMR_i = 0.0
            
            # Expand to all layers
            param_VMR_i = np.atleast_1d(param_VMR_i)
            if len(param_VMR_i) == 1:
                param_VMR_i = param_VMR_i[0] * np.ones(self.n_atm_layers)
            elif len(param_VMR_i) != self.n_atm_layers:
                raise ValueError(f"VMR for {species_i} has wrong length: {len(param_VMR_i)} vs {self.n_atm_layers}")
            
            # Apply power-law drop-off if parameters are provided
            # Use species name (species_i) as the key in VMRs dict (consistent with get_isotope_VMRs)
            # pRT_name will be used later in convert_to_MFs when creating mass_fractions
            self.VMRs[species_i] = self._power_law_drop_off(
                param_VMR_i.copy(),
                P0=ParamTable.get(f'{species_i}_P'),
                alpha=ParamTable.get(f'{species_i}_alpha'),
            )
        
        # Overwrite C/O ratio and Fe/H with constants (if given)
        self.CO = ParamTable.get('C/O', None)
        self.FeH = ParamTable.get('Fe/H', None)
    
    def get_H2(self):
        """
        Calculate the H2 abundance as the remainder of the total VMR.
        """
        if self.VMRs is None or not isinstance(self.VMRs, dict):
            self.VMRs = {}
        # H2 abundance is the remainder
        VMR_wo_H2 = np.sum([VMR_i for VMR_i in self.VMRs.values()], axis=0)
        self.VMRs['H2'] = 1 - VMR_wo_H2
        
        if isinstance(self.VMRs, dict) and 'H2' in self.VMRs:
            if (self.VMRs['H2'] < 0).any():
                # Other species are too abundant
                self.VMRs = -np.inf
    
    def get_MMW(self):
        """
        Calculate the mean molecular weight (MMW) from the VMRs.
        """
        if self.VMRs is None or not isinstance(self.VMRs, dict):
            raise ValueError("VMRs must be set before calculating MMW.")
        # Get mean-molecular weight from free-chem VMRs
        # VMRs dict uses species name as key (e.g., 'H2O', '12CO', 'H2', 'He')
        MMW = np.zeros(self.n_atm_layers)
        for species_i, VMR_i in self.VMRs.items():
            # species_i is already a species name (from species_info.index)
            if species_i not in self.species_info.index:
                # Skip if not found (should not happen, but for safety)
                continue
            mass_i = self.read_species_info(species_i, 'mass')
            MMW += mass_i * VMR_i
        
        self.MMW = MMW
    
    def __call__(self, ParamTable, temperature):
        """
        Evaluate the free chemistry model.
        
        Args:
            ParamTable (dict): Parameters for the model.
            temperature (np.ndarray): Temperature profile [K].
        
        Returns:
            dict: Mass fractions dictionary.
        """
        # Call parent __call__ but add get_H2 step
        result = super().__call__(ParamTable, temperature)
        if result == -np.inf:
            return -np.inf
        
        # Compute H2 abundance last (after other VMRs are set)
        self.get_H2()
        if self.VMRs == -np.inf:
            self.mass_fractions = -np.inf
            return -np.inf
        
        # Recalculate MMW with H2
        self.get_MMW()
        
        # Reconvert to mass fractions
        self.convert_to_MFs()
        
        return self.mass_fractions


# ========== Equilibrium Chemistry ==========

class EquilibriumChemistry(Chemistry):
    """
    Class for handling equilibrium chemistry models.
    This is a placeholder for future implementation.
    In equilibrium chemistry, abundances are calculated from elemental ratios (C/O, Fe/H, etc.)
    and temperature-pressure conditions.
    """
    
    def __init__(self, pressure, line_species, **kwargs):
        """
        Initialize the EquilibriumChemistry class.
        
        Args:
            pressure (np.ndarray): Pressure levels [bar].
            line_species (list): List of line species.
            **kwargs: Additional arguments (e.g., LineOpacity, species_info_path).
        """
        super().__init__(pressure, line_species, **kwargs)
        
        # Placeholder for quench settings (for future implementation)
        self.quench_settings = {}
    
    def get_VMRs(self, ParamTable):
        """
        Get volume mixing ratios (VMRs) for the equilibrium chemistry model.
        
        This is a placeholder implementation. Future versions should:
        - Use C/O, Fe/H, N/O ratios from ParamTable
        - Calculate equilibrium abundances from temperature-pressure profile
        - Support quenching at specified pressures
        
        Args:
            ParamTable (dict): Parameters for the model.
                - 'C/O': C/O ratio (required)
                - 'Fe/H': Fe/H metallicity (required)
                - 'N/O': N/O ratio (optional)
        """
        raise NotImplementedError(
            "EquilibriumChemistry is not yet implemented. "
            "Please use chem_mode='free' for now, or implement equilibrium chemistry calculation."
        )
    
    def quench_VMRs(self, ParamTable):
        """
        Quench the volume mixing ratios (VMRs) based on quench pressures.
        Placeholder for future implementation.
        
        Args:
            ParamTable (dict): Parameters for the model.
        """
        raise NotImplementedError("Quenching not yet implemented for EquilibriumChemistry.")


# # ========== Test/Debug section ==========
# # Run "python -m atmosphere.chemistry" in src/ to test chemistry functionality

# if __name__ == "__main__":
#     import os
#     import sys
    
#     # Add parent directory to path to import modules
#     sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
#     print("=" * 60)
#     print("Testing Chemistry Classes")
#     print("=" * 60)
    
#     # Create test pressure grid
#     pressure = np.logspace(-5, 2, 70)  # 70 layers from 1e-5 to 100 bar
#     line_species = ['12CO', 'H2O', 'CH4', 'NH3']
    
#     # Test FreeChemistry
#     print("\n" + "-" * 60)
#     print("Test 1: FreeChemistry")
#     print("-" * 60)
    
#     try:
#         chem = get_class(pressure, line_species, chem_mode='free')
        
#         # Create test ParamTable
#         ParamTable = {
#             '12CO': 1e-3,  # 0.1% CO
#             'H2O': 1e-4,   # 0.01% H2O
#             'CH4': 1e-5,   # 0.001% CH4
#         }
#         temperature = 1500 * np.ones_like(pressure)  # Constant temperature
        
#         # Call chemistry
#         mass_fractions = chem(ParamTable, temperature)
        
#         print(f"✓ FreeChemistry created successfully")
#         print(f"  - Number of layers: {len(chem.pressure)}")
#         print(f"  - Pressure range: {chem.pressure.min():.2e} - {chem.pressure.max():.2e} bar")
#         print(f"  - Species: {list(chem.species)}")
#         if isinstance(mass_fractions, dict):
#             print(f"  - Mass fractions keys: {list(mass_fractions.keys())}")
#             if 'MMW' in mass_fractions:
#                 mfw = mass_fractions['MMW']
#                 if isinstance(mfw, np.ndarray):
#                     print(f"  - MMW range: {mfw.min():.2f} - {mfw.max():.2f}")
#         if chem.CO is not None:
#             print(f"  - C/O ratio: {chem.CO:.4f}")
#         if chem.FeH is not None:
#             print(f"  - Fe/H: {chem.FeH:.4f}")
        
#     except Exception as e:
#         print(f"✗ Error in FreeChemistry test: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # Test EquilibriumChemistry (should raise NotImplementedError)
#     print("\n" + "-" * 60)
#     print("Test 2: EquilibriumChemistry (placeholder)")
#     print("-" * 60)
    
#     try:
#         chem_eq = get_class(pressure, line_species, chem_mode='equilibrium')
#         print("⚠ EquilibriumChemistry class created, but get_VMRs not implemented")
#     except NotImplementedError as e:
#         print(f"✓ EquilibriumChemistry correctly raises NotImplementedError: {e}")
#     except Exception as e:
#         print(f"✗ Unexpected error: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n" + "=" * 60)
#     print("Test completed!")
#     print("=" * 60)
#     print("\nTo run this test:")
#     print("  cd src/")
#     print("  python -m atmosphere.chemistry")

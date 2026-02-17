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

from ast import Param
import numpy as np
import pandas as pd
import pathlib
import warnings
from typing import Any, Optional
import scipy.constants as sc

# Get directory path for loading species info
directory_path = pathlib.Path(__file__).parent.resolve()
# /retrieval_base/src/atmosphere

from core.paths import SRC_DIR

# Import pRT equilibrium chemistry table at module level (singleton, loaded once)
# This avoids repeated imports and allows checking _loaded flag before calling load()
try:
    from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table  # type: ignore
except ImportError:
    pre_calculated_equilibrium_chemistry_table = None


# ========== Factory function ==========

def get_class(pressure, line_species, **kwargs):
    """
    Factory function to determine and return the appropriate Chemistry class based on chem_mode.
    
    Args:
        pressure (np.ndarray): Atmospheric pressure grid [bar].
        line_species (list): List of line species used in opacity calculations.
        chem_mode (str, optional): Chemistry mode.
            - 'free': Free chemistry (abundances are free parameters)
            - 'equilibrium': Equilibrium chemistry
            - 'fastchem_grid': Fastchem chemistry with grid interpolation
            - 'fastchem_live': Live Fastchem calculation
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
    elif chem_mode == 'fastchem_grid':
        return FastChemistry_Grid(pressure, line_species, **kwargs)
    elif chem_mode == 'fastchem_live':
        return FastChemistry_Live(pressure, line_species, **kwargs)
    else:
        raise ValueError(f"Chemistry mode '{chem_mode}' not implemented. "
                        f"Available modes: 'free', 'equilibrium', 'fastchem_grid', 'fastchem_live'.")


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
                    # Species not in species_info
                    # For FastChemistry_Live, try to get mass and pRT_name from FastChem
                    if isinstance(self, FastChemistry_Live):
                        pRT_name, mass_i = self._get_species_mass_and_prt_name(species_i)
                        if pRT_name is not None and mass_i is not None:
                            # Found in FastChem or known species
                            self.mass_fractions[str(pRT_name)] = VMR_i * mass_i / self.MMW
                        # else: skip (species not found in FastChem either)
                    # For other chemistry modes, skip (original behavior)
                    continue
            else:
                species_name = species_i
            
            # Get pRT_name and mass for this species
            line_species_i = self.read_species_info(species_name, 'pRT_name')
            mass_i = self.read_species_info(species_name, 'mass')
            
            # Use pRT_name as key in mass_fractions (for pRT compatibility)
            self.mass_fractions[str(line_species_i)] = VMR_i * mass_i / self.MMW
    
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
                - Elemental isotope ratios: '12C/13C', '12/13C_ratio', '16/18O_ratio', 'H/D_ratio', '14/15N_ratio'
                Note: '12C/13C' and '12/13C_ratio' are equivalent (both represent 12C/13C ratio)
        
        Returns:
            tuple: (all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios)
                Each is a dict mapping species names to their ratios relative to the main isotopologue.
                Example: all_CO_ratios = {'12CO': 1.0, '13CO': 0.01, 'C18O': 0.001, ...}
        """
        # Get 12C/13C ratio (support both '12C/13C' and '12/13C_ratio' for backward compatibility)
        C12_C13_ratio = ParamTable.get('12C/13C', ParamTable.get('12/13C_ratio', None))
        
        # First looks for isotopologue ratio, then for isotope ratio, otherwise sets abundance to 0
        all_CO_ratios = {
            '12CO': 1., 
            '13CO': 1./ParamTable.get('13CO_ratio', C12_C13_ratio if C12_C13_ratio is not None else np.inf), 
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
            '13CH4': 1./ParamTable.get('13CH4_ratio', C12_C13_ratio if C12_C13_ratio is not None else np.inf), 
            'CH3D':  1./ParamTable.get('CH3D_ratio', ParamTable.get('H/D_ratio', np.inf)/4.),
        }
        all_NH3_ratios = {
            'NH3':   1., 
            '15NH3': 1./ParamTable.get('15NH3_ratio', ParamTable.get('14/15N_ratio', np.inf)), 
        }
        all_CO2_ratios = {
            'CO2':     1., 
            '13CO2':   1./ParamTable.get('13CO2_ratio', C12_C13_ratio if C12_C13_ratio is not None else np.inf), 
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
        
        # Quench eq-chem abundances (if quench_VMRs method exists)
        if hasattr(self, 'quench_VMRs'):
            getattr(self, 'quench_VMRs')(ParamTable)  # type: ignore
            if self.VMRs == -np.inf:
                self.mass_fractions = -np.inf
                warnings.warn("VMRs are -np.inf after quenching, setting mass_fractions to -np.inf.")
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
    Class for handling equilibrium chemistry models using pRT interpolation tables.
    In equilibrium chemistry, abundances are calculated from elemental ratios (C/O, Fe/H, etc.)
    and temperature-pressure conditions using petitRADTRANS equilibrium chemistry tables.
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
        
        # Load the interpolation tables from pRT v3
        # Use the global singleton instance (imported at module level)
        if pre_calculated_equilibrium_chemistry_table is None:
            raise ImportError(
                "petitRADTRANS.chemistry.pre_calculated_chemistry module not found. "
                "Please ensure petitRADTRANS v3 is properly installed."
            )
        
        self.interp_tables = pre_calculated_equilibrium_chemistry_table
        
        # Only load if not already loaded (avoid repeated loading)
        # The table is a singleton, so loading once is sufficient for all instances
        if not self.interp_tables._loaded:
            try:
                # Suppress pRT's loading message by redirecting stdout temporarily
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    self.interp_tables.load()
                finally:
                    sys.stdout = old_stdout
            except (OSError, IOError) as e:
                # Check if file exists and get its size for diagnostic info
                import os
                file_path = self.interp_tables.get_default_file()
                file_exists = os.path.exists(file_path)
                file_size = os.path.getsize(file_path) if file_exists else 0
                
                error_msg = (
                    f"Failed to load equilibrium chemistry table.\n"
                    f"Error: {e}\n"
                    f"File path: {file_path}\n"
                    f"File exists: {file_exists}\n"
                )
                
                if file_exists:
                    error_msg += (
                        f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)\n"
                        f"The HDF5 file appears to be truncated or corrupted.\n"
                        f"Expected size: ~1.3 GB (based on error message)\n"
                        f"Actual size: {file_size / 1024 / 1024:.1f} MB\n"
                        f"Please re-download or regenerate the file."
                    )
                else:
                    error_msg += "File does not exist. Please check the path or download the file."
                
                raise RuntimeError(error_msg) from e
        
        # Species to quench per system
        self.quench_settings = {
            'CO_CH4': [['12CO', 'CH4', 'H2O'], None], 
            'N2_NH3': [['N2', 'NH3'], None], 
            'HCN': [['HCN'], None], 
            'CO2': [['CO2'], None],
        }
    
    def get_VMRs(self, ParamTable):
        """
        Get volume mixing ratios (VMRs) using pRT v3 interpolation tables.
        
        Args:
            ParamTable (dict): Parameters for the model.
                - 'C/O': C/O ratio (required)
                - 'N/O': N/O ratio (optional)
                - 'Fe/H': Fe/H metallicity (required, in log10 units for pRT)
                - '12C/13C': 12C/13C isotope ratio (optional, for isotope calculations)
                - '16O/18O': 16/18O isotope ratio (optional, for isotope calculations)
                - For each species, can optionally override with free parameter:
                  - <species_name>: VMR value (if provided, overrides eq-chem)
        """
        # Update the parameters
        self.CO = ParamTable.get('C/O')
        self.FeH = ParamTable.get('Fe/H')
        self.NO = ParamTable.get('N/O', None)
        self.C12_C13_ratio = ParamTable.get('12C/13C', None)  # Store for isotope calculations
        self.O16_O18_ratio = ParamTable.get('16O/18O', None)  # Store for isotope calculations
        
        if self.CO is None or self.FeH is None:
            raise ValueError("EquilibriumChemistry requires 'C/O' and 'Fe/H' parameters in ParamTable.")
        
        # Retrieve the mass fractions from the pRT v3 eq-chem table
        # Note: pRT v3 uses interpolate_mass_fractions with different signature
        pm_mass_fractions = self.interp_tables.interpolate_mass_fractions(
            co_ratios=self.CO * np.ones(self.n_atm_layers),
            log10_metallicities=self.FeH * np.ones(self.n_atm_layers),
            temperatures=self.temperature,
            pressures=self.pressure,
            carbon_pressure_quench=None,  # We handle quenching separately
            full=True  # Get MMW as well
        )
        
        # Extract mass fractions and MMW
        # If full=True, returns (mass_fractions, mean_molar_masses, nabla_adiabatic)
        if isinstance(pm_mass_fractions, tuple):
            mass_fractions_dict, self.MMW, _ = pm_mass_fractions
        else:
            # If full=False, only mass fractions returned, need to calculate MMW separately
            mass_fractions_dict = pm_mass_fractions
            self.MMW = None  # Will be calculated later if needed
        
        # Fill in the VMR dictionary
        self.VMRs = {}
        
        # If self.species is empty (no line_species specified), get all available species from pRT table
        # pRT eq-chem table contains: H2, He, CO, H2O, HCN, C2H2, CH4, PH3, CO2, NH3, H2S, VO, TiO, Na, K, SiO, e-, H-, H, FeH
        species_to_process = self.species.copy() if len(self.species) > 0 else []
        if len(species_to_process) == 0:
            # Get all available species from pRT mass fractions dict by matching with species_info
            # Use self.species_info (inherited from Chemistry base class) to find matching species
            # pRT eq-chem table uses simplified names (e.g., 'CO', 'H2O', 'CH4')
            # while species_info.csv has full names (e.g., '12CO', 'H2O', 'CH4') and pRT_name (e.g., '12C-16O', '1H2-16O', '12C-1H4')
            
            # Mapping from pRT table keys to species_info names (for special cases)
            # Most species match directly by name, but CO is special (pRT uses 'CO', species_info has '12CO')
            pRT_key_to_species_name = {
                'CO': '12CO',  # pRT uses 'CO' but species_info has '12CO'
            }
            
            # Iterate through all keys in pRT mass fractions dict and match with species_info
            for pRT_key in mass_fractions_dict.keys():
                # Skip H2 and He (handled separately)
                if pRT_key in ['H2', 'He']:
                    continue
                
                matched_species = None
                
                # Method 1: Check special mapping (e.g., 'CO' -> '12CO')
                if pRT_key in pRT_key_to_species_name:
                    species_name = pRT_key_to_species_name[pRT_key]
                    if species_name in self.species_info.index:
                        matched_species = species_name
                
                # Method 2: Direct match by species name (e.g., 'H2O' in pRT table matches 'H2O' in species_info)
                if matched_species is None and pRT_key in self.species_info.index:
                    matched_species = pRT_key
                
                # Method 3: Try to find by searching through species_info for matching pRT_name
                # This handles cases where pRT table key doesn't match species name directly
                if matched_species is None:
                    for species_name in self.species_info.index:
                        # Skip if already processed
                        if species_name in species_to_process:
                            continue
                        
                        # Get pRT_name from species_info
                        pRT_name = self.read_species_info(species_name, 'pRT_name')
                        if not isinstance(pRT_name, str):
                            continue
                        
                        # For CO, be careful: 'CO' appears in 'CO2', so only match '12CO'
                        if pRT_key == 'CO' and species_name == '12CO':
                            matched_species = species_name
                            break
                        # For other species, check if pRT_key is contained in pRT_name
                        # (e.g., 'H2O' should match '1H2-16O')
                        elif pRT_key != 'CO' and pRT_key in pRT_name:
                            matched_species = species_name
                            break
                
                # Add matched species to process list
                if matched_species is not None and matched_species not in species_to_process:
                    species_to_process.append(matched_species)
            
            if len(species_to_process) == 0:
                warnings.warn(
                    "No species found in equilibrium chemistry. "
                    "Please specify line_species in chemistry_kwargs or add log_* parameters to config."
                )
        
        for species_i in species_to_process:
            # Check if a free-chemistry VMR is provided (overrides eq-chem)
            param_VMR_i = ParamTable.get(species_i, None)
            if param_VMR_i is not None:
                # Expand to all layers
                param_VMR_i = np.atleast_1d(param_VMR_i)
                if len(param_VMR_i) == 1:
                    param_VMR_i = param_VMR_i[0] * np.ones(self.n_atm_layers)
                elif len(param_VMR_i) != self.n_atm_layers:
                    raise ValueError(f"VMR for {species_i} has wrong length: {len(param_VMR_i)} vs {self.n_atm_layers}")
                
                # Apply power-law drop-off if parameters are provided
                self.VMRs[species_i] = self._power_law_drop_off(
                    param_VMR_i.copy(),
                    P0=ParamTable.get(f'{species_i}_P'),
                    alpha=ParamTable.get(f'{species_i}_alpha'),
                )
                continue  # Skip eq-chem calculation for this species
            
            # Search a different key for 12CO (pRT uses 'CO' instead of '12CO')
            key_i = species_i
            if species_i == '12CO':
                key_i = 'CO'
            
            # Check if species exists in pRT mass fractions
            if key_i not in mass_fractions_dict:
                # Species not in pRT table, skip
                continue
            
            # Convert mass fraction to VMR
            mass_i = self.read_species_info(species_i, 'mass')
            if self.MMW is None:
                # Calculate MMW from mass fractions if not provided
                raise ValueError("MMW must be calculated when using equilibrium chemistry.")
            VMR_i = mass_fractions_dict[key_i] * self.MMW / mass_i
            
            if (VMR_i == 0.).any():
                # Some layers have zero abundance, which might cause issues
                # Set to a very small value to avoid numerical issues
                VMR_i[VMR_i == 0.] = 1e-100
            
            self.VMRs[species_i] = VMR_i
        
        # Ensure H2 and He are set (if not already)
        if 'H2' not in self.VMRs:
            # H2 abundance is typically the remainder
            VMR_wo_H2 = np.sum([VMR_i for VMR_i in self.VMRs.values()], axis=0)
            self.VMRs['H2'] = np.clip(1 - VMR_wo_H2, 0, 1)
        
        if 'He' not in self.VMRs:
            # Default He abundance
            self.VMRs['He'] = 0.15 * np.ones(self.n_atm_layers)
    
    def get_P_quench_from_Kzz(self, ParamTable, alpha=1):
        """
        Calculate the quench pressure from the eddy diffusion coefficient (Kzz).
        
        Args:
            ParamTable (dict): Parameters including Kzz.
                - 'log_Kzz_chem': log10 of eddy diffusion coefficient [cm²/s] (required)
                - 'g': Surface gravity [cm/s²] (or 'log_g' for log10 of gravity in cgs)
            alpha (float, optional): Mixing length factor. Defaults to 1.
        """
        def interpolate_for_P_quench(log_t_mix, log_t_chem):
            """Interpolate the quench pressure based on mixing and chemical timescales."""
            idx_well_mixed = np.argwhere(log_t_mix < log_t_chem).flatten()
            if len(idx_well_mixed) == 0:
                # All layers are in chemical equilibrium
                return self.pressure.min()
            
            # Lowest layer that is well-mixed
            idx_lowest_mixed_layer = idx_well_mixed[-1]
            idx_highest_equilibrium_layer = idx_lowest_mixed_layer + 1

            if idx_highest_equilibrium_layer >= len(self.pressure):
                # All layers are well-mixed
                return self.pressure.max()
            
            # Quenching happens between these two layers
            indices = [idx_lowest_mixed_layer, idx_highest_equilibrium_layer]
            return 10**np.interp(
                0, xp=log_t_mix[indices]-log_t_chem[indices], fp=np.log10(self.pressure[indices])
            )

        # Get gravity (handle both 'g' and 'log_g' parameters)
        g = ParamTable.get('g')
        if g is None:
            log_g = ParamTable.get('log_g')
            if log_g is not None:
                # Convert log_g (typically in cgs) to g in cm/s²
                g = 10 ** log_g
            else:
                raise ValueError("Either 'g' or 'log_g' must be provided in ParamTable for quenching calculation.")
        
        # Metallicity
        m = 10**self.FeH if self.FeH is not None else 1.0

        # Scale height at each layer
        # sc.k is in J/K, convert to erg/K (1 J = 1e7 erg)
        # sc.m_u (atomic mass unit) is in kg, convert to g (1 kg = 1e3 g)
        if self.temperature is None or self.MMW is None:
            raise ValueError("temperature and MMW must be set before calculating quench pressure.")
        kB = sc.k * 1e7  # J/K -> erg/K
        amu = sc.m_u * 1e3  # kg -> g (sc.m_u is atomic mass unit)
        H = kB * self.temperature / (self.MMW * amu * g)

        # Mixing length/time-scales
        L = alpha * H
        
        # Get log_Kzz_chem and convert to linear value
        log_Kzz_chem = ParamTable.get('log_Kzz_chem')
        if log_Kzz_chem is None:
            return  # Cannot calculate without log_Kzz_chem
        
        # Convert from log10 to linear value
        Kzz_chem = 10 ** log_Kzz_chem
        
        t_mix = L**2 / Kzz_chem
        log_t_mix = np.log10(t_mix)

        # Ignore overflow warnings
        with np.errstate(over='ignore'):
            # Chemical timescales from Zahnle & Marley (2014)
            
            # Calculate CO-CH4 chemical timescale (needed for CO_CH4 and CO2)
            inv_t_q1 = 1.0e6/1.5 * self.pressure * m**0.7 * np.exp(-42000/self.temperature)
            inv_t_q2 = 1/40 * self.pressure**2 * np.exp(-25000/self.temperature)
            log_t_CO_CH4 = -1 * np.log10(inv_t_q1 + inv_t_q2)  # equation

            key_q = 'CO_CH4'
            # Check for log_P_quench parameter (new format) or P_quench (backward compatibility)
            log_P_quench = ParamTable.get(f'log_P_quench_{key_q}')
            P_quench = ParamTable.get(f'P_quench_{key_q}')
            if (log_P_quench is None) and (P_quench is None) and (key_q in self.quench_settings):
                self.quench_settings[key_q][-1] = interpolate_for_P_quench(log_t_mix, log_t_CO_CH4)
            
            key_q = 'N2_NH3'
            log_P_quench = ParamTable.get(f'log_P_quench_{key_q}')
            P_quench = ParamTable.get(f'P_quench_{key_q}')
            if (log_P_quench is None) and (P_quench is None) and (key_q in self.quench_settings):
                # t_NH3 = 1.0e-7 * self.pressure**(-1) * np.exp(52000/self.temperature)
                log_t_NH3 = (-7.0 - np.log10(self.pressure) + 52000/self.temperature*np.log10(np.e))  # equation 32
                self.quench_settings[key_q][-1] = interpolate_for_P_quench(log_t_mix, log_t_NH3)

            key_q = 'HCN'
            log_P_quench = ParamTable.get(f'log_P_quench_{key_q}')
            P_quench = ParamTable.get(f'P_quench_{key_q}')
            if (log_P_quench is None) and (P_quench is None) and (key_q in self.quench_settings):
                # t_HCN = 1.5e-4 * self.pressure**(-1) * m**(-0.7) * np.exp(36000/self.temperature)
                log_t_HCN = (np.log10(1.5e-4) - np.log10(self.pressure*m**0.7) + 36000/self.temperature*np.log10(np.e))  # equation 33
                self.quench_settings[key_q][-1] = interpolate_for_P_quench(log_t_mix, log_t_HCN)

            key_q = 'CO2'
            log_P_quench = ParamTable.get(f'log_P_quench_{key_q}')
            P_quench = ParamTable.get(f'P_quench_{key_q}')
            if (log_P_quench is None) and (P_quench is None) and (key_q in self.quench_settings):
                # Approximate CO2 quenching as the same level as CO-CH4, since CO2 remains in equilibrium 
                # with the quenched (i.e. not decreasing) CO abundance
                log_t_CO2 = log_t_CO_CH4
                self.quench_settings[key_q][-1] = interpolate_for_P_quench(log_t_mix, log_t_CO2)
    
    def quench_VMRs(self, ParamTable):
        """
        Quench the volume mixing ratios (VMRs) based on the quench settings.
        
        Args:
            ParamTable (dict): Parameters for the model.
                - 'log_P_quench_<key>': log10 of quench pressure [bar] for each system (optional, preferred)
                - 'P_quench_<key>': Quench pressure [bar] for each system (optional, backward compatibility)
                - 'log_Kzz_chem': log10 of eddy diffusion coefficient [cm²/s] (optional, for calculating quench pressure)
        """
        # Update the parameters
        for key_q in list(self.quench_settings):
            # Check for log_P_quench parameter (new format) or P_quench (backward compatibility)
            log_P_quench = ParamTable.get(f'log_P_quench_{key_q}')
            if log_P_quench is not None:
                # Convert from log10 to linear pressure [bar]
                self.quench_settings[key_q][-1] = 10 ** log_P_quench
            else:
                # Backward compatibility: try P_quench (linear value)
                P_quench = ParamTable.get(f'P_quench_{key_q}')
                self.quench_settings[key_q][-1] = P_quench

        # Calculate quench pressure from Kzz if provided
        if ParamTable.get('log_Kzz_chem') is not None:
            self.get_P_quench_from_Kzz(ParamTable)

        log_P = np.log10(self.pressure)  # Take log for convenience
        for species_q, P_q in self.quench_settings.values():

            if P_q is None:  # No quench-pressure given for this species_q
                continue
            
            mask_TOA = (self.pressure < P_q)  # Top of the atmosphere

            for species_i in species_q:  # all of the species in this series
                if species_i not in self.species:  # skip if not in self.species
                    continue
                
                if species_i not in self.VMRs:  # skip if not in VMRs dict
                    continue
                
                VMR_i = np.clip(self.VMRs[species_i], 1e-100, None)  # avoid negative values
                log_VMR_i = np.log10(VMR_i)
                
                # Interpolate the VMR to the quench pressure, and set the upper VMR as it
                log_VMR_i[mask_TOA] = np.interp(np.log10(P_q), xp=log_P, fp=log_VMR_i)
                self.VMRs[species_i] = 10**log_VMR_i

# ========== FastChemistry (FastChem real-time calculation) ==========

class FastChemistry_Live(EquilibriumChemistry):
    """
    Class for handling fast chemistry models using the FastChem library.
    Uses FastChem for real-time equilibrium chemistry calculations.
    """
    
    def __init__(self, pressure, line_species, **kwargs):
        """
        Initialize the FastChemistry class.
        
        Args:
            pressure (np.ndarray): Pressure levels [bar].
            line_species (list): List of line species.
            **kwargs: Additional arguments:
                - abundance_file (str): Path to element abundance file for FastChem
                - gas_data_file (str): Path to gas data file for FastChem
                - cond_data_file (str, optional): Path to condensation data file. Defaults to 'none'.
                - use_eq_cond (bool, optional): Use equilibrium condensation. Defaults to True.
                - use_rainout_cond (bool, optional): Use rainout condensation. Defaults to False.
                - min_temperature (float, optional): Minimum temperature for FastChem [K]. Defaults to 500.0.
                - save_species (list, optional): List of species to save in VMR dictionary (species_info names).
                    If None, saves all FastChem-calculated species. If provided, only saves these species.
                    Species names will be converted to Hill notation if available in species_info.csv.
                    Example: ['12CO', 'H2O', 'Na', 'Ca', '13CO', 'C18O']
        """
        super().__init__(pressure, line_species, **kwargs)
        
        # FastChem input files
        self.abundance_file = kwargs.get('abundance_file')
        self.gas_data_file = kwargs.get('gas_data_file')
        self.cond_data_file = kwargs.get('cond_data_file', 'none')
        
        # Condensation settings
        # If cond_data_file is 'none', disable condensation automatically
        if self.cond_data_file == 'none' or self.cond_data_file is None:
            self.use_eq_cond = False
            self.use_rainout_cond = False
        else:
            self.use_eq_cond = kwargs.get('use_eq_cond', True)
            self.use_rainout_cond = kwargs.get('use_rainout_cond', False)
        
        # Minimum temperature for FastChem convergence
        self.min_temperature = kwargs.get('min_temperature', 500.0)
        
        # Species to save in VMR dictionary (species_info names, e.g., '12CO', 'H2O', '13CO')
        # If None, save all FastChem-calculated species
        # If provided, only save these species (will be converted to Hill notation if available)
        self.save_species = kwargs.get('save_species', None)
        
        # FastChem object will be created lazily in _get_fastchem_object()
        self.fastchem = None
        self.input = None
        self.output = None
        self.gas_species_tot = None
        
        # Element indices and solar abundances will be set after FastChem object is created
        self.idx = None
        self.metal_idx = None
        self.solar_abund = None
        self.solar_CO = None
        self.solar_NO = None
        self.solar_FeH = 0.0
        
        # Store Hill notation for each species (for FastChem lookup)
        self._init_hill_notations()
    
    def _init_hill_notations(self):
        """
        Initialize Hill notation mapping for species.
        FastChem uses Hill notation (e.g., 'C1O1' for CO, 'H2O1' for H2O).
        """
        self.hill = {}
        # Special mappings for atoms that use diatomic notation in FastChem
        # FastChem uses Na2, Ca2 for atomic Na and Ca (diatomic form)
        atom_to_diatomic = {
            'Na': 'Na2',
            'Ca': 'Ca2',
        }
        
        for species_i in self.species:
            # Check for special atom mappings first
            if species_i in atom_to_diatomic:
                self.hill[species_i] = atom_to_diatomic[species_i]
                continue
            
            # Try to get Hill notation from species_info
            if 'Hill_notation' in self.species_info.columns:
                hill_i = self.species_info.loc[species_i, 'Hill_notation']
                if pd.notna(hill_i) and hill_i != '':
                    # For atoms (Na, Ca), FastChem uses diatomic form (Na2, Ca2)
                    if hill_i == 'Na':
                        self.hill[species_i] = 'Na2'
                    elif hill_i == 'Ca':
                        self.hill[species_i] = 'Ca2'
                    else:
                        self.hill[species_i] = str(hill_i)
                    continue
            
            # If no Hill_notation column, try to convert from pRT_name or species name
            # This is a fallback - ideally species_info should have Hill_notation column
            hill_i = self._convert_to_hill_notation(species_i)
            if hill_i:
                self.hill[species_i] = hill_i
            else:
                # If conversion fails, store None (will skip this species in FastChem)
                self.hill[species_i] = None
    
    def _convert_to_hill_notation(self, species_name):
        """
        Convert species name to Hill notation (fallback method).
        This is a simple conversion - ideally species_info.csv should have Hill_notation column.
        
        Args:
            species_name (str): Species name (e.g., 'H2O', '12CO', 'CH4')
        
        Returns:
            str or None: Hill notation (e.g., 'H2O1', 'C1O1', 'C1H4') or None if conversion fails
        """
        # Common mappings (this is a fallback - should use species_info.csv with Hill_notation column)
        common_mappings = {
            'H2': 'H2',
            'He': 'He',
            'H2O': 'H2O1',
            '12CO': 'C1O1',
            'CO': 'C1O1',
            'CH4': 'C1H4',
            'NH3': 'H3N1',
            'CO2': 'C1O2',
            'HCN': 'C1H1N1',
            'C2H2': 'C2H2',
            'TiO': 'Ti1O1',
            'VO': 'V1O1',
            'Na': 'Na',
            'K': 'K',
            'FeH': 'Fe1H1',
        }
        
        if species_name in common_mappings:
            return common_mappings[species_name]
        
        # Try to infer from species name (very basic)
        # This is not comprehensive - should use species_info.csv
        return None
    
    def _hill_to_species_name(self, hill_notation):
        """
        Convert FastChem Hill notation to species_info name.
        
        Args:
            hill_notation (str): FastChem Hill notation (e.g., 'C1O1', 'H2O1', 'Na')
        
        Returns:
            str or None: Species name from species_info, or None if not found
        """
        # First check if hill_notation directly matches a species name
        if hill_notation in self.species_info.index:
            return hill_notation
        
        # Check if any species has this Hill notation
        if 'Hill_notation' in self.species_info.columns:
            for species_name in self.species_info.index:
                hill_i = self.species_info.loc[species_name, 'Hill_notation']
                if pd.notna(hill_i) and str(hill_i) == hill_notation:
                    return species_name
        
        # Try reverse mapping from common conversions
        # This handles cases like 'C1O1' -> '12CO', 'H2O1' -> 'H2O'
        reverse_mappings = {
            'C1O1': '12CO',
            'H2O1': 'H2O',
            'C1H4': 'CH4',
            'H3N1': 'NH3',
            'C1O2': 'CO2',
            'C1H1N1': 'HCN',
            'C2H2': 'C2H2',
            'Ti1O1': 'TiO',
            'V1O1': 'VO',
            'Fe1H1': 'FeH',
            'Na2': 'Na',  # FastChem uses Na2 for atomic Na
            'Ca2': 'Ca',  # FastChem uses Ca2 for atomic Ca
            'Na': 'Na',
            'Ca': 'Ca',
            'K': 'K',
            'H': 'H',
            'He': 'He',
            'H2': 'H2',
            'e-': 'e-',
            'H1-': 'H-',
            'F1H1': 'HF',
            'H1O1': 'OH',
            'C1N1': 'CN',
            'O1Ti1': 'TiO',  # Alternative notation
        }
        
        if hill_notation in reverse_mappings:
            species_name = reverse_mappings[hill_notation]
            if species_name in self.species_info.index:
                return species_name
        
        # If not found, return None (will be handled in get_VMRs)
        return None
    
    def _get_hill_notation_from_species_name(self, species_name):
        """
        Get Hill notation for a species_info name.
        
        Args:
            species_name (str): Species name from species_info (e.g., '12CO', 'H2O')
        
        Returns:
            str or None: Hill notation if found, None otherwise
        """
        # Check if species is in species_info
        if species_name not in self.species_info.index:
            return None
        
        # Try to get Hill notation from species_info
        if 'Hill_notation' in self.species_info.columns:
            hill_i = self.species_info.loc[species_name, 'Hill_notation']
            if pd.notna(hill_i) and hill_i != '':
                hill_i = str(hill_i)
                # Handle special cases for atoms
                if hill_i == 'Na':
                    return 'Na2'
                elif hill_i == 'Ca':
                    return 'Ca2'
                else:
                    return hill_i
        
        # Try conversion
        hill_i = self._convert_to_hill_notation(species_name)
        if hill_i:
            return hill_i
        
        # If no Hill notation found, return None (will use original name)
        return None
    
    def _contains_isotope_marker(self, species_name):
        """
        Check if species name contains isotope markers.
        
        Args:
            species_name (str): Species name to check
        
        Returns:
            bool: True if contains isotope markers
        """
        isotope_markers = ['13C', '(17)O', '(18)O', '17O', '18O', '15N', 'D']
        return any(marker in species_name for marker in isotope_markers)
    
    def _calculate_isotopologue_VMR(self, species_name, ParamTable):
        """
        Calculate VMR for an isotopologue based on isotope ratios.
        
        Args:
            species_name (str): Isotopologue name (e.g., '13CO', 'C18O', 'HDO')
            ParamTable (dict): Parameters including isotope ratios
        
        Returns:
            np.ndarray or None: VMR array if calculated successfully, None otherwise
        """
        # Get isotope ratio dictionaries
        all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios = \
            self._share_isotope_ratios(ParamTable)
        
        # Determine if we need to conserve total VMR
        conserve_tot_VMR = isinstance(self, EquilibriumChemistry)
        
        # Find which group this isotopologue belongs to
        all_ratios_dicts = [
            (all_CO_ratios, '12CO'),
            (all_H2O_ratios, 'H2O'),
            (all_CH4_ratios, 'CH4'),
            (all_NH3_ratios, 'NH3'),
            (all_CO2_ratios, 'CO2'),
        ]
        
        for all_ratios, main_iso_name in all_ratios_dicts:
            if species_name not in all_ratios:
                continue
            
            # Get ratio for this isotopologue
            minor_main_ratio = all_ratios.get(species_name)
            if minor_main_ratio is None or np.isinf(minor_main_ratio):
                continue
            
            # Get VMR of main isotopologue
            main_iso_VMR = None
            if main_iso_name in self.VMRs:
                main_iso_VMR = self.VMRs[main_iso_name]
            else:
                # Try pRT_name
                if main_iso_name in self.species_info.index:
                    pRT_name = self.read_species_info(main_iso_name, 'pRT_name')
                    if pRT_name in self.VMRs:
                        main_iso_VMR = self.VMRs[pRT_name]
            
            if main_iso_VMR is None:
                # Main isotopologue not found
                continue
            
            # Calculate sum of ratios
            sum_of_ratios = 1.
            if conserve_tot_VMR:
                sum_of_ratios = sum(all_ratios.values())
            
            # Calculate isotopologue VMR
            return main_iso_VMR * minor_main_ratio / sum_of_ratios
        
        return None
    
    def _get_species_mass_and_prt_name(self, species_name):
        """
        Get mass and pRT_name for a species not in species_info.
        Used for FastChemistry_Live to handle species like 'e-', 'H', 'H1-'.
        
        Args:
            species_name (str): Species name (e.g., 'e-', 'H', 'H1-')
        
        Returns:
            tuple: (pRT_name, mass) or (None, None) if not found
        """
        # Known mappings for common species not in species_info
        known_species = {
            'e-': ('e-', 5.486e-4),  # Electron mass in amu
            'H': ('H', 1.007825),  # Atomic hydrogen
            'H1-': ('H-', 1.007825 + 5.486e-4),  # H- ion (H + electron)
        }
        
        if species_name in known_species:
            return known_species[species_name]
        
        # Try to get from FastChem if available
        if hasattr(self, 'fastchem') and self.fastchem is not None:
            try:
                # Try to get Hill notation first
                hill_notation = self._get_hill_notation_from_species_name(species_name)
                if hill_notation is None:
                    hill_notation = species_name
                
                # Get index in FastChem
                idx = self.fastchem.getGasSpeciesIndex(hill_notation)
                gas_species_tot = getattr(self, 'gas_species_tot', None)
                if gas_species_tot is None:
                    gas_species_tot = self.fastchem.getGasSpeciesNumber()
                if idx < gas_species_tot:
                    # Get mass from FastChem
                    mass = self.fastchem.getGasSpeciesWeight(idx)
                    # Use species_name as pRT_name (or convert if needed)
                    pRT_name = species_name
                    # Special cases for pRT_name conversion
                    if species_name == 'H1-':
                        pRT_name = 'H-'
                    elif species_name == 'H':
                        # Try 'H' first, pRT might accept it
                        pRT_name = 'H'
                    return (pRT_name, mass)
            except Exception:
                pass
        
        return (None, None)
    
    def _get_fastchem_object(self):
        """
        Get or create the FastChem object.
        """
        if self.fastchem is not None:
            return
        
        if self.abundance_file is None or self.gas_data_file is None:
            raise ValueError(
                "FastChemistry requires 'abundance_file' and 'gas_data_file' parameters. "
                "These should point to FastChem input files:\n"
                "  - abundance_file: element abundance file (e.g., 'asplund_2009.dat')\n"
                "  - gas_data_file: gas data file (e.g., 'logK.dat')\n"
                "Example paths relative to FastChem installation:\n"
                "  - '../input/element_abundances/asplund_2009.dat'\n"
                "  - '../input/logK/logK.dat'"
            )
        
        try:
            import pyfastchem as pyfc
        except ImportError:
            raise ImportError(
                "pyfastchem is not installed. Please install FastChem Python bindings:\n"
                "  cd /home/chenyangji/ESO/tools/FastChem\n"
                "  pip install -e ."
            )
        
        verbose = 1
        if hasattr(self, 'solar_abund'):
            verbose = 0  # Suppress initialization message during sampling
        
        # Create FastChem object
        self.fastchem = pyfc.FastChem(
            self.abundance_file,
            self.gas_data_file,
            self.cond_data_file,
            verbose
        )
        
        # Configure FastChem's internal parameters
        self.fastchem.setParameter('accuracyChem', 1e-4)
        self.fastchem.setVerboseLevel(1)
        
        # Create input/output structures for FastChem
        self.input = pyfc.FastChemInput()
        self.output = pyfc.FastChemOutput()
        
        self.input.equilibrium_condensation = self.use_eq_cond
        self.input.rainout_condensation = self.use_rainout_cond
        
        # Get gas species number
        self.gas_species_tot = self.fastchem.getGasSpeciesNumber()
        
        # Get element indices and solar properties
        self._get_element_indices()
        self._get_solar()
    
    def _get_element_indices(self):
        """
        Get the indices of relevant elements in the FastChem library.
        """
        if self.fastchem is None:
            self._get_fastchem_object()
        
        # Get the indices of relevant elements
        self.idx = {
            self.fastchem.getElementSymbol(i): i
            for i in range(self.fastchem.getElementNumber())
        }
        
        # All but the H, He indices (for metallicity scaling)
        self.metal_idx = np.arange(self.fastchem.getElementNumber())
        self.metal_idx = np.delete(
            self.metal_idx, [self.idx['H'], self.idx['He']]
        )
    
    def _get_solar(self):
        """
        Get the solar abundances and ratios.
        """
        if self.fastchem is None:
            self._get_fastchem_object()
        
        # Make a copy of the solar abundances
        self.solar_abund = np.array(self.fastchem.getElementAbundances())
        
        # Solar abundance ratios
        self.solar_CO = self.solar_abund[self.idx['C']] / self.solar_abund[self.idx['O']]
        self.solar_NO = self.solar_abund[self.idx['N']] / self.solar_abund[self.idx['O']]
        
        self.solar_FeH = 0.0
    
    def _set_metallicity(self):
        """
        Set the metallicity for the element abundances.
        """
        self.el_abund[self.metal_idx] *= 10**self.FeH
    
    def _set_CO(self):
        """
        Set the C/O ratio for the element abundances.
        """
        # C = C/O * O_sol
        self.el_abund[self.idx['C']] = self.CO * self.el_abund[self.idx['O']]
        
        # Correct for the summed abundance of C+O
        tot_abund_ratio = (1 + self.solar_CO) / (1 + self.CO)
        self.el_abund[self.idx['C']] *= tot_abund_ratio
        self.el_abund[self.idx['O']] *= tot_abund_ratio
    
    def _set_NO(self):
        """
        Set the N/O ratio for the element abundances.
        """
        # N = N/O * O_sol
        self.el_abund[self.idx['N']] = self.NO * self.el_abund[self.idx['O']]
        
        # Correct for the summed abundance of N+O
        tot_abund_ratio = (1 + self.solar_NO) / (1 + self.NO)
        self.el_abund[self.idx['N']] *= tot_abund_ratio
        self.el_abund[self.idx['O']] *= tot_abund_ratio
    
    def _set_elemental_abundances(self, ParamTable):
        """
        Set the abundances of each element separately.
        
        Args:
            ParamTable (dict): Parameters including alpha_<element> for element enhancement.
        """
        for el, i in self.idx.items():
            if (el in ['K', 'Na']) and (ParamTable.get(f'alpha_{el}') is None):
                el = 'K+Na'
            
            # Enhance the elemental abundance
            alpha_i = ParamTable.get(f'alpha_{el}', ParamTable.get(f'[M/H]', None))
            if alpha_i is None:
                continue
            
            if el in ['e-', 'H', 'He']:
                continue
            
            self.el_abund[i] = 10**alpha_i * self.el_abund[i]
    
    def get_VMRs(self, ParamTable):
        """
        Get volume mixing ratios (VMRs) using the FastChem library.
        
        Processing logic for each species in save_species (or all FastChem species):
        1. Convert species_info name to Hill notation (if available in CSV)
        2. Check if FastChem calculated this species -> add to VMR
        3. If not, check log_<species> parameter -> add free chemistry VMR
        4. If not, check if contains isotope markers -> calculate from isotope ratios
        5. If none of above, raise error "cannot parse species '{species_name}'"
        
        Args:
            ParamTable (dict): Parameters for the model.
                - 'C/O': C/O ratio (optional, defaults to solar)
                - 'N/O': N/O ratio (optional, defaults to solar)
                - 'Fe/H': Fe/H metallicity (optional, defaults to 0.0)
                - 'alpha_<element>': Element enhancement factors (optional)
                - log_<species_name>: log10 of VMR value (free chemistry override)
                - <species_name>_P: Reference pressure for drop-off [bar] (optional)
                - <species_name>_alpha: Power-law exponent for drop-off (optional)
        """
        # Initialize FastChem object if needed
        self._get_fastchem_object()
        
        # Flip to order by increasing altitude (FastChem expects this)
        self.input.pressure = self.pressure[::-1]
        
        # FastChem doesn't converge for low temperatures
        temperature = self.temperature[::-1].copy()
        temperature[temperature < self.min_temperature] = self.min_temperature
        self.input.temperature = temperature
        
        # Update the parameters, if not provided, use solar values
        self.CO = ParamTable.get('C/O', self.solar_CO)
        self.NO = ParamTable.get('N/O', self.solar_NO)
        self.FeH = ParamTable.get('Fe/H', self.solar_FeH)
        
        # Modify the elemental abundances, initially solar
        self.el_abund = self.solar_abund.copy()
        self._set_CO()
        self._set_NO()
        self._set_metallicity()
        self._set_elemental_abundances(ParamTable)
        
        # Update the element abundances in FastChem
        self.fastchem.setElementAbundances(self.el_abund)
        
        # Compute the number densities
        fastchem_flag = self.fastchem.calcDensities(self.input, self.output)
        
        # # DEBUG: Check FastChem output species
        # print([self.fastchem.getGasSpeciesSymbol(i) for i in range(self.fastchem.getGasSpeciesNumber())])
        
        if (fastchem_flag != 0) or (np.amin(self.output.element_conserved) != 1):
            # FastChem failed to converge or conserve elements
            self.VMRs = -np.inf
            return
        
        # Species-specific and total number densities
        n = np.array(self.output.number_densities)
        n_tot = n.sum(axis=1)
        
        # Fill in the VMRs dictionary
        self.MMW = np.array(self.output.mean_molecular_weight)[::-1]  # Flip back
        self.VMRs = {}
        
        # Get list of FastChem-calculated species (Hill notation)
        fc = self.fastchem
        n_species_fc = fc.getGasSpeciesNumber()
        fastchem_species_hill = [
            fc.getGasSpeciesSymbol(i) for i in range(n_species_fc)
        ]
        
        # Determine which species to process
        if self.save_species is not None:
            # Process only save_species (species_info names)
            species_to_process = self.save_species.copy()
        else:
            # Process all FastChem-calculated species
            # Convert Hill notation to species_info names
            species_to_process = []
            for hill_notation in fastchem_species_hill:
                species_name = self._hill_to_species_name(hill_notation)
                if species_name is not None:
                    species_to_process.append(species_name)
                else:
                    # If not in species_info, use Hill notation as name
                    species_to_process.append(hill_notation)
        
        # Process each species
        for species_name in species_to_process:
            # Skip if already processed
            if species_name in self.VMRs:
                continue
            
            # Skip H2 and He (handled separately at the end)
            if species_name in ['H2', 'He']:
                continue
            
            # Step 1: Get Hill notation from species_info name
            # If CSV has Hill_notation, use it; otherwise keep original format
            hill_notation = self._get_hill_notation_from_species_name(species_name)
            if hill_notation is None:
                # No Hill notation found, use original name
                hill_notation = species_name
            
            # Step 2: Check if FastChem calculated this species
            idx = self.fastchem.getGasSpeciesIndex(hill_notation)
            if idx < self.gas_species_tot:
                # Found in FastChem -> add to VMR
                self.VMRs[species_name] = (n[:, idx] / n_tot)[::-1]
                continue
            
            # Step 3: Check log_<species> parameter
            param_VMR = ParamTable.get(f'log_{species_name}', None)
            if param_VMR is not None:
                # Found log parameter -> add free chemistry VMR
                param_VMR = 10 ** param_VMR
                param_VMR = np.atleast_1d(param_VMR)
                if len(param_VMR) == 1:
                    param_VMR = param_VMR[0] * np.ones(self.n_atm_layers)
                elif len(param_VMR) != self.n_atm_layers:
                    raise ValueError(
                        f"VMR for {species_name} has wrong length: "
                        f"{len(param_VMR)} vs {self.n_atm_layers}"
                    )
                
                self.VMRs[species_name] = self._power_law_drop_off(
                    param_VMR.copy(),
                    P0=ParamTable.get(f'{species_name}_P'),
                    alpha=ParamTable.get(f'{species_name}_alpha'),
                )
                continue
            
            # Step 4: Check if contains isotope markers
            if self._contains_isotope_marker(species_name):
                # Contains isotope marker -> calculate from isotope ratios
                isotopologue_VMR = self._calculate_isotopologue_VMR(species_name, ParamTable)
                if isotopologue_VMR is not None:
                    self.VMRs[species_name] = isotopologue_VMR
                    continue
                else:
                    # Contains isotope marker but cannot calculate
                    # Check if it's in isotope ratio dictionaries
                    all_CO_ratios, all_H2O_ratios, all_CH4_ratios, all_NH3_ratios, all_CO2_ratios = \
                        self._share_isotope_ratios(ParamTable)
                    all_isotopologues = [
                        *all_CO_ratios, *all_H2O_ratios, *all_CH4_ratios, 
                        *all_NH3_ratios, *all_CO2_ratios
                    ]
                    if species_name in all_isotopologues:
                        # In isotope ratio dict but main isotopologue not found
                        raise ValueError(
                            f"cannot parse isotope species '{species_name}': "
                            f"the species is in the isotope ratio dictionary, but the main isotopologue is not in the VMR dictionary. "
                            f"please ensure the main isotopologue (e.g., '12CO', 'H2O') is in save_species or has been calculated by FastChem. "
                        )
                    else:
                        # Contains isotope marker but not in isotope ratio dict
                        raise ValueError(
                            f"cannot parse isotope species '{species_name}': "
                            f"the species contains isotope markers, but is not in the supported isotope ratio dictionary. "
                            f"supported isotopes: CO (12CO, 13CO, C17O, C18O), "
                            f"H2O (H2O, H2(17)O, H2(18)O, HDO), "
                            f"CH4 (CH4, 13CH4, CH3D), NH3 (NH3, 15NH3), "
                            f"CO2 (CO2, 13CO2, CO(17)O, CO(18)O)."
                        )
            
            # Step 5: None of above -> raise error
            raise ValueError(
                f"cannot parse species '{species_name}': "
                f"FastChem did not calculate this species, and did not find the log_{species_name} parameter, "
                f"and the species does not contain isotope markers ('13C', '(17)O', '(18)O', '17O', '18O', '15N', 'D')."
            )
        
        # Ensure H2 and He are set (if not already)
        if 'H2' not in self.VMRs:
            idx_H2 = self.fastchem.getGasSpeciesIndex('H2')
            if idx_H2 < self.gas_species_tot:
                self.VMRs['H2'] = (n[:, idx_H2] / n_tot)[::-1]
            else:
                VMR_wo_H2 = np.sum([VMR_i for VMR_i in self.VMRs.values()], axis=0)
                self.VMRs['H2'] = np.clip(1 - VMR_wo_H2, 0, 1)
        
        if 'He' not in self.VMRs:
            idx_He = self.fastchem.getGasSpeciesIndex('He')
            if idx_He < self.gas_species_tot:
                self.VMRs['He'] = (n[:, idx_He] / n_tot)[::-1]
            else:
                self.VMRs['He'] = 0.15 * np.ones(self.n_atm_layers)
        
        # Handle zero abundances (set to minimum value)
        for species_i, VMR_i in self.VMRs.items():
            mask_zero = (VMR_i <= 0.)
            if mask_zero.any():
                VMR_i[mask_zero] = 1e-100

        # # DEBUG: Check VMRs dictionary
        # print(self.VMRs)
        # import pdb; pdb.set_trace() 

class FastChemstryTable(EquilibriumChemistry):
    """
    Placeholder for FastChem table-based chemistry (not yet implemented).
    Use FastChemistry for real-time FastChem calculations instead.
    """
    pass


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

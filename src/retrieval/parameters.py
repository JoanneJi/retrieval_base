"""
Read the retrieval parameters from a config file (Python .py or .txt format).
class Parameters:
    - self.free_params
    - self.constant_params
    - self.ndim

"""

from typing import Any


import os
import numpy as np
from configparser import ConfigParser
from pathlib import Path
import warnings
import importlib.util
from scipy.stats import norm

from core.paths import SRC_DIR

class Parameters:
    """
    Stores retrieval parameters (free + constant) and handles
    mapping from unit cube [0,1] to actual parameter values.
    Can load free/constant params from a config file (Python .py or .txt format).
    """

    def __init__(self, free_params=None, constant_params=None, config_file=None, debug=False):
        """
        Initialize Parameters either from dicts or a config file.
        
        Args:
            free_params (dict, optional): Free parameters dict. If None, will try to load from config_file.
            constant_params (dict, optional): Constant parameters dict. If None, will try to load from config_file.
            config_file (str, optional): Path to config file. 
                - If .py file: expects free_params and constant_params dictionaries
                - If .txt file: expects ConfigParser format
                - If None: defaults to config/parameters.py
        """
        # Default config file path
        if config_file is None:
            print("No config file provided. Using default: config/parameters.py")
            config_file = os.path.join(SRC_DIR, "config/parameters.py")

        # debug flag
        self.debug = debug
        
        # ----- initialize parameters -----
        # Store config_file path for later use
        self.config_file = config_file
        
        # load the parameters from config file if provided
        if config_file is not None and os.path.exists(config_file):
            loaded_free, loaded_constant = self.load_from_file(config_file)
            loaded_TP_kwargs, loaded_chemistry_kwargs = self.load_model_kwargs_from_file(config_file)
            # Use loaded params if not explicitly provided
            if free_params is None:
                free_params = loaded_free
            if constant_params is None:
                constant_params = loaded_constant
            # Store loaded kwargs as instance attributes
            self.TP_kwargs = loaded_TP_kwargs
            self.chemistry_kwargs = loaded_chemistry_kwargs
        elif config_file is not None and not os.path.exists(config_file):
            warnings.warn(f"Config file {config_file} not found. Using provided dicts or defaults.")
             # Initialize kwargs as empty dicts if config file doesn't exist
            self.TP_kwargs = {}
            self.chemistry_kwargs = {}
        else:
            warnings.warn("No config file provided. Initializing empty TP_kwargs and chemistry_kwargs.")
            # No config_file provided, initialize kwargs as empty dicts
            self.TP_kwargs = {}
            self.chemistry_kwargs = {}
        
        # if there is no input, initialize empty dicts and raise warnings
        if free_params is None:
            free_params = {}
            warnings.warn("No free parameters provided. Initializing empty free_params.")
        if constant_params is None:
            constant_params = {}
            warnings.warn("No constant parameters provided. Initializing empty constant_params.")
        
        # Convert free_params format if needed (from tuple format to dict format)
        self.free_params = self._normalize_free_params(free_params)
        self.constant_params = constant_params
        self.params = {}  # dict storing current parameter values

        # ----- extract math labels for plotting -----
        self.param_bounds = {k: v["bounds"] for k, v in self.free_params.items()}
        self.param_mathtext = {k: v["label"] for k, v in self.free_params.items()}
        self.param_types = {k: v.get("type", "uniform") for k, v in self.free_params.items()}

        # ----- keys and dimensions -----
        self.param_keys = list[Any](self.free_params.keys())
        self.n_params = len(self.param_keys)
        self.ndim = self.n_params

        # ----- update params dict with constant values -----
        self.params.update(constant_params)
    
    def _normalize_free_params(self, free_params):
        """
        Normalize free_params format to internal dict format.
        
        Supports two input formats:
        1. Tuple format: {'T0': ([0, 5000], r'$T_0$', 'uniform'), ...}
        2. Dict format: {'T0': {'bounds': [0, 5000], 'label': r'$T_0$', 'type': 'uniform'}, ...}
        
        Returns:
            dict: Normalized free_params in dict format
        """
        normalized = {}
        for key, value in free_params.items():
            if isinstance(value, tuple) and len(value) >= 2:
                # Tuple format: (bounds, label, type)
                bounds = value[0]
                label = value[1]
                param_type = value[2] if len(value) > 2 else "uniform"
                normalized[key] = {
                    "bounds": bounds,
                    "label": label,
                    "type": param_type
                }
            elif isinstance(value, dict):
                # Already in dict format
                normalized[key] = value
            else:
                raise ValueError(f"Invalid format for parameter {key}. Expected tuple (bounds, label, type) or dict.")
        
        return normalized

    @staticmethod
    def uniform_prior(min, max, x):
        """Map [0,1] x to [bounds[0], bounds[1]]"""
        return min + x * (max - min)

    @staticmethod
    def normal_prior(mean, sigma, x):
        """Map x ∈ [0,1] -> value distributed normally with mean, sigma."""
        return norm.ppf(x, loc=mean, scale=sigma)

    @staticmethod
    def load_from_file(filename):
        """
        Load free and constant parameters from a config file.
        
        Supports two file formats:
        1. Python .py file: expects free_params and constant_params dictionaries
        2. ConfigParser .txt file: expects [constant] and [free] sections
        
        Args:
            filename (str): Path to config file
            
        Returns:
            tuple: (free_params, constant_params) dictionaries
        """
        filename = Path(filename)
        
        if filename.suffix == '.py':
            # Load from Python file
            print(f"Loading parameters from Python file: {filename}")
            return Parameters._load_from_py_file(str(filename))
        elif filename.suffix == '.txt':
            # Load from ConfigParser file
            print(f"Loading parameters from txt file: {filename}")
            return Parameters._load_from_txt_file(str(filename))
        else:
            # Try to detect format by content
            try:
                return Parameters._load_from_py_file(str(filename))
            except:
                return Parameters._load_from_txt_file(str(filename))
    
    @staticmethod
    def _load_from_py_file(filename):
        """Load parameters from a Python config file."""
        spec = importlib.util.spec_from_file_location("config_module", filename)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load config file: {filename}")
        
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract free_params and constant_params
        free_params = getattr(config_module, 'free_params', {})
        constant_params = getattr(config_module, 'constant_params', {})
        
        return free_params, constant_params
    
    @staticmethod
    def load_model_kwargs_from_file(filename):
        """
        Load TP_kwargs and chemistry_kwargs from a Python config file.
        
        Args:
            filename (str): Path to config file
            
        Returns:
            tuple: (TP_kwargs, chemistry_kwargs) dictionaries
        """
        filename = Path(filename)
        
        if filename.suffix != '.py':
            # Only Python files support model kwargs
            warnings.warn(f"Only Python files support model kwargs. Add **kwargs to the Python config file.")
            return {}, {}
        
        spec = importlib.util.spec_from_file_location("config_module", str(filename))
        if spec is None or spec.loader is None:
            return {}, {}
        
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Extract TP_kwargs and chemistry_kwargs
        TP_kwargs = getattr(config_module, 'TP_kwargs', {})
        chemistry_kwargs = getattr(config_module, 'chemistry_kwargs', {})
        
        return TP_kwargs, chemistry_kwargs
    
    @staticmethod
    def _load_from_txt_file(filename):
        """Load parameters from a ConfigParser .txt file."""
        cfg = ConfigParser()
        cfg.read(filename)

        # Constant parameters
        constant_params = {}
        if "constant" in cfg:
            for k, v in cfg["constant"].items():
                # Format: "value | label" or just "value"
                parts = v.split("|")
                value = float(parts[0].strip())
                constant_params[k.strip()] = value

        # Free parameters
        free_params = {}
        if "free" in cfg:
            for key, val in cfg["free"].items():
                parts = val.split("|")
                bounds = parts[0].strip()
                label = parts[1].strip() if len(parts) > 1 else key
                prior_type = parts[2].strip() if len(parts) > 2 else "uniform"
                # parse bounds
                lo, hi = map(float, bounds.split(","))
                free_params[key.strip()] = {
                    "bounds": [lo, hi],
                    "label": label,
                    "type": prior_type
                }

        return free_params, constant_params

    def __call__(self, cube: np.ndarray, ndim=None, nparams=None):
        """
        Map unit cube [0,1] to actual parameter values and update self.params.
        
        This is the prior transformation function used by MultiNest.
        It maps the unit cube [0,1]^ndim to the physical parameter space.
        
        Special handling for temperature knots (T0, T1, T2, T3, T4):
        - Ensures monotonic decrease: T0 > T1 > T2 > T3 > T4
        - This prevents temperature inversions for isolated objects
        
        Args:
            cube (np.ndarray): Unit cube values in [0,1], length must equal ndim
            ndim (int, optional): Number of dimensions (passed by pymultinest)
            nparams (int, optional): Number of parameters (passed by pymultinest, can be ignored)
            
        Returns:
            np.ndarray: The same cube (for MultiNest compatibility)
        """
        # Handle pymultinest calling with (cube, ndim, nparams)
        # Extract only the relevant dimensions if ndim is provided
        if ndim is not None:
            cube = np.array(cube[:ndim])
        else:
            cube = np.array(cube)
        
        if len(cube) != self.ndim:
            raise ValueError(f"Cube length {len(cube)} != number of free parameters {self.ndim}")

        # First pass: process all non-temperature parameters and T_0/T0
        # Second pass: process temperature knots T_1, T_2, ... in order
        temp_knot_indices = {}  # Map temp_knot_index -> (i, key) for deferred processing
        processed_temp_knots = set()  # Track which temperature knots have been processed
        
        for i, key in enumerate(self.param_keys):
            info = self.free_params[key]
            
            # Special handling for temperature knots to prevent inversions
            # Supports both T0/T1/T2/... and T_0/T_1/T_2/... naming conventions
            is_temp_knot = False
            temp_knot_index = None
            if key.startswith('T_') and key[2:].isdigit():
                # Format: T_0, T_1, T_2, ...
                temp_knot_index = int(key[2:])
                is_temp_knot = True
            elif key.startswith('T') and key[1:].isdigit():
                # Format: T0, T1, T2, ...
                temp_knot_index = int(key[1:])
                is_temp_knot = True
            
            # For temperature knots T_1, T_2, T_3, T_4 (or T1, T2, T3, T4),
            # ensure monotonic decrease: T_0 > T_1 > T_2 > T_3 > T_4
            # This prevents temperature inversions for isolated objects (like in Zhang+2021)
            if is_temp_knot and temp_knot_index > 0:
                # Defer processing of temperature knots until we've processed all non-temp params
                # and T_0, so we can ensure proper ordering
                temp_knot_indices[temp_knot_index] = (i, key)
                continue
            else:
                # Normal parameter conversion (non-temperature knots or T_0/T0)
                if info["type"] == "uniform":
                    min_val, max_val = info["bounds"]
                    converted_value = self.uniform_prior(min_val, max_val, cube[i])
                    self.params[key] = converted_value
                    # Debug: Print conversion for first few parameters
                    if (i < 3) & (self.debug):
                        print(f"[DEBUG Parameters.__call__]: {key}: cube[{i}]={cube[i]:.4f} -> {converted_value:.2f} (bounds=[{min_val}, {max_val}])")
                elif info["type"] == "normal":
                    mean, sigma = info["bounds"]  # bounds stand for [mean, sigma]
                    converted_value = self.normal_prior(mean, sigma, cube[i])
                    self.params[key] = converted_value
                else:
                    # Default to uniform if type not recognized
                    warnings.warn(f"Unknown prior type '{info['type']}' for parameter {key}, using uniform.")
                    min_val, max_val = info["bounds"]
                    converted_value = self.uniform_prior(min_val, max_val, cube[i])
                    self.params[key] = converted_value
        
        # Second pass: process temperature knots T_1, T_2, ... in order
        # This ensures that T_0 is processed before T_1, T_1 before T_2, etc.
        for temp_knot_index in sorted(temp_knot_indices.keys()):
            i, key = temp_knot_indices[temp_knot_index]
            info = self.free_params[key]
            
            # Get the previous temperature knot value (should already be converted)
            use_underscore = key.startswith('T_')
            prev_key = f'T_{temp_knot_index-1}' if use_underscore else f'T{temp_knot_index-1}'
            
            if prev_key not in self.params:
                # Try to find previous knot in param_keys (handles different naming conventions)
                found_prev = False
                for k in self.param_keys:
                    if use_underscore:
                        if k.startswith('T_') and k[2:].isdigit() and int(k[2:]) == temp_knot_index - 1:
                            prev_key = k
                            found_prev = True
                            break
                    else:
                        if k.startswith('T') and k[1:].isdigit() and int(k[1:]) == temp_knot_index - 1:
                            prev_key = k
                            found_prev = True
                            break
                
                if not found_prev or prev_key not in self.params:
                    # This shouldn't happen if T_0 is defined, but handle gracefully
                    warnings.warn(f"Previous temperature knot {prev_key} not found for {key}. "
                                f"Using full prior range. This may cause temperature inversions.")
                    if info["type"] == "uniform":
                        min_val, max_val = info["bounds"]
                        converted_value = self.uniform_prior(min_val, max_val, cube[i])
                    else:
                        min_val, max_val = info["bounds"]
                        converted_value = self.uniform_prior(min_val, max_val, cube[i])
                    self.params[key] = converted_value
                    continue
            
            prev_temp = self.params[prev_key]
            
            # Constrain current temperature to be between 0.5*prev_temp and prev_temp
            # This ensures T_i < T_{i-1} (monotonic decrease)
            constrained_min = prev_temp * 0.5
            constrained_max = prev_temp
            converted_value = self.uniform_prior(constrained_min, constrained_max, cube[i])
            self.params[key] = converted_value
            
            if self.debug:
                print(f"[DEBUG Parameters.__call__]: {key}: cube[{i}]={cube[i]:.4f} -> {converted_value:.2f} "
                      f"(constrained: [{constrained_min:.2f}, {constrained_max:.2f}], "
                      f"prev {prev_key}={prev_temp:.2f})")
        
        return cube

# run "python -m retrieval.parameters" in src/ to test
# parameters = Parameters()
# parameters = Parameters(config_file=os.path.join(SRC_DIR, "config/config_example.py"))
# print(parameters.free_params, parameters.constant_params)
# print(parameters.param_keys)
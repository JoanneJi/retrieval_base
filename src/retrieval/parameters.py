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

    def __init__(self, free_params=None, constant_params=None, config_file=None, debug=False, inverse_flag=True):
        """
        Initialize Parameters either from dicts or a config file.
        
        Args:
            free_params (dict, optional): Free parameters dict. If None, will try to load from config_file.
            constant_params (dict, optional): Constant parameters dict. If None, will try to load from config_file.
            config_file (str, optional): Path to config file. 
                - If .py file: expects free_params and constant_params dictionaries
                - If .txt file: expects ConfigParser format
                - If None: defaults to config/parameters.py
            inverse_flag (bool, optional): If True (default), temperature knots are sampled from prior only.
                If False, sampling is constrained so that T_4 < T_3 < T_2 < T_1 < T_0 (monotonic in altitude).
                For .py config files, this can be overridden by defining `inverse_flag` in the config.
        """
        # Default config file path
        if config_file is None:
            print("No config file provided. Using default: config/parameters.py")
            config_file = os.path.join(SRC_DIR, "config/parameters.py")

        # debug flag
        self.debug = debug
        self.inverse_flag = inverse_flag

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
            # Read optional options from .py config (e.g. inverse_flag)
            if Path(config_file).suffix == '.py':
                opts = Parameters._load_optional_from_py(str(config_file), ['inverse_flag'])
                if 'inverse_flag' in opts:
                    self.inverse_flag = bool(opts['inverse_flag'])
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

    def _get_sorted_temperature_knot_keys(self):
        """
        Return temperature knot parameter keys sorted by knot index (T_0, T_1, ..., T_4),
        as (key, param_keys_index) list. Only keys that exist in param_keys are included.
        """
        temp_knot_list = []  # (knot_index, key, param_keys_index)
        for i, key in enumerate(self.param_keys):
            if key.startswith('T_') and key[2:].isdigit():
                knot_index = int(key[2:])
                temp_knot_list.append((knot_index, key, i))
            elif key.startswith('T') and key[1:].isdigit() and len(key) > 1:
                knot_index = int(key[1:])
                temp_knot_list.append((knot_index, key, i))
        temp_knot_list.sort(key=lambda x: x[0])
        return [(key, idx) for _, key, idx in temp_knot_list]

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
    def _load_optional_from_py(filename, option_names):
        """
        Load optional attributes from a Python config file.
        Only returns attributes that exist in the config module.

        Args:
            filename (str): Path to .py config file
            option_names (list of str): Attribute names to read

        Returns:
            dict: {name: value} for each name that exists in the config
        """
        filename = Path(filename)
        if filename.suffix != '.py' or not filename.exists():
            return {}
        spec = importlib.util.spec_from_file_location("config_module", str(filename))
        if spec is None or spec.loader is None:
            return {}
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return {k: getattr(config_module, k) for k in option_names if hasattr(config_module, k)}

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
        Temperature knot values are strictly clipped to their config prior bounds.
        If inverse_flag is True (default), no monotonic constraint is applied.
        If inverse_flag is False, sampling enforces T_4 < T_3 < T_2 < T_1 < T_0
        while each knot remains within its prior bounds.

        Args:
            cube (np.ndarray): Unit cube values in [0,1], length must equal ndim
            ndim (int, optional): Number of dimensions (passed by pymultinest)
            nparams (int, optional): Number of parameters (passed by pymultinest, can be ignored)

        Returns:
            np.ndarray: The same cube (for MultiNest compatibility)
        """
        # Handle pymultinest calling with (cube, ndim, nparams)
        if ndim is not None:
            cube = np.array(cube[:ndim])
        else:
            cube = np.array(cube)

        if len(cube) != self.ndim:
            raise ValueError(f"Cube length {len(cube)} != number of free parameters {self.ndim}")

        for i, key in enumerate(self.param_keys):
            info = self.free_params[key]
            if info["type"] == "uniform":
                min_val, max_val = info["bounds"]
                converted_value = self.uniform_prior(min_val, max_val, cube[i])
                # Strictly clip to prior interval (ensure knots lie inside bounds)
                converted_value = float(np.clip(converted_value, min_val, max_val))
                self.params[key] = converted_value
                if (i < 3) and self.debug:
                    print(f"[DEBUG Parameters.__call__]: {key}: cube[{i}]={cube[i]:.4f} -> {converted_value:.2f} (bounds=[{min_val}, {max_val}])")
            elif info["type"] == "normal":
                mean, sigma = info["bounds"]  # bounds stand for [mean, sigma]
                converted_value = self.normal_prior(mean, sigma, cube[i])
                # Clip to a reasonable range (e.g. mean ± 5*sigma) to avoid infinities
                clip_lo = mean - 5 * sigma
                clip_hi = mean + 5 * sigma
                converted_value = float(np.clip(converted_value, clip_lo, clip_hi))
                self.params[key] = converted_value
            else:
                warnings.warn(f"Unknown prior type '{info['type']}' for parameter {key}, using uniform.")
                min_val, max_val = info["bounds"]
                converted_value = self.uniform_prior(min_val, max_val, cube[i])
                converted_value = float(np.clip(converted_value, min_val, max_val))
                self.params[key] = converted_value

        # When inverse_flag is False, enforce T_4 < T_3 < T_2 < T_1 < T_0 (each within prior)
        if not self.inverse_flag:
            sorted_knots = self._get_sorted_temperature_knot_keys()
            if len(sorted_knots) > 1:
                upper = np.inf
                for key, idx in sorted_knots:
                    info = self.free_params[key]
                    low, high = info["bounds"]
                    # Valid range: [low, min(high, upper)] so result < upper and in prior
                    effective_high = min(high, upper)
                    u = float(np.clip(cube[idx], 0.0, 1.0))
                    self.params[key] = low + u * (effective_high - low)
                    upper = self.params[key]

        return cube

# run "python -m retrieval.parameters" in src/ to test
# parameters = Parameters()
# parameters = Parameters(config_file=os.path.join(SRC_DIR, "config/config_example.py"))
# print(parameters.free_params, parameters.constant_params)
# print(parameters.param_keys)
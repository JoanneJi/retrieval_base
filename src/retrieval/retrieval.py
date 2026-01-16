import os
import numpy as np
import pathlib
import pickle
import warnings
import re
import time

import pymultinest

from retrieval.likelihood import LogLikelihood, Covariance
from atmosphere.make_spectrum import pRT_spectrum
from atmosphere import get_species_from_params, setup_radtrans_atmosphere
from utils.plotting import cornerplot, plot_spectrum, plot_tp_profile

from core.paths import OUTPUT_DIR

class Retrieval:
    """
    High-level controller for atmospheric retrieval.

    Responsibilities:
    - Interface with sampler (MultiNest)
    - Map parameters -> forward model -> likelihood
    - Store and analyze posterior samples
    - Plot and save retrieval results

    Workflow:
        cube ∈ [0,1]^ndim
        ↓ Parameters.__call__
        physical parameters θ
        ↓ pRT_spectrum
        model spectrum m(θ)
        ↓ LogLikelihood
        ln L(d | θ)
        ↓ MultiNest
        posterior samples

    """

    # ===== Initialization =====
    def __init__(self, parameters, target, N_live_points=200, evidence_tolerance=0.5, output_base=OUTPUT_DIR / "retrievals", output_subdir=None, lbl_opacity_sampling=3, n_atm_layers=50, wl_pad=7, redo_atmosphere=True):
        self.parameters = parameters
        self.target = target
    
        # Sampler configuration
        self.N_live_points = int(N_live_points)
        self.evidence_tolerance = float(evidence_tolerance)

        # Output directory
        # If output_subdir is provided, create: output_base / output_subdir / N{...}_ev{...}
        # Otherwise, create: output_base / N{...}_ev{...}
        output_base_path = pathlib.Path(output_base)
        if output_subdir is not None:
            self.output_dir = output_base_path / output_subdir / f"N{self.N_live_points}_ev{self.evidence_tolerance}"
        else:
            self.output_dir = output_base_path / f"N{self.N_live_points}_ev{self.evidence_tolerance}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Atmosphere configuration
        self.lbl_opacity_sampling = lbl_opacity_sampling
        self.n_atm_layers = n_atm_layers
        self.wl_pad = wl_pad  # wavelength padding for atmosphere object
        
        # Create pressure grid from TP_kwargs (to ensure consistency with TP_profile)
        self.pressure = self._create_pressure_from_tp_kwargs()

        # Get species_info_path from chemistry_kwargs (will be None if not specified, which uses default)
        chemistry_kwargs = self.parameters.chemistry_kwargs if hasattr(self.parameters, 'chemistry_kwargs') else {}
        species_info_path = chemistry_kwargs.get('species_info_path')
        # Create a dict with all parameter keys (free + constant) for species extraction
        # Note: params dict may not have free_params keys yet (they're set during sampling),
        # so we need to include free_params keys as well. We only need the keys, not the values.
        all_param_keys = dict.fromkeys(list(self.parameters.params.keys()) + self.parameters.param_keys)
        self.species = get_species_from_params(param_dict=all_param_keys, species_info_path=species_info_path)

        # Create atmosphere objects
        self.atmosphere = setup_radtrans_atmosphere(
            species=self.species,
            target_wavelengths=self.target.wl,
            pressure=self.pressure,
            lbl_opacity_sampling=self.lbl_opacity_sampling,
            wl_pad=self.wl_pad,
            cache_file=self.output_dir / "atmosphere_objects.pickle",
            redo=redo_atmosphere
        )

        # Likelihood components
        self.cov = Covariance(err=self.target.err[self.target.mask])
        self.loglike = LogLikelihood(
            target=self.target,
            covariance=self.cov,
            scale_flux=True,
            scale_err=True,
        )

        # Runtime containers
        self.posterior = None
        self.bestfit_params = None
        self.model_flux = None
        self.params_dict = None
        self.model = None  # Store pRT_spectrum model for accessing TP profile
        
        # Store history of TP profiles for live plotting (max 50)
        self.tp_history = []  # List of (temperature, pressure) tuples

        # Plotting
        self.color = self.target.color
        self.callback_label = "live_"
        self.prefix = "pmn_"

        print(
            f"[retrieval.py/Retrieval.__init__] Initialized Retrieval for {self.target.name} "
            f"with {self.parameters.ndim} free parameters.\n"
            f"Parameters: {self.parameters.params}\n"
            f"TP_kwargs: {self.parameters.TP_kwargs}\n"
            f"chemistry_kwargs: {self.parameters.chemistry_kwargs}\n"
        )

    # ----- MultiNest interface -----
    def PMN_lnL(self, cube, ndim, nparams):
        """
        From the parameter space to the log likelihood, and pass the log-likelihood function to MultiNest.
        """
        # Always log to file to verify function is being called (even if print doesn't work)
        # Use absolute path and ensure directory exists
        debug_file = self.output_dir / "PMN_lnL_debug.log"
        try:
            # Ensure directory exists
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            with open(debug_file, 'a') as f:
                import sys
                f.write(f"[PMN_lnL] CALLED! cube[:3]={cube[:3] if len(cube) >= 3 else cube}, ndim={ndim}, nparams={nparams}\n")
                f.write(f"[PMN_lnL] sys.stdout: {sys.stdout}\n")
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
        except Exception as e:
            # Even if logging fails, try to write to a simple file
            try:
                with open("/tmp/PMN_lnL_called.txt", 'a') as f:
                    f.write(f"Called at {__import__('time').time()}\n")
            except:
                pass
        
        try:
            # Map unit cube to physical parameters
            self.parameters(cube, ndim=ndim, nparams=nparams)
            
            # Debug: Log to file
            try:
                with open(debug_file, 'a') as f:
                    f.write(f"[PMN_lnL] After conversion: T_0={self.parameters.params.get('T_0')}, T_1={self.parameters.params.get('T_1')}\n")
                    f.write(f"[PMN_lnL] All params keys: {list(self.parameters.params.keys())}\n")
                    f.flush()
            except:
                pass
            
            # Forward model
            model = pRT_spectrum(
                parameters=self.parameters,
                target=self.target,
                atmosphere=self.atmosphere,
                lbl_opacity_sampling=self.lbl_opacity_sampling,
            )
            model_flux = model.make_spectrum()

            # Likelihood
            lnL = self.loglike(model_flux)
            
            # Debug: Log likelihood
            try:
                with open(debug_file, 'a') as f:
                    f.write(f"[PMN_lnL] lnL = {lnL:.2f}\n")
                    f.flush()
            except:
                pass

            return lnL
        except (ValueError, TypeError, AttributeError) as e:
            # Log exception to file
            try:
                with open(debug_file, 'a') as f:
                    f.write(f"[PMN_lnL] Exception: {type(e).__name__}: {e}\n")
                    import traceback
                    f.write(traceback.format_exc())
                    f.flush()
            except:
                pass
            # If model creation fails (e.g., invalid chemistry, temperature profile),
            # return -inf to reject this parameter combination
            return -np.inf
        except Exception as e:
            # Catch all other exceptions
            try:
                with open(debug_file, 'a') as f:
                    f.write(f"[PMN_lnL] Unexpected exception: {type(e).__name__}: {e}\n")
                    import traceback
                    f.write(traceback.format_exc())
                    f.flush()
            except:
                pass
            return -np.inf

    # ----- Callback & analysis -----
    def PMN_callback(self, n_samples, n_live, n_params, live_points, posterior, stats, max_ln_L, ln_Z, ln_Z_err, nullcontext):
        """
        Live callback during MultiNest run.
        
        Note: The posterior passed to this callback is weighted samples in unit cube [0,1].
        For live plotting, we use weighted samples directly (consistent with original implementation).
        Final analysis will use equal-weighted posterior for more accurate error bars.
        """
        print(
            f"Callback: {n_samples} samples, "
            f"max lnL = {max_ln_L:.2f}, "
            f"lnZ = {ln_Z:.2f} ± {ln_Z_err:.2f}"
        )

        # Use weighted posterior directly (same as original implementation)
        # Note: posterior is in unit cube [0,1], will be converted to physical values in cornerplot()
        lnL = posterior[:, -2]
        self.posterior = posterior[:, :-2]  # Remove last 2 columns (lnL and weight)
        self.bestfit_params = self.posterior[np.argmax(lnL)]

        try:
            self.params_dict, self.model_flux = self.get_params_and_spectrum()
            
            # Save current TP profile to history (for live plotting)
            if self.model is not None:
                self.tp_history.append((self.model.temperature.copy(), self.model.pressure.copy()))
                # Keep only last 50 profiles
                if len(self.tp_history) > 50:
                    self.tp_history.pop(0)
            
            self.cornerplot()
            self.plot_spectrum()
            self.plot_tp_profile()
        except (ValueError, TypeError, AttributeError):
            # If get_params_and_spectrum fails, skip plotting but continue sampling
            pass

    # ===== MultiNest run =====
    def PMN_run(self, resume=True):
        """
        Run MultiNest sampling.
        """
        print(
            f"\n--- Running retrieval "
            f"(N_live={self.N_live_points}, ev_tol={self.evidence_tolerance}) ---\n"
        )
        
        # Print diagnostic information before starting MultiNest
        try:
            import psutil
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            print(f"[Diagnostics] CPU cores: {cpu_count}, CPU usage: {cpu_percent:.1f}%, "
                  f"Memory: {mem.percent:.1f}% used ({mem.available / 1024**3:.1f} GB available)")
        except ImportError:
            print("[Diagnostics] psutil not available, skipping system diagnostics")
        print(f"[Diagnostics] Thread limits: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}")
        print(f"[Diagnostics] Starting MultiNest (this may take a moment to initialize)...\n")

        pymultinest.run(
            LogLikelihood=self.PMN_lnL,
            Prior=self.parameters,
            n_dims=self.parameters.ndim,
            outputfiles_basename=str(self.output_dir / self.prefix),
            n_live_points=self.N_live_points,
            evidence_tolerance=self.evidence_tolerance,
            resume=resume,
            verbose=True,
            const_efficiency_mode=True,
            sampling_efficiency=0.5,
            dump_callback=self.PMN_callback,
            n_iter_before_update=10,  # call back every 10 iterations
        )

    # ----- Post-MultiNest analysis -----
    def _fix_multinest_file_format(self, filepath):
        """
        Fix scientific notation format issues in MultiNest output files.
        
        MultiNest sometimes writes numbers like '-0.139053279005341165-308' 
        instead of '-0.139053279005341165e-308'. This function fixes such issues.
        
        Args:
            filepath: Path to the file to fix
        """
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Fix scientific notation: pattern matches number followed by -/+
            # followed by digits, but NOT if preceded by 'e' or 'E'
            # Pattern: (negative lookbehind for 'e' or 'E') + number + sign + exponent
            # Matches cases like: "1.23-308" or "-0.139-308" but not "1.23e-308"
            pattern = r'(?<![eE])([-+]?\d+\.?\d*)([-+])(\d+)'
            def replace_sci_notation(match):
                num, sign, exp = match.groups()
                # Only replace if it looks like scientific notation (exponent >= 2 digits)
                # This avoids replacing things like "1-2" (which might be subtraction)
                if len(exp) >= 2:
                    return f"{num}e{sign}{exp}"
                return match.group(0)  # Return original if not scientific notation
            
            fixed_content = re.sub(pattern, replace_sci_notation, content)
            
            # Only write if content changed
            if fixed_content != content:
                with open(filepath, 'w') as f:
                    f.write(fixed_content)
        except Exception:
            # Silently fail if we can't fix the file
            pass
    
    def analyse(self):
        """
        Final analysis after MultiNest finishes.
        """
        # Fix MultiNest output file format issues before reading
        data_file = self.output_dir / f"{self.prefix}post_equal_weights.dat"
        if data_file.exists():
            self._fix_multinest_file_format(data_file)
        
        # Also fix other potential output files
        for file_pattern in ["*.txt", "*.dat"]:
            for filepath in self.output_dir.glob(f"{self.prefix}{file_pattern}"):
                self._fix_multinest_file_format(filepath)
        
        try:
            analyzer = pymultinest.Analyzer(
                n_params=self.parameters.ndim,
                outputfiles_basename=str(self.output_dir / self.prefix),
            )

            stats = analyzer.get_stats()
            self.posterior = analyzer.get_equal_weighted_posterior()[:, :-1]
            self.lnZ = stats["nested importance sampling global log-evidence"]

            np.save(self.output_dir / "posterior.npy", self.posterior)
            self.lnZ = stats['nested importance sampling global log-evidence']
        except ValueError as e:
            # If file format is still problematic, try to fix all files and retry
            if "could not convert string" in str(e) or "float64" in str(e):
                # Try to fix all MultiNest output files
                for file_pattern in ["*.txt", "*.dat"]:
                    for filepath in self.output_dir.glob(f"{self.prefix}{file_pattern}"):
                        self._fix_multinest_file_format(filepath)
                
                # Retry analysis
                analyzer = pymultinest.Analyzer(
                    n_params=self.parameters.ndim,
                    outputfiles_basename=str(self.output_dir / self.prefix),
                )
                stats = analyzer.get_stats()
                self.posterior = analyzer.get_equal_weighted_posterior()[:, :-1]
                self.lnZ = stats["nested importance sampling global log-evidence"]
                np.save(self.output_dir / "posterior.npy", self.posterior)
            else:
                raise

    # ----- Post-processing -----
    def get_params_and_spectrum(self):
        """
        Compute median parameters and final model spectrum.

        Workflow:
            posterior → median θ (normalized) → physical θ
            ↓ pRT_spectrum
            final model
            ↓ LogLikelihood (again)
            diagnostics (phi, s2, chi2)
        """
        if self.posterior is None:
            raise ValueError("Posterior not available. Run analyse() first.")
        
        self.params_dict = {}

        # Get median values from posterior (these are normalized [0,1] values)
        median_normalized = np.array([
            np.percentile(self.posterior[:, i], 50.0) 
            for i in range(self.parameters.ndim)
        ])

        # Convert normalized values to physical parameter values using Parameters.__call__
        # This properly handles prior transformations (uniform, normal, etc.)
        self.parameters(median_normalized)  # This updates self.parameters.params with physical values
        
        # Store physical parameter values in params_dict
        for key in self.parameters.param_keys:
            self.params_dict[key] = self.parameters.params[key]
        
        # Also update constant_params to ensure all required parameters are present
        self.parameters.params.update(self.parameters.constant_params)

        # Build final model
        model = pRT_spectrum(
            parameters=self.parameters,
            target=self.target,
            atmosphere=self.atmosphere,
            lbl_opacity_sampling=self.lbl_opacity_sampling,
        )
        self.model_flux = model.make_spectrum()
        self.model = model  # Store model for accessing TP profile

        # Likelihood diagnostics
        _ = self.loglike(self.model_flux)  # cov already set as self.cov by default before
        self.params_dict["phi"] = self.loglike.phi  # flux scaling parameter
        self.params_dict["s2"] = self.loglike.s2  # error inflation parameter
        self.params_dict["chi2_red"] = self.loglike.chi2_0_red  # reduced chi-squared, based on degrees of freedom

        with open(self.output_dir / "params_dict.pkl", "wb") as f:
            pickle.dump(self.params_dict, f)

        return self.params_dict, self.model_flux

    # ===== Full retrieval =====
    def run_retrieval(self):
        """
        Full retrieval pipeline.
        """
        start_time = time.time()
        self.PMN_run()
        print(f"\n[retrieval.py/Retrieval.run_retrieval] ----- MultiNest run completed. -----\n")
        
        self.analyse()
        print(f"\n[retrieval.py/Retrieval.run_retrieval] ----- Analysis completed. -----\n")
        
        self.get_params_and_spectrum()
        self.callback_label = "final_"
        
        print(f"[retrieval.py/Retrieval.run_retrieval] Creating plots...")
        try:
            self.cornerplot()
            print(f"[retrieval.py/Retrieval.run_retrieval] cornerplot completed.")
        except Exception as e:
            print(f"[retrieval.py/Retrieval.run_retrieval] Error in cornerplot: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            self.plot_spectrum()
            print(f"[retrieval.py/Retrieval.run_retrieval] plot_spectrum completed.")
            
            # ----- save the spectrum to a dat file -----
            # Ensure all arrays are 1D numpy arrays for column_stack
            wl = np.asarray(self.target.wl).flatten()
            fl = np.asarray(self.target.fl).flatten()
            err = np.asarray(self.target.err).flatten()
            model_fl = np.asarray(self.model_flux).flatten()
            residuals = np.asarray(self.target.fl - self.model_flux).flatten()
            np.savetxt(self.output_dir / f"{self.callback_label}model_spectrum.dat", np.column_stack((wl, fl, err, model_fl, residuals)), header='Wavelength (nm) Data_Flux Data_Err Model_Flux Residuals', fmt='%.6f %.6f %.6f %.6f %.6f')
            print(f"[retrieval.py/Retrieval.run_retrieval] Model spectrum saved to {self.output_dir / f'{self.callback_label}model_spectrum.dat'}")

        except Exception as e:
            print(f"[retrieval.py/Retrieval.run_retrieval] Error in plot_spectrum: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            self.plot_tp_profile()
            print(f"[retrieval.py/Retrieval.run_retrieval] plot_tp_profile completed.")
        except Exception as e:
            print(f"[retrieval.py/Retrieval.run_retrieval] Error in plot_tp_profile: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- Retrieval finished ---\n")

        end_time = time.time()
        print(f"[retrieval.py/Retrieval.run_retrieval] Retrieval finished in {end_time - start_time:.2f} seconds.")
    # ===== Plotting =====
    def _convert_posterior_to_physical(self, posterior_normalized):
        """
        Convert posterior samples from unit cube [0,1] to physical parameter values.
        
        Args:
            posterior_normalized (np.ndarray): Posterior samples in unit cube [0,1], 
                shape (n_samples, n_params)
        
        Returns:
            np.ndarray: Posterior samples in physical parameter space, 
                shape (n_samples, n_params)
        """
        n_samples = posterior_normalized.shape[0]
        posterior_physical = np.zeros_like(posterior_normalized)
        
        # Save current parameter state to restore later
        saved_params = self.parameters.params.copy()
        
        try:
            # Convert each sample from unit cube to physical values
            for i in range(n_samples):
                # Create a copy of the cube for this sample
                cube = posterior_normalized[i].copy()
                # Use Parameters.__call__ to convert to physical values
                # This updates self.parameters.params, but we only need the conversion
                self.parameters(cube)
                # Extract physical values in the order of param_keys
                for j, key in enumerate(self.parameters.param_keys):
                    posterior_physical[i, j] = self.parameters.params[key]
        finally:
            # Restore original parameter state
            self.parameters.params = saved_params
        
        return posterior_physical
    
    def cornerplot(self):
        """
        Corner plot of posterior samples.
        """
        if self.posterior is None:
            print(f"[retrieval.py/Retrieval.cornerplot] Warning: self.posterior is None, skipping corner plot.")
            return

        # Check if we have enough samples for corner plot
        # corner.corner() requires n_samples >= n_params
        n_samples, n_params = self.posterior.shape[0], self.posterior.shape[1]
        if n_samples < n_params:
            print(f"[retrieval.py/Retrieval.cornerplot] Warning: Not enough samples ({n_samples}) for {n_params} parameters. "
                  f"Skipping corner plot. Need at least {n_params} samples.")
            return

        # Convert posterior from unit cube [0,1] to physical parameter values
        print(f"[retrieval.py/Retrieval.cornerplot] Converting posterior from unit cube to physical values...")
        posterior_physical = self._convert_posterior_to_physical(self.posterior)
        
        print(f"[retrieval.py/Retrieval.cornerplot] Creating corner plot, saving to {self.output_dir / f'{self.callback_label}corner.pdf'}")
        cornerplot(
            posterior=posterior_physical,
            param_mathtext=self.parameters.param_mathtext,
            color=self.color,
            output_path=self.output_dir,
            callback_label=self.callback_label
        )
        print(f"[retrieval.py/Retrieval.cornerplot] Corner plot saved successfully.")
    
    def plot_spectrum(self):
        """
        Plot data vs model spectrum comparison.
        """
        if self.model_flux is None:
            print(f"[retrieval.py/Retrieval.plot_spectrum] Warning: self.model_flux is None, skipping spectrum plot.")
            return
        
        print(f"[retrieval.py/Retrieval.plot_spectrum] Creating spectrum plot, saving to {self.output_dir / f'{self.callback_label}spectrum.pdf'}")
        plot_spectrum(
            data_wave=self.target.wl,
            data_flux=self.target.fl,
            model_flux=self.model_flux,
            data_err=self.target.err,
            mask=self.target.mask,
            color=self.color,
            output_path=self.output_dir,
            callback_label=self.callback_label,
            title=f"{self.target.name} Spectrum Comparison",
            residual_flag=True
        )
        print(f"[retrieval.py/Retrieval.plot_spectrum] Spectrum plot saved successfully.")
    
    def plot_tp_profile(self):
        """
        Plot temperature-pressure (TP) profile.
        """
        # Determine which model to use (self.model or temp_model)
        model_to_use = None
        if self.model is None:
            print(f"[retrieval.py/Retrieval.plot_tp_profile] Warning: self.model is None, trying bestfit_params...")
            # Try to get TP profile from bestfit parameters if model not available
            if self.bestfit_params is not None:
                print(f"[retrieval.py/Retrieval.plot_tp_profile] Creating temporary model from bestfit_params...")
                # Create temporary model to get TP profile without modifying self.parameters.params
                from copy import deepcopy
                temp_parameters = deepcopy(self.parameters)
                # Convert bestfit_params (normalized) to physical values
                temp_parameters(self.bestfit_params)
                temp_model = pRT_spectrum(
                    parameters=temp_parameters,
                    target=self.target,
                    atmosphere=self.atmosphere,
                    lbl_opacity_sampling=self.lbl_opacity_sampling,
                )
                temperature = temp_model.temperature
                pressure = temp_model.pressure
                model_to_use = temp_model  # Use temp_model for TP profile info
            else:
                print(f"[retrieval.py/Retrieval.plot_tp_profile] Warning: self.bestfit_params is also None, skipping TP profile plot.")
                return
        else:
            temperature = self.model.temperature
            pressure = self.model.pressure
            model_to_use = self.model  # Use self.model for TP profile info
        
        # For live plotting, pass history; for final plotting, don't
        tp_history = self.tp_history if self.callback_label == "live_" else None
        
        # Calculate knots error range for final plot
        knots_temperature = None
        knots_pressure = None
        knots_error_positive = None
        knots_error_negative = None
        
        if self.posterior is not None:
            # Identify TP knot parameters (T_0, T_1, T_2, ... or T0, T1, T2, ...)
            temp_knot_keys = []
            for key in self.parameters.param_keys:
                if (key.startswith('T_') and key[2:].isdigit()) or (key.startswith('T') and key[1:].isdigit() and len(key) > 1):
                    temp_knot_keys.append(key)
            
            if len(temp_knot_keys) > 0:
                # Convert posterior to physical values
                posterior_physical = self._convert_posterior_to_physical(self.posterior)
                
                # Get indices of temperature knot parameters in param_keys
                temp_knot_indices = [self.parameters.param_keys.index(key) for key in temp_knot_keys]
                
                # Get actual temperature values at knots from the model (for scatter points to match TP profile)
                # These should match the TP profile line exactly
                knots_temperature_actual = []
                for key in temp_knot_keys:
                    knots_temperature_actual.append(self.parameters.params[key])
                knots_temperature_actual = np.array(knots_temperature_actual)
                
                # Calculate percentiles (16th, 50th, 84th) for each knot (for error bars)
                knots_error_positive = []
                knots_error_negative = []
                
                for idx in temp_knot_indices:
                    knot_samples = posterior_physical[:, idx]
                    median = np.percentile(knot_samples, 50.0)
                    p16 = np.percentile(knot_samples, 16.0)
                    p84 = np.percentile(knot_samples, 84.0)
                    
                    knots_error_positive.append(p84 - median)
                    knots_error_negative.append(median - p16)
                
                knots_temperature = knots_temperature_actual  # Use actual model values, not median
                knots_error_positive = np.array(knots_error_positive)
                knots_error_negative = np.array(knots_error_negative)
                
                # Get pressure knots from TP profile
                # Try to get log_P_knots from the TP profile object (use model_to_use instead of self.model)
                if model_to_use is not None and hasattr(model_to_use, 'tp_profile') and hasattr(model_to_use.tp_profile, 'log_P_knots'):
                    log_P_knots = model_to_use.tp_profile.log_P_knots
                    if log_P_knots is not None:
                        knots_pressure = 10**np.array(log_P_knots)
                    else:
                        # If log_P_knots is None, create equally-spaced knots
                        knots_pressure = np.logspace(
                            np.log10(pressure.min()),
                            np.log10(pressure.max()),
                            num=len(knots_temperature)
                        )
                else:
                    # Fallback: create equally-spaced knots
                    knots_pressure = np.logspace(
                        np.log10(pressure.min()),
                        np.log10(pressure.max()),
                        num=len(knots_temperature)
                    )
                
                # Ensure knots are sorted by pressure (ascending: low pressure to high pressure)
                knots_temperature = knots_temperature[::-1]
                knots_error_positive = knots_error_positive[::-1]
                knots_error_negative = knots_error_negative[::-1]
                # Now sort by pressure (ascending)
                sort_idx = np.argsort(knots_pressure)
                knots_pressure = knots_pressure[sort_idx]
                knots_temperature = knots_temperature[sort_idx]
                knots_error_positive = knots_error_positive[sort_idx]
                knots_error_negative = knots_error_negative[sort_idx]
                
                print(f"[retrieval.py/Retrieval.plot_tp_profile] Found {len(temp_knot_keys)} TP knots: {temp_knot_keys}")
                print(f"[retrieval.py/Retrieval.plot_tp_profile] knots_temperature shape: {knots_temperature.shape}, knots_pressure shape: {knots_pressure.shape}")
                print(f"[retrieval.py/Retrieval.plot_tp_profile] knots_error_positive shape: {knots_error_positive.shape}, knots_error_negative shape: {knots_error_negative.shape}")
            else:
                print(f"[retrieval.py/Retrieval.plot_tp_profile] Warning: No TP knot parameters found in param_keys.")
        else:
            print(f"[retrieval.py/Retrieval.plot_tp_profile] Warning: self.posterior is None, cannot calculate knots error range.")
        
        # Get interpolation mode from TP profile (for consistent interpolation in plotting)
        interp_mode = 'cubic'  # default
        if model_to_use is not None and hasattr(model_to_use, 'tp_profile') and hasattr(model_to_use.tp_profile, 'interp_mode'):
            interp_mode = model_to_use.tp_profile.interp_mode
        
        print(f"[retrieval.py/Retrieval.plot_tp_profile] Creating TP profile plot, saving to {self.output_dir / f'{self.callback_label}tp_profile.pdf'}")
        plot_tp_profile(
            temperature=temperature,
            pressure=pressure,
            color=self.color,
            output_path=self.output_dir,
            callback_label=self.callback_label,
            title=f"{self.target.name} TP Profile",
            tp_history=tp_history,
            knots_temperature=knots_temperature,
            knots_pressure=knots_pressure,
            knots_error_positive=knots_error_positive,
            knots_error_negative=knots_error_negative,
            interp_mode=interp_mode
        )
        print(f"[retrieval.py/Retrieval.plot_tp_profile] TP profile plot saved successfully.")
    
    # ===== Helper methods =====
    def _create_pressure_from_tp_kwargs(self):
        """
        Create pressure grid from TP_kwargs.
        
        This method implements the same logic as TP_profile._set_pressures() to ensure
        consistency between Retrieval, Radtrans, TP_profile, and Chemistry classes.
        
        Priority:
        1. If 'pressure' is explicitly provided in TP_kwargs, use it
        2. If 'log_P_range' and 'n_atm_layers' are provided, create pressure from them
        3. Otherwise, use default values: log_P_range=(-5., 2.), n_atm_layers=70
        
        Returns:
            np.ndarray: Pressure grid [bar], sorted in ascending order
        """
        TP_kwargs = self.parameters.TP_kwargs.copy() if hasattr(self.parameters, 'TP_kwargs') and self.parameters.TP_kwargs else {}
        
        if 'pressure' in TP_kwargs and TP_kwargs['pressure'] is not None:
            # If pressure is explicitly provided in TP_kwargs, use it
            pressure = np.asarray(TP_kwargs['pressure'])
            pressure = np.sort(pressure)
        elif 'log_P_range' in TP_kwargs and 'n_atm_layers' in TP_kwargs:
            # Create pressure from log_P_range and n_atm_layers
            log_P_range = TP_kwargs['log_P_range']
            n_atm_layers = TP_kwargs['n_atm_layers']
            pressure = np.logspace(log_P_range[0], log_P_range[1], n_atm_layers)
            pressure = np.sort(pressure)
        else:
            # Default pressure grid (consistent with make_spectrum.py default)
            warnings.warn(
                "No pressure grid specified in TP_kwargs. Using default: "
                f"log_P_range=(-5., 2.), n_atm_layers=70"
            )
            pressure = np.logspace(-5., 2., 70)
            pressure = np.sort(pressure)
        
        return pressure

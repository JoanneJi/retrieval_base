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
from utils.plotting import cornerplot, plot_spectrum, plot_tp_profile, plot_vmr_profile

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
    def __init__(self, parameters, target, N_live_points=200, evidence_tolerance=0.5, output_base=OUTPUT_DIR / "retrievals", output_subdir=None, lbl_opacity_sampling=3, n_atm_layers=50, wl_pad=7, redo_atmosphere=True, normalize=True, normalize_method='simplistic_normalization', star_mode=False):
        self.parameters = parameters
        self.target = target
        self.normalize = normalize
        self.normalize_method = normalize_method
        
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
        self.species, self.species_colors = get_species_from_params(param_dict=all_param_keys, species_info_path=species_info_path)
        
        # For equilibrium chemistry, if line_species is empty (no log_* params),
        # try to get from chemistry_kwargs and convert to pRT names
        chem_mode = chemistry_kwargs.get('chem_mode', 'free')
        if chem_mode in ['equilibrium', 'fastchem_live', 'fastchem_grid'] or len(self.species) == 0:
            if 'line_species' in chemistry_kwargs:
                line_species = chemistry_kwargs['line_species']
                if line_species and isinstance(line_species, list):
                    # Convert species_info names (e.g., 'H2O', '12CO') to pRT names (e.g., '1H2-16O', '12C-16O')
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
                        for species in line_species:
                            # Check if it's already a pRT name (contains '-' and numbers, e.g., '12C-16O')
                            # or if it's a species_info name (e.g., 'H2O', '12CO')
                            if '-' in species and any(c.isdigit() for c in species):
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
                        self.species = line_species_pRT
                        print(f"[retrieval.py/Retrieval.__init__] Converted line_species from chemistry_kwargs to pRT names: {self.species}")
                    else:
                        warnings.warn(f"species_info.csv not found at {species_info_path}, using line_species as provided: {line_species}")
                        self.species = line_species
                else:
                    self.species = line_species if line_species else []

        # Create atmosphere objects
        # Check if target is in chips_mode
        chips_mode = getattr(self.target, 'chips_mode', False)
        if chips_mode:
            # Multi-chip mode: pass wave_ranges_chips
            self.atmosphere = setup_radtrans_atmosphere(
                species=self.species,
                target_wavelengths=self.target.wl,  # list of arrays
                pressure=self.pressure,
                lbl_opacity_sampling=self.lbl_opacity_sampling,
                wl_pad=self.wl_pad,
                cache_file=self.output_dir / "atmosphere_objects.pickle",
                redo=redo_atmosphere,
                chips_mode=True,
                wave_ranges_chips=self.target.wave_ranges_chips,
                star_mode=star_mode
            )
        else:
            # Single spectrum mode (backward compatible)
            self.atmosphere = setup_radtrans_atmosphere(
                species=self.species,
                target_wavelengths=self.target.wl,
                pressure=self.pressure,
                lbl_opacity_sampling=self.lbl_opacity_sampling,
                wl_pad=self.wl_pad,
                cache_file=self.output_dir / "atmosphere_objects.pickle",
                redo=redo_atmosphere,
                star_mode=star_mode
            )

        # Likelihood components
        # Use err_flat which is always available (1D array in both chips_mode and non-chips_mode)
        self.cov = Covariance(err=self.target.err_flat[self.target.mask])
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
                    f.write(f"[PMN_lnL] After conversion: T_0={self.parameters.params.get('T_0')}, T_1={self.parameters.params.get('T_1')}, T_2={self.parameters.params.get('T_2')}, T_3={self.parameters.params.get('T_3')}, T_4={self.parameters.params.get('T_4')}\n")
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
                normalize=self.normalize,
                normalize_method=self.normalize_method,
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
            self.plot_vmr_profile()
            
            # Save VMR profile (live)
            if self.model is not None:
                self._save_vmr_profile()
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
            n_iter_before_update=10,  # call back every 25 iterations
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
            normalize=self.normalize,
            normalize_method=self.normalize_method,
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
    def run_retrieval(self, resume=True):
        """
        Full retrieval pipeline.
        
        Args:
            resume (bool): If True, resume from previous run. If False, start fresh.
                          Default: True
        """
        start_time = time.time()
        
        # Check if retrieval has already converged (post_equal_weights.dat exists)
        # If converged and resume=True, skip sampling to avoid changing posterior
        data_file = self.output_dir / f"{self.prefix}post_equal_weights.dat"
        if resume and data_file.exists():
            try:
                # Try to read the file to verify it's complete
                test_data = np.loadtxt(data_file)
                if test_data.size > 0 and len(test_data.shape) >= 1:
                    print(f"\n[retrieval.py/Retrieval.run_retrieval] Found existing posterior file. "
                          f"Retrieval appears to have converged. Skipping sampling to preserve existing results.\n")
                    skip_sampling = True
                else:
                    skip_sampling = False
            except (ValueError, IOError, OSError) as e:
                # File exists but may be incomplete, continue with sampling
                print(f"\n[retrieval.py/Retrieval.run_retrieval] Found posterior file but it appears incomplete. "
                      f"Continuing with sampling.\n")
                skip_sampling = False
        else:
            skip_sampling = False
        
        if not skip_sampling:
            self.PMN_run(resume=resume)
            print(f"\n[retrieval.py/Retrieval.run_retrieval] ----- MultiNest run completed. -----\n")
        else:
            print(f"\n[retrieval.py/Retrieval.run_retrieval] ----- Skipping MultiNest sampling (using existing results). -----\n")
        
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
            # Use *_flat arrays which are always available (1D arrays in both chips_mode and non-chips_mode)
            wl = self.target.wl_flat
            fl = self.target.fl_flat
            err = self.target.err_flat
            model_fl = np.asarray(self.model_flux).flatten()
            residuals = np.asarray(fl - model_fl).flatten()
            np.savetxt(self.output_dir / f"{self.callback_label}model_spectrum.dat", np.column_stack((wl, fl, err, model_fl, residuals)), header='Wavelength (nm) Data_Flux Data_Err Model_Flux Residuals', fmt='%.6f %.6f %.6f %.6f %.6f')
            print(f"[retrieval.py/Retrieval.run_retrieval] Model spectrum saved to {self.output_dir / f'{self.callback_label}model_spectrum.dat'}")
            
            # ----- save TP profile to a dat file (only for final plot, not live) -----
            if self.callback_label == "final_" and self.model is not None:
                self._save_tp_profile()
            
            # ----- save VMR profile to a dat file (for both live and final) -----
            if self.model is not None:
                self._save_vmr_profile()

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
        
        try:
            self.plot_vmr_profile()
            print(f"[retrieval.py/Retrieval.run_retrieval] plot_vmr_profile completed.")
        except Exception as e:
            print(f"[retrieval.py/Retrieval.run_retrieval] Error in plot_vmr_profile: {e}")
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
        # Use *_flat arrays which are always available (1D arrays in both chips_mode and non-chips_mode)
        plot_spectrum(
            data_wave=self.target.wl_flat,
            data_flux=self.target.fl_flat,
            model_flux=self.model_flux,
            data_err=self.target.err_flat,
            mask=self.target.mask,
            color=self.color,
            output_path=self.output_dir,
            callback_label=self.callback_label,
            # title=f"{self.target.name} Spectrum Comparison",
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
                    normalize=self.normalize,
                    normalize_method=self.normalize_method,
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
                knots_error_1sigma_positive = []
                knots_error_1sigma_negative = []
                knots_error_2sigma_positive = []
                knots_error_2sigma_negative = []
                
                for idx in temp_knot_indices:
                    knot_samples = posterior_physical[:, idx]
                    median = np.percentile(knot_samples, 50.0)
                    # 1sigma
                    p16 = np.percentile(knot_samples, 16.0)
                    p84 = np.percentile(knot_samples, 84.0)
                    # 2sigma
                    p2_5 = np.percentile(knot_samples, 2.5)
                    p97_5 = np.percentile(knot_samples, 97.5)

                    knots_error_1sigma_positive.append(p84 - median)
                    knots_error_1sigma_negative.append(median - p16)
                    knots_error_2sigma_positive.append(p97_5 - median)
                    knots_error_2sigma_negative.append(median - p2_5)
                
                knots_temperature = knots_temperature_actual  # Use actual model values, not median
                knots_error_1sigma_positive = np.array(knots_error_1sigma_positive)
                knots_error_1sigma_negative = np.array(knots_error_1sigma_negative)
                knots_error_2sigma_positive = np.array(knots_error_2sigma_positive)
                knots_error_2sigma_negative = np.array(knots_error_2sigma_negative)
                
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
                knots_error_1sigma_positive = knots_error_1sigma_positive[::-1]
                knots_error_1sigma_negative = knots_error_1sigma_negative[::-1]
                knots_error_2sigma_positive = knots_error_2sigma_positive[::-1]
                knots_error_2sigma_negative = knots_error_2sigma_negative[::-1]
                # Now sort by pressure (ascending)
                sort_idx = np.argsort(knots_pressure)
                knots_pressure = knots_pressure[sort_idx]
                knots_temperature = knots_temperature[sort_idx]
                knots_error_1sigma_positive = knots_error_1sigma_positive[sort_idx]
                knots_error_1sigma_negative = knots_error_1sigma_negative[sort_idx]
                knots_error_2sigma_positive = knots_error_2sigma_positive[sort_idx]
                knots_error_2sigma_negative = knots_error_2sigma_negative[sort_idx]
                
                print(f"[retrieval.py/Retrieval.plot_tp_profile] Found {len(temp_knot_keys)} TP knots: {temp_knot_keys}")
                print(f"[retrieval.py/Retrieval.plot_tp_profile] knots_temperature shape: {knots_temperature.shape}, knots_pressure shape: {knots_pressure.shape}")
                print(f"[retrieval.py/Retrieval.plot_tp_profile] knots_error_1sigma_positive shape: {knots_error_1sigma_positive.shape}, knots_error_1sigma_negative shape: {knots_error_1sigma_negative.shape}, knots_error_2sigma_positive shape: {knots_error_2sigma_positive.shape}, knots_error_2sigma_negative shape: {knots_error_2sigma_negative.shape}")
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
            knots_error_positive=knots_error_2sigma_positive,
            knots_error_negative=knots_error_2sigma_negative,
            interp_mode=interp_mode
        )
        print(f"[retrieval.py/Retrieval.plot_tp_profile] TP profile plot saved successfully.")
    
    # ===== Helper methods =====
    def _save_tp_profile(self):
        """
        Save TP profile and error bars to final_tp.dat file.
        
        The file contains:
        - Pressure [bar]
        - Temperature [K] (median)
        - Temperature error positive [K] (84th percentile - median)
        - Temperature error negative [K] (median - 16th percentile)
        """
        if self.model is None:
            print(f"[retrieval.py/Retrieval._save_tp_profile] Warning: self.model is None, cannot save TP profile.")
            return
        
        if self.posterior is None:
            print(f"[retrieval.py/Retrieval._save_tp_profile] Warning: self.posterior is None, cannot calculate TP profile errors.")
            return
        
        # Get TP profile from model
        pressure = self.model.pressure.copy()
        temperature_median = self.model.temperature.copy()
        
        # Calculate temperature errors from posterior
        # We need to compute TP profiles for all posterior samples to get error bars
        n_atm_layers = len(pressure)
        n_samples = self.posterior.shape[0]
        
        # Limit number of samples for efficiency (use at most 1000 samples)
        max_samples = min(1000, n_samples)
        sample_indices = np.linspace(0, n_samples - 1, max_samples, dtype=int)
        
        print(f"[retrieval.py/Retrieval._save_tp_profile] Computing TP profiles for {max_samples} posterior samples to estimate errors...")
        
        # Store temperature profiles for all samples
        temperature_samples = np.zeros((max_samples, n_atm_layers))
        
        # Save current parameter state
        saved_params = self.parameters.params.copy()
        
        try:
            # Convert posterior to physical values and compute TP profiles
            for idx, sample_idx in enumerate(sample_indices):
                if (idx + 1) % 100 == 0:
                    print(f"[retrieval.py/Retrieval._save_tp_profile] Processing sample {idx + 1}/{max_samples}...")
                
                # Get normalized sample from posterior
                cube = self.posterior[sample_idx].copy()
                
                # Convert to physical parameters
                self.parameters(cube)
                
                # Create temporary model to get TP profile
                temp_model = pRT_spectrum(
                    parameters=self.parameters,
                    target=self.target,
                    atmosphere=self.atmosphere,
                    lbl_opacity_sampling=self.lbl_opacity_sampling,
                    normalize=self.normalize,
                    normalize_method=self.normalize_method,
                )
                
                # Store temperature profile
                temperature_samples[idx] = temp_model.temperature.copy()
        finally:
            # Restore original parameter state
            self.parameters.params = saved_params
        
        # Calculate percentiles for each pressure layer
        temperature_error_1sigma_positive = np.zeros(n_atm_layers)
        temperature_error_1sigma_negative = np.zeros(n_atm_layers)
        temperature_error_2sigma_positive = np.zeros(n_atm_layers)
        temperature_error_2sigma_negative = np.zeros(n_atm_layers)
        
        for i in range(n_atm_layers):
            temp_values = temperature_samples[:, i]
            median = np.percentile(temp_values, 50.0)
            # 1sigma
            p16 = np.percentile(temp_values, 16.0)
            p84 = np.percentile(temp_values, 84.0)
            # 2sigma
            p2_5 = np.percentile(temp_values, 2.5)
            p97_5 = np.percentile(temp_values, 97.5)

            temperature_error_1sigma_positive[i] = p84 - median
            temperature_error_1sigma_negative[i] = median - p16
            temperature_error_2sigma_positive[i] = p97_5 - median
            temperature_error_2sigma_negative[i] = median - p2_5
        
        # Save to file
        output_file = self.output_dir / "final_tp.dat"
        header = 'Pressure (bar) Temperature (K) Temperature_error_-1sigma (K) Temperature_error_+1sigma (K) Temperature_error_-2sigma (K) Temperature_error_+2sigma (K)'
        data = np.column_stack((
            pressure,
            temperature_median,
            temperature_error_1sigma_positive,
            temperature_error_1sigma_negative,
            temperature_error_2sigma_positive,
            temperature_error_2sigma_negative
        ))
        np.savetxt(output_file, data, header=header, fmt='%.6e %.2f %.2f %.2f %.2f %.2f')
        print(f"[retrieval.py/Retrieval._save_tp_profile] TP profile saved to {output_file}")
    
    def _compute_vmr_errors(self):
        """
        Compute VMR error bars (95% confidence interval) from posterior samples.
        
        Returns:
            dict: Dictionary mapping species names to error arrays.
                Each error array has shape (n_layers, 2) where columns are [lower, upper] errors.
        """
        if self.posterior is None:
            print(f"[retrieval.py/Retrieval._compute_vmr_errors] Warning: self.posterior is None, cannot calculate VMR errors.")
            return None
        
        if self.model is None:
            print(f"[retrieval.py/Retrieval._compute_vmr_errors] Warning: self.model is None, cannot get VMR structure.")
            return None
        
        # Get pressure grid from model
        pressure = self.model.pressure.copy()
        n_atm_layers = len(pressure)
        n_samples = self.posterior.shape[0]
        
        # Limit number of samples for efficiency (use at most 1000 samples)
        max_samples = min(1000, n_samples)
        sample_indices = np.linspace(0, n_samples - 1, max_samples, dtype=int)
        
        print(f"[retrieval.py/Retrieval._compute_vmr_errors] Computing VMR profiles for {max_samples} posterior samples to estimate errors...")
        
        # Get species list from current model
        if not hasattr(self.model, 'chemistry') or not hasattr(self.model.chemistry, 'VMRs'):
            print(f"[retrieval.py/Retrieval._compute_vmr_errors] Warning: Cannot access VMRs from model.chemistry.")
            return None
        
        current_vmrs = self.model.chemistry.VMRs
        if current_vmrs is None or not isinstance(current_vmrs, dict):
            print(f"[retrieval.py/Retrieval._compute_vmr_errors] Warning: VMRs is None or not a dict.")
            return None
        
        species_list = list(current_vmrs.keys())
        # Filter out H2, He (MMW will be handled separately from chemistry.MMW attribute)
        species_list = [s for s in species_list if s not in ['H2', 'He']]
        
        # Check if MMW should be computed (from chemistry.MMW attribute)
        include_mmw = False
        if hasattr(self.model.chemistry, 'MMW') and self.model.chemistry.MMW is not None:
            include_mmw = True
        
        if len(species_list) == 0 and not include_mmw:
            print(f"[retrieval.py/Retrieval._compute_vmr_errors] Warning: No species found in VMRs (excluding H2, He) and no MMW available.")
            return None
        
        # Store VMR profiles for all samples
        vmr_samples = {species: np.zeros((max_samples, n_atm_layers)) for species in species_list}
        # Also store MMW samples if needed
        if include_mmw:
            vmr_samples['MMW'] = np.zeros((max_samples, n_atm_layers))
        
        # Save current parameter state
        saved_params = self.parameters.params.copy()
        
        try:
            # Convert posterior to physical values and compute VMR profiles
            for idx, sample_idx in enumerate(sample_indices):
                if (idx + 1) % 100 == 0:
                    print(f"[retrieval.py/Retrieval._compute_vmr_errors] Processing sample {idx + 1}/{max_samples}...")
                
                # Get normalized sample from posterior
                cube = self.posterior[sample_idx].copy()
                
                # Convert to physical parameters
                self.parameters(cube)
                
                # Create temporary model to get VMR profile
                temp_model = pRT_spectrum(
                    parameters=self.parameters,
                    target=self.target,
                    atmosphere=self.atmosphere,
                    lbl_opacity_sampling=self.lbl_opacity_sampling,
                    normalize=self.normalize,
                    normalize_method=self.normalize_method,
                )
                
                # Get VMRs from chemistry object
                if hasattr(temp_model, 'chemistry') and hasattr(temp_model.chemistry, 'VMRs'):
                    temp_vmrs = temp_model.chemistry.VMRs
                    if temp_vmrs is not None and isinstance(temp_vmrs, dict):
                        for species in species_list:
                            if species in temp_vmrs:
                                vmr_samples[species][idx] = temp_vmrs[species].copy()
                
                # Get MMW from chemistry object if needed
                if include_mmw and hasattr(temp_model, 'chemistry') and hasattr(temp_model.chemistry, 'MMW'):
                    if temp_model.chemistry.MMW is not None:
                        vmr_samples['MMW'][idx] = temp_model.chemistry.MMW.copy()
        finally:
            # Restore original parameter state
            self.parameters.params = saved_params
        
        # Calculate percentiles (2.5th, 50th, 97.5th) for each pressure layer (95% CI)
        vmr_errors = {}
        vmr_medians = {}
        
        # Process regular species
        for species in species_list:
            vmr_error_positive = np.zeros(n_atm_layers)
            vmr_error_negative = np.zeros(n_atm_layers)
            vmr_median = np.zeros(n_atm_layers)
            
            for i in range(n_atm_layers):
                vmr_values = vmr_samples[species][:, i]
                # Remove any invalid values (NaN, inf, negative)
                vmr_values = vmr_values[np.isfinite(vmr_values) & (vmr_values > 0)]
                
                if len(vmr_values) > 0:
                    median = np.percentile(vmr_values, 50.0)
                    p2_5 = np.percentile(vmr_values, 2.5)
                    p97_5 = np.percentile(vmr_values, 97.5)
                    
                    vmr_median[i] = median
                    vmr_error_positive[i] = p97_5 - median
                    vmr_error_negative[i] = median - p2_5
                else:
                    # If all values are invalid, use zeros
                    vmr_median[i] = 0.0
                    vmr_error_positive[i] = 0.0
                    vmr_error_negative[i] = 0.0
            
            # Store errors as [lower, upper] for each layer
            vmr_errors[species] = np.column_stack([vmr_error_negative, vmr_error_positive])
            vmr_medians[species] = vmr_median
        
        # Process MMW if included
        if include_mmw:
            mmw_error_positive = np.zeros(n_atm_layers)
            mmw_error_negative = np.zeros(n_atm_layers)
            mmw_median = np.zeros(n_atm_layers)
            
            for i in range(n_atm_layers):
                mmw_values = vmr_samples['MMW'][:, i]
                # Remove any invalid values (NaN, inf)
                mmw_values = mmw_values[np.isfinite(mmw_values) & (mmw_values > 0)]
                
                if len(mmw_values) > 0:
                    median = np.percentile(mmw_values, 50.0)
                    p2_5 = np.percentile(mmw_values, 2.5)
                    p97_5 = np.percentile(mmw_values, 97.5)
                    
                    mmw_median[i] = median
                    mmw_error_positive[i] = p97_5 - median
                    mmw_error_negative[i] = median - p2_5
                else:
                    # If all values are invalid, use zeros
                    mmw_median[i] = 0.0
                    mmw_error_positive[i] = 0.0
                    mmw_error_negative[i] = 0.0
            
            # Store errors as [lower, upper] for each layer
            vmr_errors['MMW'] = np.column_stack([mmw_error_negative, mmw_error_positive])
            vmr_medians['MMW'] = mmw_median
        
        n_species_total = len(species_list) + (1 if include_mmw else 0)
        print(f"[retrieval.py/Retrieval._compute_vmr_errors] VMR errors computed for {n_species_total} species (including MMW).")
        return vmr_errors, vmr_medians
    
    def _save_vmr_profile(self):
        """
        Save VMR profiles and error bars to {callback_label}vmr.dat file.
        
        For live plots: saves current best-fit VMRs without errors.
        For final plots: saves median VMRs with 95% confidence intervals.
        
        The file contains:
        - Pressure [bar]
        - For each species: VMR (median or best-fit), VMR_error_lower, VMR_error_upper (final only)
        """
        if self.model is None:
            print(f"[retrieval.py/Retrieval._save_vmr_profile] Warning: self.model is None, cannot save VMR profile.")
            return
        
        # Get VMRs from current model
        if not hasattr(self.model, 'chemistry') or not hasattr(self.model.chemistry, 'VMRs'):
            print(f"[retrieval.py/Retrieval._save_vmr_profile] Warning: Cannot access VMRs from model.chemistry.")
            return
        
        vmrs = self.model.chemistry.VMRs
        if vmrs is None or not isinstance(vmrs, dict):
            print(f"[retrieval.py/Retrieval._save_vmr_profile] Warning: VMRs is None or not a dict.")
            return
        
        # Get pressure grid
        pressure = self.model.pressure.copy()
        
        # Compute errors only for final plot (requires posterior)
        vmr_errors = {}
        vmr_medians = {}
        if self.callback_label == "final_" and self.posterior is not None:
            result = self._compute_vmr_errors()
            if result is not None:
                vmr_errors, vmr_medians = result
            else:
                print(f"[retrieval.py/Retrieval._save_vmr_profile] Warning: Could not compute VMR errors. Saving without errors.")
        
        # Filter out H2, He --> MMW should be saved in the vmr.dat file
        species_list = [s for s in vmrs.keys() if s not in ['H2', 'He']]
        
        # Check if MMW should be added (from chemistry.MMW attribute, not from VMRs dict)
        include_mmw = False
        mmw_value = None
        mmw_median = None
        mmw_errors = None
        if hasattr(self.model.chemistry, 'MMW') and self.model.chemistry.MMW is not None:
            include_mmw = True
            mmw_value = self.model.chemistry.MMW.copy()
        
        if len(species_list) == 0 and not include_mmw:
            print(f"[retrieval.py/Retrieval._save_vmr_profile] Warning: No species found in VMRs (excluding H2, He) and no MMW available.")
            return
        
        # Prepare data for saving
        # Format: Pressure, then for each species: VMR (best-fit for live, median for final), VMR_error_lower, VMR_error_upper (final only)
        data_columns = [pressure]
        header_parts = ['Pressure (bar)']
        
        for species in species_list:
            if species in vmrs:
                # Use median from posterior if available (final), otherwise use current model value (live)
                if species in vmr_medians:
                    vmr_value = vmr_medians[species]
                else:
                    vmr_value = vmrs[species]
                
                data_columns.append(vmr_value)
                header_parts.append(f'{species}_VMR')
                
                # Add errors only for final plot
                if self.callback_label == "final_" and species in vmr_errors:
                    errors = vmr_errors[species]
                    data_columns.append(errors[:, 0])  # lower error
                    data_columns.append(errors[:, 1])  # upper error
                    header_parts.append(f'{species}_VMR_error_lower')
                    header_parts.append(f'{species}_VMR_error_upper')
        
        # Add MMW if available
        if include_mmw:
            # Use median from posterior if available (final), otherwise use current model value (live)
            if 'MMW' in vmr_medians:
                mmw_value = vmr_medians['MMW']
            elif mmw_value is None:
                mmw_value = self.model.chemistry.MMW.copy()
            
            data_columns.append(mmw_value)
            header_parts.append('MMW')
            
            # Add errors only for final plot
            if self.callback_label == "final_" and 'MMW' in vmr_errors:
                mmw_errors = vmr_errors['MMW']
                data_columns.append(mmw_errors[:, 0])  # lower error
                data_columns.append(mmw_errors[:, 1])  # upper error
                header_parts.append('MMW_error_lower')
                header_parts.append('MMW_error_upper')
        
        # Save to file
        output_file = self.output_dir / f"{self.callback_label}vmr.dat"
        header = ' '.join(header_parts)
        data = np.column_stack(data_columns)
        
        # Create format string
        fmt_parts = ['%.6e']  # Pressure
        for species in species_list:
            fmt_parts.append('%.6e')  # VMR
            if self.callback_label == "final_" and species in vmr_errors:
                fmt_parts.append('%.6e')  # lower error
                fmt_parts.append('%.6e')  # upper error
        # Add format for MMW if included
        if include_mmw:
            fmt_parts.append('%.6e')  # MMW
            if self.callback_label == "final_" and mmw_errors is not None:
                fmt_parts.append('%.6e')  # lower error
                fmt_parts.append('%.6e')  # upper error
        fmt = ' '.join(fmt_parts)
        
        np.savetxt(output_file, data, header=header, fmt=fmt)
        print(f"[retrieval.py/Retrieval._save_vmr_profile] VMR profile saved to {output_file}")
        
        # Also save as pickle for easier loading
        vmr_dict = {
            'pressure': pressure,
            'vmrs': {species: vmrs[species] for species in species_list},
            'vmr_medians': vmr_medians if vmr_medians else {},
            'vmr_errors': vmr_errors if vmr_errors else {}
        }
        # Add MMW to pickle if available
        if include_mmw:
            vmr_dict['mmw'] = mmw_value
            if 'MMW' in vmr_medians:
                vmr_dict['mmw_median'] = vmr_medians['MMW']
            if 'MMW' in vmr_errors:
                vmr_dict['mmw_errors'] = vmr_errors['MMW']
        import pickle
        with open(self.output_dir / f"{self.callback_label}vmr.pkl", "wb") as f:
            pickle.dump(vmr_dict, f)
        print(f"[retrieval.py/Retrieval._save_vmr_profile] VMR dictionary saved to {self.output_dir / f'{self.callback_label}vmr.pkl'}")
    
    def plot_vmr_profile(self):
        """
        Plot VMR profiles with error bars.
        """
        # Get chemistry object and VMRs
        chemistry = None
        vmrs = None
        pressure = None
        
        if self.model is None:
            print(f"[retrieval.py/Retrieval.plot_vmr_profile] Warning: self.model is None, trying bestfit_params...")
            # Try to get VMR profile from bestfit parameters if model not available
            if self.bestfit_params is not None:
                print(f"[retrieval.py/Retrieval.plot_vmr_profile] Creating temporary model from bestfit_params...")
                from copy import deepcopy
                temp_parameters = deepcopy(self.parameters)
                temp_parameters(self.bestfit_params)
                temp_model = pRT_spectrum(
                    parameters=temp_parameters,
                    target=self.target,
                    atmosphere=self.atmosphere,
                    lbl_opacity_sampling=self.lbl_opacity_sampling,
                    normalize=self.normalize,
                    normalize_method=self.normalize_method,
                )
                vmrs = temp_model.chemistry.VMRs if hasattr(temp_model, 'chemistry') and hasattr(temp_model.chemistry, 'VMRs') else None
                pressure = temp_model.pressure
                chemistry = temp_model.chemistry if hasattr(temp_model, 'chemistry') else None
            else:
                print(f"[retrieval.py/Retrieval.plot_vmr_profile] Warning: self.bestfit_params is also None, skipping VMR profile plot.")
                return
        else:
            vmrs = self.model.chemistry.VMRs if hasattr(self.model, 'chemistry') and hasattr(self.model.chemistry, 'VMRs') else None
            pressure = self.model.pressure
            chemistry = self.model.chemistry if hasattr(self.model, 'chemistry') else None
        
        if vmrs is None or not isinstance(vmrs, dict):
            print(f"[retrieval.py/Retrieval.plot_vmr_profile] Warning: VMRs is None or not a dict, skipping VMR profile plot.")
            return
        
        # Convert species_colors from pRT_name keys to species_info index keys
        # vmrs_dict uses species_info index names (e.g., '12CO'), but species_colors uses pRT_name (e.g., '12C-16O__HITEMP')
        species_colors_converted = {}
        if self.species_colors is not None and chemistry is not None and hasattr(chemistry, 'species_info'):
            for species_name in vmrs.keys():
                # Skip H2, He, MMW
                if species_name in ['H2', 'He', 'MMW']:
                    continue
                # Try to get pRT_name for this species
                try:
                    if species_name in chemistry.species_info.index:
                        pRT_name = chemistry.read_species_info(species_name, 'pRT_name')
                        # Look up color using pRT_name
                        if pRT_name in self.species_colors:
                            species_colors_converted[species_name] = self.species_colors[pRT_name]
                        else:
                            # If not found, try to get color directly from species_info
                            try:
                                color = chemistry.read_species_info(species_name, 'color')
                                species_colors_converted[species_name] = str(color)
                            except (ValueError, KeyError):
                                pass  # Will use default color
                except (ValueError, KeyError):
                    pass  # Species not in species_info, will use default color
        elif self.species_colors is not None:
            # Fallback: try direct lookup (in case keys already match)
            species_colors_converted = self.species_colors
        
        # Compute errors if posterior is available (only for final plot)
        vmr_error_dict = None
        if self.posterior is not None and self.callback_label == "final_":
            print(f"[retrieval.py/Retrieval.plot_vmr_profile] Computing VMR errors from posterior...")
            result = self._compute_vmr_errors()
            if result is not None:
                vmr_error_dict, vmr_medians = result
                # Use medians instead of current model values for final plot
                for species in vmr_medians:
                    if species in vmrs:
                        vmrs[species] = vmr_medians[species]
        
        print(f"[retrieval.py/Retrieval.plot_vmr_profile] Creating VMR profile plot, saving to {self.output_dir / f'{self.callback_label}vmr_profile.pdf'}")
        plot_vmr_profile(
            vmrs_dict=vmrs,
            pressure=pressure,
            output_path=self.output_dir,
            callback_label=self.callback_label,
            title=f"{self.target.name} VMR Profile",
            vmr_error_dict=vmr_error_dict,
            species_colors=species_colors_converted if species_colors_converted else None
        )
        print(f"[retrieval.py/Retrieval.plot_vmr_profile] VMR profile plot saved successfully.")
    
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

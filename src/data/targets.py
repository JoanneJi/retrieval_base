"""
Define the target
"""

from numpy._typing._array_like import NDArray


from typing import Any


import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u


class Target:
    """
    Target object: stores spectrum and basic target information.
    Supports both 1D flattened spectra and multi-chip spectra (list of arrays).
    No instrument-specific logic, no normalization or reshaping.
    """

    def __init__(
        self,
        wl,
        fl,
        err,
        name,
        JD,
        ra=None,
        dec=None,
        color="limegreen",
        chips_mode=False,
        chips_per_order=None,
    ):
        """
        Initialize Target object.
        
        Args:
            wl: wavelength array(s). Can be:
                - 1D array for flattened spectrum
                - list of 1D arrays for multi-chip spectrum
            fl: flux array(s), same structure as wl
            err: error array(s), same structure as wl
            name: target name
            JD: Julian Date
            ra, dec: right ascension and declination (optional)
            color: color for plotting
            chips_mode: if True, expects wl/fl/err as lists of arrays (one per chip)
            chips_per_order: (optional) number of chips per order, if the chips
                are grouped by diffraction order (e.g., 3 detectors per order
                for CRIRES+). Used for order-level operations such as shared
                normalization patterns.
        """
        self.chips_mode = chips_mode
        
        if chips_mode:
            # Multi-chip mode: wl, fl, err are lists of arrays
            if not isinstance(wl, list) or not isinstance(fl, list) or not isinstance(err, list):
                raise ValueError("In chips_mode, wl, fl, err must be lists of arrays")
            if not (len(wl) == len(fl) == len(err)):
                raise ValueError("In chips_mode, wl, fl, err must have the same number of chips")
            
            self.wl = [np.asarray(w) for w in wl]
            self.fl = [np.asarray(f) for f in fl]
            self.err = [np.asarray(e) for e in err]
            
            # Check that each chip has matching shapes
            for i, (w, f, e) in enumerate(zip(self.wl, self.fl, self.err)):
                if not (w.shape == f.shape == e.shape):
                    raise ValueError(f"Chip {i}: wl, fl, err must have the same shape")
                if w.ndim != 1:
                    raise ValueError(f"Chip {i}: arrays must be 1D")
            
            # Calculate wave_ranges_chips
            self.wave_ranges_chips = np.array([[w.min(), w.max()] for w in self.wl])
            self.n_chips = len(self.wl)
            # Optional: number of chips per order (e.g. 3 detectors per order)
            self.chips_per_order = chips_per_order
            
            # Create flattened versions for backward compatibility
            self.wl_flat = np.concatenate(self.wl)
            self.fl_flat = np.concatenate(self.fl)
            self.err_flat = np.concatenate(self.err)
            
            # Mask for valid pixels (flattened)
            self.mask = np.isfinite(self.fl_flat)
        else:
            # Original 1D mode
            self.wl = np.asarray(wl)
            self.fl = np.asarray(fl)
            self.err = np.asarray(err)
            # check shapes
            if not (self.wl.shape == self.fl.shape == self.err.shape):
                raise ValueError("wl, fl, and err must have the same shape")
            # check 1D
            if self.wl.ndim != 1:
                raise ValueError("Target expects 1D flattened spectra (or use chips_mode=True)")
            
            # For backward compatibility, create flat versions
            self.wl_flat = self.wl
            self.fl_flat = self.fl
            self.err_flat = self.err
            
            # Mask for valid pixels
            self.mask = np.isfinite(self.fl)
            
            # Single chip mode
            self.n_chips = 1
            self.wave_ranges_chips = np.array([[self.wl.min(), self.wl.max()]])

        # ----- Target information -----
        self.name = name
        self.JD = JD
        self.color = color
        # RA/DEC handling: user can provide manually or resolve via name
        if ra is not None and dec is not None:
            self.ra = ra
            self.dec = dec
            self.coords = SkyCoord(self.ra, self.dec, frame='icrs')
        else:
            try:
                # Attempt to resolve from target name
                self.coords = SkyCoord.from_name(self.name, frame='icrs')
                self.ra  = self.coords.ra.to_string(unit=u.hour, sep='hms', precision=10)
                self.dec = self.coords.dec.to_string(unit=u.deg, sep='dms', precision=10)
            except Exception as e:
                raise ValueError(
                    f"Could not resolve target name '{self.name}' automatically. "
                    f"Please provide ra and dec manually. Error: {e}"
                )

        print(f"Loaded Target '{self.name}' with {self.mask.sum()} valid pixels")
        if self.chips_mode:
            print(f"  Multi-chip mode: {self.n_chips} chips")

# run "python -m data.targets" in src/ to test
# wl = np.linspace(2000, 2100, 1000)
# fl = np.random.normal(1.0, 0.01, size=1000)
# err = np.full(1000, 0.01)
# target = Target(wl, fl, err, name="CD-35 2722", JD=2459945.5)
# print(target.coords)
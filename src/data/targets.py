"""
Define the target
"""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u


class Target:
    """
    Target object: stores 1D spectrum and basic target information.
    No instrument-specific logic, no normalization or reshaping.
    """

    def __init__(self, wl, fl, err, name, JD, ra=None, dec=None, color="limegreen"):
        # ----- spectrum information, all 1D array -----
        self.wl = np.asarray(wl)
        self.fl = np.asarray(fl)
        self.err = np.asarray(err)
        # check shapes
        if not (self.wl.shape == self.fl.shape == self.err.shape):
            raise ValueError("wl, fl, and err must have the same shape")
        # check 1D
        if self.wl.ndim != 1:
            raise ValueError("Target expects 1D flattened spectra")

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

        # ----- Mask for valid pixels -----
        self.mask = np.isfinite(self.fl)

        print(f"Loaded Target '{self.name}' with {self.mask.sum()} valid pixels")

# run "python -m data.targets" in src/ to test
# wl = np.linspace(2000, 2100, 1000)
# fl = np.random.normal(1.0, 0.01, size=1000)
# err = np.full(1000, 0.01)
# target = Target(wl, fl, err, name="CD-35 2722", JD=2459945.5)
# print(target.coords)
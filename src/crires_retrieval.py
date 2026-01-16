"""
CRIRES+ exoplanet atmosphere retrieval framework.

This module provides a lightweight retrieval pipeline based on
petitRADTRANS (pRT3) and PyMultiNest, designed for high-resolution
emission spectroscopy with CRIRES+ data.

Features
--------
- Fixed wavelength setting (CRIRES+/K2166 by default)
- Parametric temperature-pressure profile (knot-based)
- Log-likelihood following Ruffio et al. (2019)
- Free-chemistry model compatible with pRT3
- No cloud model (clear atmosphere)

Notes
-----
This file is intended as a runnable entry point for CRIRES+ data.
Most physical models (atmosphere, chemistry, temperature) should be
implemented in dedicated submodules.

Authors
-------
Original version: Natalie Grasser  
Modified by: Jiacheng Peng  
Refactored by: Chenyang Ji (2025-12-22)
"""

# crires_retrieval.py
from core.paths import setup_prt_path, SRC_DIR
from retrieval.parameters import Parameters
from data.loaders import load_crires_dat
from data.preprocessing import select_order_and_flatten
from data.targets import Target
from retrieval.retrieval import Retrieval


def main():
    """Main function to run the retrieval with CRIRES+ data."""
    # 1. set up pRT input data path
    setup_prt_path()

    # 2. load CRIRES+ spectrum
    wave, flux, err = load_crires_dat(  # micron -> nm
        target='CD-35_2722',
        night="2022-12-31",
        savename="starA",
        n_orders=7,
        n_dets=3,
        n_pixels=2048,
    )

    # 3. select order and detector, flatten to 1D
    wave_norm, flux_norm, err_norm = select_order_and_flatten(
        wave, flux, err,
        orders=[5],
        dets=[0],
        normalize=True
    )

    # 4. Create target object
    target = Target(
        wl=wave_norm, fl=flux_norm, err=err_norm,
        name="CD-35 2722", JD=2459945.58464374,
        ra="06h09m19.2081174720s", dec="-35d49m31.065774636s"
    )

    # 5. load parameters & run retrieval
    parameters = Parameters(config_file=SRC_DIR / "config" / "config_example.py", debug=False)

    retrieval = Retrieval(
        parameters=parameters,
        target=target,
        N_live_points=200,
        evidence_tolerance=0.5
    )

    retrieval.run_retrieval()


if __name__ == '__main__':
    main()

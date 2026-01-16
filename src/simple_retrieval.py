"""
Simple exoplanet atmosphere retrieval framework.

This module provides a lightweight retrieval pipeline based on
petitRADTRANS (pRT3) and PyMultiNest, designed for high-resolution
emission spectroscopy with simple two-column data files.

Features
--------
- Simple two-column data file input (wavelength, flux)
- No order/detector selection needed
- Parametric temperature-pressure profile (knot-based)
- Log-likelihood following Ruffio et al. (2019)
- Free-chemistry model compatible with pRT3
- No cloud model (clear atmosphere)

Notes
-----
This file is intended as a runnable entry point for simple data files
that don't require instrument-specific processing (e.g., CRIRES+ order/detector).
Most physical models (atmosphere, chemistry, temperature) should be
implemented in dedicated submodules.

Authors
-------
Original version: Natalie Grasser  
Modified by: Jiacheng Peng  
Refactored by: Chenyang Ji (2025-12-22)
"""

# simple_retrieval.py
# Set thread limits BEFORE importing numpy or other libraries that use threading
# from utils.system import setup_thread_limits
# setup_thread_limits()

from pathlib import Path
from core.paths import setup_prt_path, CONFIG_DIR, DATA_DIR
from retrieval.parameters import Parameters
from data.loaders import load_simple_dat
from data.targets import Target
from retrieval.retrieval import Retrieval


def main():
    """Main function to run the retrieval with simple data file."""
    # 1. set up pRT input data path
    setup_prt_path()
    order_detector = "5_whole"

    # 2. load simple data file
    # File format: wavelength [micron], flux [arbitrary units, typically normalized], optional error
    data_file = DATA_DIR / "simple" / order_detector / "example_SNR300.dat"
    wave, flux, err = load_simple_dat(filepath=data_file, SNR=300)  # Use SNR to calculate error = flux / SNR

    # 3. Create target object
    # Using CD-35 2722 as default target
    target = Target(
        wl=wave, fl=flux, err=err,
        name="CD-35 2722",
        JD=2459945.58464374,  # Change to your observation JD if needed
        ra="06h09m19.2081174720s",
        dec="-35d49m31.065774636s"
    )

    # 4. load parameters & run retrieval
    parameters = Parameters(config_file=CONFIG_DIR / "simple" / "config_example.py", debug=False)

    retrieval = Retrieval(
        parameters=parameters,
        target=target,
        N_live_points=200,
        evidence_tolerance=0.5,
        output_subdir=f'{order_detector}_simple'  # Output to output/retrievals/{output_subdir}/N{...}_ev{...}/
    )

    retrieval.run_retrieval()


if __name__ == '__main__':
    main()

"""
CRIRES+ exoplanet atmosphere retrieval framework with multi-chip support.

This is an example of how to use the new chips_mode feature to handle
discontinuous wavelength ranges (e.g., gaps between orders).

Authors
-------
Chenyang Ji (2025-12-22)
"""

from typing import Any


from core.paths import setup_prt_path, CONFIG_DIR
from retrieval.parameters import Parameters
from data.loaders import load_crires_dat
from data.preprocessing import select_orders_chips  # New function for chips mode
from data.targets import Target
from retrieval.retrieval import Retrieval


def main():
    """Main function to run the retrieval with CRIRES+ data in chips mode."""
    # 1. set up pRT input data path
    setup_prt_path()
    # prefix_crires = "5_012"
    prefix_crires = "whole_chips"
    prefix_retrieval = "fastchem_eqchem_R500"
    normalize_flag = True
    normalize_method = 'low-resolution'  # Options: 'simplistic_normalization', 'low-resolution', 'median_highpass', 'gaussian_lfp', 'savgol_lfp'

    date = "2023-01-03"  # observation night; change this to switch between nights
    JD_per_night = {
        "2022-12-31": 2459945.58464374,
        "2023-01-03": 2459948.77354145,
    }
    JD = JD_per_night[date]

    # 2. load CRIRES+ spectrum
    wave, flux, err = load_crires_dat(  # micron -> nm
        target='CD-35_2722',
        night=date,
        savename="starA",
        n_orders=7,
        n_dets=3,
        n_pixels=2048,
    )

    # 3. select orders and detectors, KEEP CHIP STRUCTURE (not flattened)
    wave_chips, flux_chips, err_chips, wave_ranges_chips = select_orders_chips(
        wave, flux, err,
        orders=[2,3,4,5,6],
        # orders=[5],
        dets=[0,1,2],
        # dets=[1],
        normalize=normalize_flag,
        normalize_method=normalize_method,
        remove_metal_lines=False,  # keep metal line regions for fastchem calculation
        concatenate_dets=True,  # get low-res normalization using all dets per order
    )

    # # plot the normalized spectrum
    # import matplotlib.pyplot as plt
    # import numpy as np
    # # print(np.shape(wave_chips), np.shape(flux_chips), np.shape(err_chips))
    # # print(np.shape(wave[np.ix_([4],[1])][0][0]), np.shape(flux[np.ix_([4],[1])]), np.shape(err[np.ix_([4],[1])]))
    # plt.figure(figsize=(10, 5))
    # for i in range(len(wave_chips)):
    #     plt.errorbar(
    #             wave_chips[i], flux_chips[i], yerr=err_chips[i],
    #             fmt='o', alpha=0.8, markersize=0.5, elinewidth=0.5,
    #         )
    # plt.savefig(f"CD-35_2722_spectrum_{prefix_crires}.pdf")

    # 4. Create target object in chips_mode
    target = Target(
        wl=wave_chips, fl=flux_chips, err=err_chips,
        name="CD-35 2722", JD=JD,
        ra="06h09m19.2081174720s", dec="-35d49m31.065774636s",
        chips_mode=True,  # Enable chips mode
        chips_per_order=len([0, 1, 2]),  # number of detectors per order, provide this if you want to normalize by order instead of by det/chip
    )
    # ra_value = target.coords.ra.value if target.coords.ra is not None else 0.0
    # dec_value = target.coords.dec.value if target.coords.dec is not None else 0.0
    # from PyAstronomy.pyasl import helcorr
    # v_bary, _ = helcorr(
    #     obs_long=-70.40,
    #     obs_lat=-24.62,
    #     obs_alt=2635,
    #     ra2000=ra_value,
    #     dec2000=dec_value,
    #     jd=target.JD,
    # )
    # print(v_bary)

    # 5. load parameters & run retrieval
    parameters = Parameters(config_file=CONFIG_DIR / "CD-35_2722" / date / f"config_starA_{prefix_retrieval}.py", debug=False)

    retrieval = Retrieval(
        parameters=parameters,
        target=target,
        N_live_points=200,
        evidence_tolerance=0.5,
        output_subdir=f"CD-35_2722/{date}/starA/{prefix_crires}/{prefix_retrieval}",
        normalize=normalize_flag,  # Must match data normalization: each chip normalized independently
        normalize_method=normalize_method,  # Must match data normalization method
        star_mode=True
    )

    # Set resume=False to start fresh if you get resume file errors
    retrieval.run_retrieval(resume=True)  # Change to False to start fresh


if __name__ == '__main__':
    main()

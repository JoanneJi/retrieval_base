"""
Plotting utilities for retrieval results.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import corner


def cornerplot(posterior, param_mathtext, color="limegreen", output_path=None, callback_label=""):
    """
    Create corner plot of posterior samples.
    
    Args:
        posterior (np.ndarray): Posterior samples, shape (n_samples, n_params)
        param_mathtext (dict): Dictionary mapping parameter keys to LaTeX labels
        color (str): Color for the plot
        output_path (str, optional): Path to save the plot. If None, plot is not saved.
        callback_label (str): Label prefix for output filename (e.g., "live_", "final_")
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if posterior is None or len(posterior) == 0:
        warnings.warn("Posterior is empty, cannot create corner plot.")
        return None
    
    # Check if we have enough samples for corner plot
    # corner.corner() requires n_samples >= n_params
    n_samples, n_params = posterior.shape[0], posterior.shape[1]
    if n_samples < n_params:
        warnings.warn(
            f"Not enough samples ({n_samples}) for {n_params} parameters. "
            f"Cannot create corner plot. Need at least {n_params} samples."
        )
        return None
    
    labels = list(param_mathtext.values())
    n = len(labels)
    
    # Additional check: labels count should match parameter count
    if n != n_params:
        warnings.warn(
            f"Number of labels ({n}) does not match number of parameters ({n_params}). "
            f"Using first {min(n, n_params)} labels."
        )
        labels = labels[:n_params]
    
    fig = plt.figure(figsize=(n, n), dpi=200)
    corner.corner(
        posterior,
        labels=labels,
        color=color,
        fill_contours=True,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 10},
        fig=fig,
        quiet=True,
    )
    
    if output_path is not None:
        fig.savefig(
            output_path / f"{callback_label}corner.pdf",
            bbox_inches="tight",
        )
    
    plt.close(fig)
    return fig


def plot_spectrum(
    data_wave,
    data_flux,
    model_flux,
    data_err=None,
    mask=None,
    color="limegreen",
    output_path=None,
    callback_label="",
    title=None,
    xlim=None,
    figsize=(10, 4),
    dpi=200,
    residual_flag=False
):
    """
    Plot data vs model spectrum comparison.
    
    Args:
        data_wave (np.ndarray): Data wavelength array [nm]
        data_flux (np.ndarray): Data flux array
        model_flux (np.ndarray): Model flux array (same length as data_wave)
        data_err (np.ndarray, optional): Data error array
        mask (np.ndarray, optional): Boolean mask for valid data points
        color (str): Color for model spectrum. Default: "limegreen"
        output_path (pathlib.Path, optional): Path to save the plot
        callback_label (str): Label prefix for output filename (e.g., "live_", "final_")
        title (str, optional): Plot title
        xlim (tuple, optional): Wavelength range to plot (min, max) in nm
        figsize (tuple): Figure size (width, height). Default: (10, 4)
        dpi (int): Figure resolution. Default: 200
        residual_flag (bool): If True, add a residual subplot below the spectrum plot. Default: False
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Apply mask if provided
    if mask is not None:
        data_wave = data_wave[mask]
        data_flux = data_flux[mask]
        model_flux = model_flux[mask]
        if data_err is not None:
            data_err = data_err[mask]
    
    # Create figure with subplots if residual_flag is True
    if residual_flag:
        # Adjust figsize to accommodate two subplots
        fig, axes = plt.subplots(
            2, 1,
            figsize=(figsize[0], figsize[1] * 1.5),
            dpi=dpi,
            sharex=True,
            gridspec_kw={'wspace': 0, 'hspace': 0}
        )
        ax = axes[0]  # Top subplot for spectrum
        ax_res = axes[1]  # Bottom subplot for residuals
    else:
        # Create single figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax_res = None
    
    # Plot data
    if data_err is not None:
        ax.errorbar(
            data_wave,
            data_flux,
            yerr=data_err,
            fmt='o',
            markersize=0.5,
            elinewidth=0.4,
            alpha=0.7,
            color='black',
            label='Data',
            zorder=1
        )
    else:
        ax.scatter(
            data_wave,
            data_flux,
            s=0.5,
            alpha=0.7,
            color='black',
            label='Data',
            zorder=1
        )
    
    # Plot model
    ax.plot(
        data_wave,
        model_flux,
        color=color,
        linewidth=1.5,
        alpha=0.6,
        label='Model',
        zorder=2
    )
    
    # Set labels and title for spectrum plot
    if not residual_flag:
        ax.set_xlabel('Wavelength [nm]', fontsize=12)
        ax.set_ylabel('Normalized Flux', fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=12)
    elif callback_label:
        ax.set_title(f'Spectrum Comparison ({callback_label.rstrip("_")})', fontsize=12)
    else:
        ax.set_title('Spectrum Comparison', fontsize=12)
    
    # Set xlim if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    
    # Add legend
    ax.legend(loc='best', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Plot residuals if residual_flag is True
    if residual_flag:
        # Calculate residuals: data - model
        residuals = data_flux - model_flux
        
        # Plot residuals as scatter points with error bars
        ax_res.errorbar(
            data_wave,
            residuals,
            yerr=data_err,
            fmt='o',
            markersize=0.6,
            elinewidth=0.1,
            alpha=0.3,
            color='black',
            zorder=1
        )
        
        # Plot residual=0 line in black
        ax_res.axhline(
            y=0,
            color='black',
            linewidth=1.8,
            linestyle='--',
            zorder=2
        )
        
        # Set labels for residual plot
        ax_res.set_xlabel('Wavelength [nm]', fontsize=12)
        ax_res.set_ylabel('Residuals', fontsize=12)
        
        # Add grid
        ax_res.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path is not None:
        fig.savefig(
            output_path / f"{callback_label}spectrum.pdf",
            bbox_inches="tight",
            dpi=dpi
        )
    
    plt.close(fig)
    return fig


def plot_tp_profile(
    temperature,
    pressure,
    color="limegreen",
    output_path=None,
    callback_label="",
    title=None,
    figsize=(6, 5),
    dpi=200,
    tp_history=None,
    knots_temperature=None,
    knots_pressure=None,
    knots_error_positive=None,
    knots_error_negative=None,
    interp_mode='cubic'
):
    """
    Plot temperature-pressure (TP) profile.
    
    Args:
        temperature (np.ndarray): Temperature array [K]
        pressure (np.ndarray): Pressure array [bar]
        color (str): Color for the TP profile line. Default: "limegreen"
        output_path (pathlib.Path, optional): Path to save the plot
        callback_label (str): Label prefix for output filename (e.g., "live_", "final_")
        title (str, optional): Plot title
        figsize (tuple): Figure size (width, height). Default: (6, 8)
        dpi (int): Figure resolution. Default: 200
        tp_history (list, optional): List of (temperature, pressure) tuples for historical profiles.
            Used for live plotting with gradient colors. Default: None
        knots_temperature (np.ndarray, optional): Temperature values at knots [K]
        knots_pressure (np.ndarray, optional): Pressure values at knots [bar]
        knots_error_positive (np.ndarray, optional): Positive error on temperature at knots [K]
        knots_error_negative (np.ndarray, optional): Negative error on temperature at knots [K]
        interp_mode (str): Interpolation mode for error range ('cubic', 'linear', 'quadratic'). 
            Should match the TP profile interpolation mode. Default: 'cubic'
    
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Validate inputs
    if temperature is None or pressure is None:
        warnings.warn("Temperature or pressure is None, cannot create TP profile plot.")
        return None
    
    if len(temperature) != len(pressure):
        raise ValueError(f"Temperature and pressure arrays must have the same length. "
                        f"Got {len(temperature)} and {len(pressure)}.")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot historical TP profiles if provided (for live plotting)
    if tp_history is not None and len(tp_history) > 0 and callback_label == "live_":
        n_history = len(tp_history)
        # Use dark red gradient: older profiles are lighter, newer are darker
        # Color range: from very light (0.15 alpha) to darker (0.7 alpha)
        for i, (temp_hist, press_hist) in enumerate(tp_history):
            # Calculate alpha: older (smaller i) -> lighter, newer (larger i) -> darker
            # Normalize to [0.15, 0.7] range for subtle gradient effect
            alpha = 0.15 + 0.55 * (i + 1) / n_history
            # Use dark red color with varying alpha
            ax.plot(
                temp_hist,
                press_hist,
                color='darkred',
                linewidth=0.8,
                alpha=alpha,
                zorder=1
            )
    
    # Plot knots error range if provided (live plot also)
    if (knots_temperature is not None and knots_pressure is not None and 
        knots_error_positive is not None and knots_error_negative is not None):
        # Calculate upper and lower bounds for error range
        T_upper = knots_temperature + knots_error_positive
        T_lower = knots_temperature - knots_error_negative
        
        # Sort by pressure (ascending) to ensure proper interpolation
        sort_idx = np.argsort(knots_pressure)
        P_sorted = knots_pressure[sort_idx]
        T_upper_sorted = T_upper[sort_idx]
        T_lower_sorted = T_lower[sort_idx]
        T_knots_sorted = knots_temperature[sort_idx]  # Also sort actual knots for scatter
        
        # Interpolate error bounds onto the full pressure grid using the same method as TP profile
        # Use log10(pressure) for interpolation (consistent with TP profile)
        from scipy.interpolate import interp1d, CubicSpline
        
        log_P_sorted = np.log10(P_sorted)
        log_pressure = np.log10(pressure)
        
        if interp_mode == 'cubic':
            # Use CubicSpline (same as TP profile)
            T_upper_interp_func = CubicSpline(log_P_sorted, T_upper_sorted)
            T_lower_interp_func = CubicSpline(log_P_sorted, T_lower_sorted)
            T_upper_interp = T_upper_interp_func(log_pressure)
            T_lower_interp = T_lower_interp_func(log_pressure)
        elif interp_mode == 'quadratic':
            T_upper_interp_func = interp1d(
                log_P_sorted, T_upper_sorted,
                kind='quadratic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            T_lower_interp_func = interp1d(
                log_P_sorted, T_lower_sorted,
                kind='quadratic',
                bounds_error=False,
                fill_value='extrapolate'
            )
            T_upper_interp = T_upper_interp_func(log_pressure)
            T_lower_interp = T_lower_interp_func(log_pressure)
        else:  # linear or default
            # Use linear interpolation
            T_upper_interp = np.interp(log_pressure, log_P_sorted, T_upper_sorted)
            T_lower_interp = np.interp(log_pressure, log_P_sorted, T_lower_sorted)
        
        # scatter the temperature values at knots (use sorted values to match TP profile)
        ax.scatter(
            T_knots_sorted,
            P_sorted,
            color=color,
            s=10,
            zorder=2
        )
        
        # Fill the error range with light green color
        ax.fill_betweenx(
            pressure,
            T_lower_interp,
            T_upper_interp,
            color=color,
            edgecolor='none',
            alpha=0.1,
            zorder=0,
            # label='Error Range'
            )
    
    # Plot current TP profile
    ax.plot(
        temperature,
        pressure,
        color=color,
        linewidth=2.0,
        label='TP Profile',
        zorder=2
    )
    
    # Set labels
    ax.set_xlabel('Temperature [K]', fontsize=12)
    ax.set_ylabel('Pressure [bar]', fontsize=12)
    
    # Set title
    if title is not None:
        ax.set_title(title, fontsize=12)
    elif callback_label:
        ax.set_title(f'TP Profile ({callback_label.rstrip("_")})', fontsize=12)
    else:
        ax.set_title('TP Profile', fontsize=12)
    
    # Set y-axis to log scale and invert (pressure: small at top, large at bottom)
    ax.set_yscale('log')
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(True, alpha=0.3, which='both')
    
    # Add legend
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path is not None:
        fig.savefig(
            output_path / f"{callback_label}tp_profile.pdf",
            bbox_inches="tight",
            dpi=dpi
        )
    
    plt.close(fig)
    return fig


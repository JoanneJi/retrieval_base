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
    
    labels = list(param_mathtext.values())
    n = len(labels)
    
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
    dpi=200
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot data
    if data_err is not None:
        ax.errorbar(
            data_wave,
            data_flux,
            yerr=data_err,
            fmt='o',
            markersize=0.5,
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
        linewidth=1.0,
        label='Model',
        zorder=2
    )
    
    # Set labels and title
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
    figsize=(6, 8),
    dpi=200,
    tp_history=None
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


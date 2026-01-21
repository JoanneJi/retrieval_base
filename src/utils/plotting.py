"""
Plotting utilities for retrieval results.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
import corner


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "font.size": 12,
})


def _set_axes_style(ax, title=None, xlabel=None, ylabel=None):
    """
    Set consistent axes style for all plots.
    
    Args:
        ax: matplotlib axes object
        title (str, optional): Title text
        xlabel (str, optional): X-axis label text
        ylabel (str, optional): Y-axis label text
    """
    # Set border linewidth to 2.0
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    
    # Set major ticks to point inward
    ax.tick_params(axis='both', which='major', direction='in', 
                   length=4, width=2, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', 
                   length=2, width=1)
    
    # Set grid style
    ax.grid(True, linestyle=':', linewidth=0.8, color='lightgrey', zorder=0)
    
    # Set labels with bold font
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=12, weight='bold')
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=12, weight='bold')
    if title is not None:
        ax.set_title(title, fontsize=14, weight='bold')
    
    # Set tick label fontsize
    ax.tick_params(axis='both', labelsize=10)


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
    figsize=(15, 2),
    dpi=300,
    residual_flag=False,
    ylim=None
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
        figsize (tuple): Figure size (width, height). Default: (15, 2)
        dpi (int): Figure resolution. Default: 300
        residual_flag (bool): If True, plot residuals at y=0 position. Default: False
        ylim (tuple, optional): Y-axis range (min, max). Default: (-0.2, 1.15)
    
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
    
    # Sort by wavelength to ensure proper gap detection
    sort_idx = np.argsort(data_wave)
    data_wave = data_wave[sort_idx]
    data_flux = data_flux[sort_idx]
    model_flux = model_flux[sort_idx]
    if data_err is not None:
        data_err = data_err[sort_idx]
    
    # Detect wavelength gaps to determine subplot structure
    # Gap threshold: if gap > (length of previous segment + length of next segment), create new subplot
    gaps = np.diff(data_wave)
    segment_starts = [0]  # Start indices of each continuous segment
    
    if len(gaps) > 0:
        i = 0
        while i < len(gaps):
            gap = gaps[i]
            if gap <= 0:
                i += 1
                continue
            
            # Current segment start
            seg_start = segment_starts[-1]
            # Length of previous segment (from segment start to current position)
            prev_seg_length = data_wave[i] - data_wave[seg_start]
            
            # Find next significant gap to estimate next segment length
            next_seg_length = 0
            if i + 1 < len(data_wave):
                # Look ahead to find next gap or end
                for j in range(i + 1, len(gaps)):
                    if gaps[j] > np.median(gaps) * 2:  # Next significant gap
                        next_seg_length = data_wave[j] - data_wave[i + 1]
                        break
                else:
                    # No more gaps, use remaining data as next segment
                    next_seg_length = data_wave[-1] - data_wave[i + 1]
            
            # If gap is larger than sum of adjacent segment lengths, create new subplot
            if prev_seg_length > 0 and next_seg_length > 0:
                if gap > (prev_seg_length + next_seg_length):
                    segment_starts.append(i + 1)
            elif prev_seg_length > 0 and gap > prev_seg_length * 2:
                # Fallback: if gap is more than 2x previous segment length
                segment_starts.append(i + 1)
            
            i += 1
    
    segment_ends = segment_starts[1:] + [len(data_wave)]
    
    # Create subplots based on detected segments
    n_subplots = len(segment_starts)
    
    # Set default ylim
    if ylim is None:
        ylim = (-0.2, 1.5)
    
    # If residual_flag is True, use broken axis to skip middle blank region
    if residual_flag:

        # Calculate global data point range for upper axis
        # Get min and max of all data points (flux)
        flux_min = np.nanmin(data_flux)
        flux_max = np.nanmax(data_flux)
        flux_range = flux_max - flux_min
        # Expand by 0.02 (2% of range) on each side
        padding = flux_range * 0.02
        ylim_upper = (flux_min - padding, flux_max + padding)
        
        # Lower part: residual range (around -0.1 to 0.1)
        ylim_lower = (-0.15, 0.15)
        
        # Create figure
        fig = plt.figure(figsize=(figsize[0], figsize[1] * n_subplots), dpi=dpi)
        
        axes_upper = []
        axes_lower = []
        
        # Calculate positions for each subplot
        total_height = 0.9  # Total height available (leaving margins)
        subplot_height = total_height / n_subplots
        bottom_margin = 0.1
        # Reduce gap between subplots to make them closer
        gap_between_subplots = 0.14 / n_subplots if n_subplots > 1 else 0
        
        for subplot_idx in range(n_subplots):
            # Calculate vertical position for this subplot
            subplot_bottom = bottom_margin + (n_subplots - 1 - subplot_idx) * (subplot_height + gap_between_subplots)
            
            # Upper axis: takes top 60% of subplot space (for spectrum)
            upper_bottom = subplot_bottom + subplot_height * 0.4
            upper_height = subplot_height * 0.6
            ax_upper = fig.add_axes((0.15, upper_bottom, 0.75, upper_height))
            
            # Lower axis: takes bottom 38% of subplot space (for residuals)
            lower_bottom = subplot_bottom
            lower_height = subplot_height * 0.38
            ax_lower = fig.add_axes((0.15, lower_bottom, 0.75, lower_height))
            
            axes_upper.append(ax_upper)
            axes_lower.append(ax_lower)
        
        axes = list(zip(axes_upper, axes_lower))
    else:
        # Normal plotting without broken axis
        if n_subplots > 1:
            fig, axes = plt.subplots(
                n_subplots, 1,
                figsize=(figsize[0], figsize[1] * n_subplots),
                dpi=dpi,
                sharey=True,
                gridspec_kw={'wspace': 0, 'hspace': 0.2}
            )
            # Ensure axes is always a list
            if n_subplots == 1:
                axes = [axes]
            else:
                axes = list(axes)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            axes = [ax]
    
    # Plot each segment
    for subplot_idx, (start_idx, end_idx) in enumerate(zip(segment_starts, segment_ends)):
        if residual_flag:
            ax_upper, ax_lower = axes[subplot_idx]
        else:
            ax = axes[subplot_idx]
        
        # Extract segment data
        seg_wave = data_wave[start_idx:end_idx]
        seg_flux = data_flux[start_idx:end_idx]
        seg_model = model_flux[start_idx:end_idx]
        seg_err = data_err[start_idx:end_idx] if data_err is not None else None
        
        # Calculate residuals if residual_flag is True
        if residual_flag:
            residuals = seg_flux - seg_model
            
            # Plot on upper axis (spectrum)
            # Plot data points with errorbar
            if seg_err is not None:
                ax_upper.errorbar(
                    seg_wave,
                    seg_flux,
                    yerr=seg_err,
                    fmt='o',
                    markersize=0.8,
                    elinewidth=0.2,
                    capsize=0.2,
                    capthick=0.2,
                    alpha=0.6,
                    color='k',
                    label='Data' if subplot_idx == 0 else '',
                    zorder=1
                )
            else:
                ax_upper.scatter(
                    seg_wave,
                    seg_flux,
                    s=0.8,
                    alpha=0.6,
                    color='k',
                    label='Data' if subplot_idx == 0 else '',
                    zorder=1
                )
            
            # Plot model line on upper axis
            ax_upper.plot(
                seg_wave,
                seg_model,
                color=color,
                linewidth=2,
                alpha=0.85,
                label='Model' if subplot_idx == 0 else '',
                zorder=2
            )
            
            # Plot residuals on lower axis
            ax_lower.scatter(
                seg_wave,
                residuals,
                s=0.3,
                alpha=0.6,
                color='gray',
                label='Residuals' if subplot_idx == 0 else '',
                zorder=1
            )
            
            # Set ylim for broken axis
            ax_upper.set_ylim(ylim_upper)
            ax_lower.set_ylim(ylim_lower)
            
            # Hide the gap region
            ax_upper.spines['bottom'].set_visible(False)
            ax_lower.spines['top'].set_visible(False)
            ax_upper.xaxis.tick_top()
            ax_upper.tick_params(labeltop=False)
            ax_lower.xaxis.tick_bottom()
            
            # Set xlim for both axes first
            wave_min = seg_wave.min()
            wave_max = seg_wave.max()
            wave_range = wave_max - wave_min
            padding = wave_range * 0.02
            xlim_seg = (wave_min - padding, wave_max + padding)
            ax_upper.set_xlim(xlim_seg)
            ax_lower.set_xlim(xlim_seg)
            
            # Add broken axis marks (parallel diagonal lines)
            # Line length as fraction of x-axis range
            line_length_frac = 0.005  # 1.5% of x-axis range
            line_length = (xlim_seg[1] - xlim_seg[0]) * line_length_frac
            
            # To ensure parallel lines, use the same absolute height for both axes
            y_range_upper = ylim_upper[1] - ylim_upper[0]
            line_height_frac = 0.025  # 2% of reference y-axis range
            line_height = y_range_upper * line_height_frac

            line_height_upper = line_height
            line_height_lower = line_height
            
            # Left diagonal: from bottom-left to top-right
            ax_upper.plot([xlim_seg[0] - line_length, xlim_seg[0] + line_length], 
                         [ylim_upper[0] - line_height_upper, ylim_upper[0] + line_height_upper], 
                         'k-', linewidth=1.5, clip_on=False, zorder=10)
            # Right diagonal: from bottom-right to top-left
            ax_upper.plot([xlim_seg[1] - line_length, xlim_seg[1] + line_length], 
                         [ylim_upper[0] - line_height_upper, ylim_upper[0] + line_height_upper], 
                         'k-', linewidth=1.5, clip_on=False, zorder=10)

            # Left diagonal: from top-left to bottom-right
            ax_lower.plot([xlim_seg[0] - line_length, xlim_seg[0] + line_length], 
                         [ylim_lower[1] - line_height_lower, ylim_lower[1] + line_height_lower], 
                         'k-', linewidth=1.5, clip_on=False, zorder=10)
            # Right diagonal: from top-right to bottom-left
            ax_lower.plot([xlim_seg[1] - line_length, xlim_seg[1] + line_length], 
                         [ylim_lower[1] - line_height_lower, ylim_lower[1] + line_height_lower], 
                         'k-', linewidth=1.5, clip_on=False, zorder=10)
            
            # Apply style to axes
            for ax_temp in [ax_upper, ax_lower]:
                # Set border linewidth to 2.0
                for spine in ax_temp.spines.values():
                    spine.set_linewidth(2.0)
                # Set major ticks to point inward
                ax_temp.tick_params(axis='both', which='major', direction='in', 
                                   length=4, width=2, labelsize=10)
                ax_temp.tick_params(axis='both', which='minor', direction='in', 
                                   length=2, width=1)
                # Set x-axis tick spacing: major every 5, minor every 1
                ax_temp.xaxis.set_major_locator(MultipleLocator(5))
                ax_temp.xaxis.set_minor_locator(MultipleLocator(1))
                # Set grid style (only on major ticks)
                ax_temp.grid(True, linestyle='--', linewidth=1, zorder=0, alpha=0.3, which='major')
            
            # Set title (only on first subplot)
            if subplot_idx == 0:
                if title is not None:
                    ax_upper.set_title(title, fontsize=14, weight='bold')
                elif callback_label:
                    ax_upper.set_title(f'Spectrum Comparison ({callback_label.rstrip("_")})', fontsize=14, weight='bold')
                else:
                    ax_upper.set_title('Spectrum Comparison', fontsize=14, weight='bold')
            
            # Set xlabel and ylabel at figure level (only once, after all subplots are drawn)
            if subplot_idx == len(axes) - 1:  # Last subplot
                # Get the position of the first upper axis to align ylabel with axes left edge
                ax_upper_first = axes[0][0]  # First subplot's upper axis
                bbox = ax_upper_first.get_position()  # Get axes position in figure coordinates
                axes_left = bbox.x0  # Left edge of axes
                
                # Set xlabel at figure level (bottom center) with bold
                fig.text(0.5, 0.02, 'Wavelength [nm]', fontsize=12, weight='bold',
                        ha='center', va='bottom')
                # Set ylabel at figure level (aligned with axes left edge, slightly to the left) with bold
                fig.text(axes_left - 0.03, 0.5, 'Normalized Flux', fontsize=12, weight='bold',
                        rotation=90, va='center', ha='center')
            
            # Add legend only on first subplot (no border, horizontal layout)
            # Collect handles and labels from both upper and lower axes to include residuals
            if subplot_idx == 0:
                handles_upper, labels_upper = ax_upper.get_legend_handles_labels()
                handles_lower, labels_lower = ax_lower.get_legend_handles_labels()
                # Combine handles and labels from both axes
                handles_combined = handles_upper + handles_lower
                labels_combined = labels_upper + labels_lower
                # Create legend with all three entries (Data, Model, Residuals)
                legend = ax_upper.legend(handles_combined, labels_combined, loc='best', fontsize=12, 
                                        ncol=3, frameon=False, columnspacing=0.2, 
                                        handlelength=1.0, handletextpad=0.4, markerscale=0.8)
            
        else:
            # Normal plotting without broken axis
            # Plot data points with errorbar
            if seg_err is not None:
                ax.errorbar(
                    seg_wave,
                    seg_flux,
                    yerr=seg_err,
                    fmt='o',
                    markersize=0.8,
                    elinewidth=0.2,
                    capsize=0.2,
                    capthick=0.2,
                    alpha=0.6,
                    color='k',
                    label='Data' if subplot_idx == 0 else '',
                    zorder=1
                )
            else:
                ax.scatter(
                    seg_wave,
                    seg_flux,
                    s=0.8,
                    alpha=0.6,
                    color='k',
                    label='Data' if subplot_idx == 0 else '',
                    zorder=1
                )
            
            # Plot model line
            ax.plot(
                seg_wave,
                seg_model,
                color=color,
                linewidth=2,
                alpha=0.85,
                label='Model' if subplot_idx == 0 else '',
                zorder=2
            )
            
            # Apply style to axes
            # Set border linewidth to 2.0
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            # Set major ticks to point inward
            ax.tick_params(axis='both', which='major', direction='in', 
                           length=4, width=2, labelsize=10)
            ax.tick_params(axis='both', which='minor', direction='in', 
                           length=2, width=1)
            # Set x-axis tick spacing: major every 5, minor every 1
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            # Set grid style (only on major ticks)
            ax.grid(True, linestyle='--', linewidth=1, zorder=0, alpha=0.3, which='major')
            
            # Set title (only on first subplot)
            if subplot_idx == 0:
                if title is not None:
                    ax.set_title(title, fontsize=14, weight='bold')
                elif callback_label:
                    ax.set_title(f'Spectrum Comparison ({callback_label.rstrip("_")})', fontsize=14, weight='bold')
                else:
                    ax.set_title('Spectrum Comparison', fontsize=14, weight='bold')
            
            # Set xlabel and ylabel at figure level (only once, after all subplots are drawn)
            if subplot_idx == len(axes) - 1:  # Last subplot
                # Get the position of the first axis to align ylabel with axes left edge
                ax_first = axes[0]
                bbox = ax_first.get_position()  # Get axes position in figure coordinates
                axes_left = bbox.x0  # Left edge of axes
                
                # Set xlabel at figure level (bottom center) with bold
                fig.text(0.5, 0.02, 'Wavelength [nm]', fontsize=12, weight='bold',
                        ha='center', va='bottom')
                # Set ylabel at figure level (aligned with axes left edge, slightly to the left) with bold
                fig.text(axes_left - 0.03, 0.5, 'Normalized Flux', fontsize=12, weight='bold',
                        rotation=90, va='center', ha='center')
            
            # Set ylim
            ax.set_ylim(ylim)
            
            # Set xlim: use provided xlim if given, otherwise use wavelength range of this segment
            if xlim is not None:
                # If user provided xlim, apply it (typically only to first subplot or all)
                if subplot_idx == 0 or len(axes) == 1:
                    ax.set_xlim(xlim)
            else:
                # Set xlim to the wavelength range of this segment with small padding
                wave_min = seg_wave.min()
                wave_max = seg_wave.max()
                wave_range = wave_max - wave_min
                padding = wave_range * 0.02  # 2% padding on each side
                ax.set_xlim(wave_min - padding, wave_max + padding)
            
            # Add legend only on first subplot (no border, horizontal layout)
            if subplot_idx == 0:
                ax.legend(loc='best', fontsize=12, ncol=2, frameon=False, columnspacing=0.2, handlelength=1.0, handletextpad=0.4, markerscale=0.8)
    
    # Adjust layout: tight_layout for normal mode, manual adjustment for broken axis mode
    if residual_flag:
        # Broken axis mode: manually created axes don't work with tight_layout
        # Use subplots_adjust to control spacing manually
        if n_subplots > 1:
            # Adjust spacing between subplots
            fig.subplots_adjust(hspace=0.2)
    else:
        # Normal mode: use tight_layout, then further adjust hspace if needed
        plt.tight_layout()
        if n_subplots > 1:
            # Further reduce hspace after tight_layout
            fig.subplots_adjust(hspace=0.2)
    
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
    
    # Apply style to axes
    # Set border linewidth to 2.0
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
    # Set major ticks to point inward
    ax.tick_params(axis='both', which='major', direction='in', 
                   length=6, width=1.5, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', 
                   length=3, width=1)
    
    # Set labels with bold font
    ax.set_xlabel('Temperature [K]', fontsize=12, weight='bold')
    ax.set_ylabel('Pressure [bar]', fontsize=12, weight='bold')
    
    # Set title
    if title is not None:
        ax.set_title(title, fontsize=14, weight='bold')
    elif callback_label:
        ax.set_title(f'TP Profile ({callback_label.rstrip("_")})', fontsize=14, weight='bold')
    else:
        ax.set_title('TP Profile', fontsize=14, weight='bold')
    
    # Set y-axis to log scale and invert (pressure: small at top, large at bottom)
    ax.set_yscale('log')
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(True, linestyle='--', linewidth=1, zorder=0, alpha=0.3, which='both')
    
    # Add legend (no border, horizontal layout)
    ax.legend(loc='best', fontsize=10, ncol=2, frameon=False, columnspacing=0.2, handlelength=1.0, handletextpad=0.4, markerscale=0.8)
    
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


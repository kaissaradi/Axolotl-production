# axolotl/plotting.py
"""
Functions for visualizing spike sorting data, including electrical images (EIs),
cluster quality metrics, and diagnostic plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.cm as cm

def plot_ei_waveforms(
    ei: np.ndarray,
    positions: np.ndarray,
    ref_channel: int = None,
    ax: plt.Axes = None,
    # --- Main Visual Controls ---
    scale: float = 2.2,
    use_colormap: bool = True,
    colormap: str = 'viridis',
    # --- Fine-Tuning Parameters ---
    alpha_gamma: float = 0.5,
    alpha_min: float = 0.1,
    alpha_max: float = 0.9,
    linewidth_min: float = 0.75,
    linewidth_max: float = 1.6,
    color: str = 'steelblue',
    highlight_color: str = 'red',
    highlight_alpha: float = 1.0,
    highlight_lw: float = 2.5,
    p2p_q: float = 0.98
):
    """
    Plots a high-clarity electrical image (EI) with dynamic styling.

    Waveform appearance is scaled by amplitude. A perceptual colormap can
    be used to intuitively visualize signal strength across channels.
    """
    # 1. Setup Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_aspect('equal')

    # 2. Prepare Data & Colors
    n_channels, n_samples = ei.shape
    p2p = np.ptp(ei, axis=1)
    
    channel_colors = [color] * n_channels
    if use_colormap:
        cmap = cm.get_cmap(colormap)
        # Log-normalize p2p for better perceptual range
        p2p_norm = np.log1p(p2p)
        if p2p_norm.max() > 0:
            p2p_norm /= p2p_norm.max()
        channel_colors = cmap(p2p_norm)

    # Use a robust measure for vertical scaling
    norm_val = np.quantile(p2p, p2p_q) if np.any(p2p > 0) else 1.0
    if norm_val == 0: norm_val = 1.0

    # 3. Create Time Vector
    x_dists = np.abs(np.diff(np.unique(positions[:, 0])))
    y_dists = np.abs(np.diff(np.unique(positions[:, 1])))
    valid_dists = np.concatenate([d for d in (x_dists, y_dists) if d.size > 0])
    median_dist = np.median(valid_dists) if valid_dists.size > 0 else 20.0
    
    box_width = median_dist * 0.8
    t = np.linspace(-0.5, 0.5, n_samples) * box_width
    
    # 4. Main Plotting Loop
    max_p2p = np.max(p2p) + 1e-9
    for i in range(n_channels):
        x_offset, y_offset = positions[i]
        
        waveform_scaled = (ei[i] / norm_val) * median_dist * scale
        
        relative_amp = p2p[i] / max_p2p
        
        # Apply gamma correction to alpha for better visibility of mid-range spikes
        alpha_scaled = relative_amp ** alpha_gamma
        plot_alpha = alpha_min + (alpha_max - alpha_min) * alpha_scaled
        
        plot_lw = linewidth_min + (linewidth_max - linewidth_min) * relative_amp
        plot_color = channel_colors[i]
        
        # Highlight the reference channel if specified
        if ref_channel is not None and i == ref_channel:
            plot_color = highlight_color
            plot_alpha = highlight_alpha
            plot_lw = highlight_lw
        
        ax.plot(t + x_offset, waveform_scaled + y_offset, 
                color=plot_color, linewidth=plot_lw, alpha=plot_alpha)

    # 5. Set Final Axis Limits
    pad_x = box_width
    pad_y = median_dist * scale
    min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
    min_y, max_y = positions[:, 1].min(), positions[:, 1].max()
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

def plot_array_layout(ei_positions):
    """
    Plots the layout of the electrode array with channel IDs.
    """
    plt.figure(figsize=(12, 4))
    for i in range(ei_positions.shape[0]):
        x_pos, y_pos = ei_positions[i, 0], ei_positions[i, 1]
        plt.text(x_pos, y_pos, str(i), fontsize=8, ha='center', va='center')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Electrode Array Layout')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cluster_diagnostics(
    spike_times,
    sampling_rate,
    dat_path,
    ei_positions,
    h5_path=None,
    triggers_mat_path=None,
    clusters=None,
    template_ei=None
):
    """
    Comprehensive diagnostic plot for a set of clusters, showing EI, ISI, firing rate, and STA.
    This is a simplified version for demonstration.
    """
    if clusters is None:
        clusters = [{'inds': np.arange(len(spike_times))}]
    
    n_clusters = len(clusters)
    fig = plt.figure(figsize=(4 * n_clusters, 8))
    gs = gridspec.GridSpec(3, n_clusters, figure=fig)

    for col, cluster in enumerate(clusters):
        inds = cluster['inds']
        if len(inds) == 0:
            continue

        spike_samples = spike_times[inds]
        spikes_sec = spike_samples / sampling_rate

        # --- EI Calculation and Plot ---
        if 'ei' in cluster:
            ei = cluster['ei']
        else: # Placeholder for EI calculation
            ei = np.random.randn(ei_positions.shape[0], 80) 

        ax_ei = fig.add_subplot(gs[0, col])
        ax_ei.set_title(f"Cluster {col}\n({len(inds)} spikes)")
        plot_ei_waveforms(ei, ei_positions, ax=ax_ei)

        # --- ISI Histogram ---
        ax_isi = fig.add_subplot(gs[1, col])
        if len(spikes_sec) > 1:
            isi = np.diff(spikes_sec) * 1000  # in ms
            ax_isi.hist(isi[isi < 100], bins=100, range=(0, 100))
            ax_isi.set_xlabel("ISI (ms)")
        ax_isi.set_title("ISI")

        # --- Firing Rate ---
        ax_rate = fig.add_subplot(gs[2, col])
        if len(spikes_sec) > 1:
            bins = np.arange(0, spikes_sec.max() + 1, 1)
            rate, _ = np.histogram(spikes_sec, bins=bins)
            ax_rate.plot(bins[:-1], rate)
            ax_rate.set_xlabel("Time (s)")
            ax_rate.set_ylabel("Spikes/sec")
        ax_rate.set_title("Firing Rate")

    plt.tight_layout()
    plt.show()